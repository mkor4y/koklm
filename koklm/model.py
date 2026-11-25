"""
Llama tarzı decoder-only transformer (PyTorch).

Boyut gösterimi:
  B = batch, T = dizi uzunluğu, d = dim, H = kafa, Hkv = KV kafası, dh = d/H, V = sözlük

Akış:
  h_0     = E[token]
  h'      = h + Attn(RMSNorm(h))        (pre-norm + residual)
  h_{l+1} = h' + FFN(RMSNorm(h'))
  logits  = RMSNorm(h_L) W_out          (tied ise W_out = E^T)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

# torch>=2.5: SDPA, GQA'yı K/V'yi kopyalamadan kendisi yapabilir
_SURUM = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
_SDPA_GQA = _SURUM >= (2, 5)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g      (ortalama çıkarma yok, bias yok)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # bf16/fp16'da kararlılık için normu fp32'de hesapla
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


# ---------------------------------------------------------------------------
# RoPE: Rotary Position Embedding
# ---------------------------------------------------------------------------
def _llama3_olcekle(freqs: torch.Tensor, s: dict) -> torch.Tensor:
    """
    Llama 3.1+ uzun bağlam ölçeklemesi. Dalga boyu  λ = 2π / ω.
      λ < orijinal_bağlam / high_freq_factor  -> dokunma (yüksek frekans, yerel bilgi)
      λ > orijinal_bağlam / low_freq_factor   -> ω / factor (düşük frekans, uzak bağlam)
      arada                                    -> iki uç arasında yumuşak geçiş
    """
    factor, lo, hi = s["factor"], s["low_freq_factor"], s["high_freq_factor"]
    eski = s["original_max_position_embeddings"]
    dalga = 2 * math.pi / freqs
    yumusak = (eski / dalga - lo) / (hi - lo)
    ara = (1 - yumusak) * freqs / factor + yumusak * freqs
    out = torch.where(dalga > eski / lo, freqs / factor, freqs)
    orta = (dalga >= eski / hi) & (dalga <= eski / lo)
    return torch.where(orta, ara, out)


def rope_tablosu(head_dim: int, max_seq_len: int, theta: float,
                 scaling: Optional[dict] = None):
    """
    ω_i = θ^(-2i/dh),  i = 0..dh/2-1      Konum m için açı: m·ω_i
    Dönüş: cos, sin  şekil (max_seq_len, dh/2), fp32
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if scaling is not None:
        freqs = _llama3_olcekle(freqs, scaling)
    m = torch.arange(max_seq_len, dtype=torch.float32)
    aci = torch.outer(m, freqs)
    return torch.cos(aci), torch.sin(aci)


def rope_uygula(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, kafa, dh);  cos, sin: (T, dh/2)
    Ardışık (x_2i, x_2i+1) çiftlerini m·ω_i açısıyla döndürür:
        [x'_2i  ]   [cos  -sin] [x_2i  ]
        [x'_2i+1] = [sin   cos] [x_2i+1]
    Sonuç: <R(m)q, R(n)k> = q^T R(n-m) k  -> skor sadece göreli konuma bağlı.
    (Meta'nın orijinal "interleaved" düzeni. HF checkpoint'leri "rotate_half" kullanır.)
    """
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = xf[..., 0], xf[..., 1]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.type_as(x)


# ---------------------------------------------------------------------------
# Attention (GQA)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """
    Grouped-Query Attention:
      Q = xWq -> (B,T,H,dh)      K = xWk, V = xWv -> (B,T,Hkv,dh)
      Attn = softmax(Q K^T / sqrt(dh) + M) V,   M_ij = -inf (j > i)
    Hesap F.scaled_dot_product_attention ile yapılır -> GPU'da FlashAttention çekirdeği.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.head_dim, self.n_rep = cfg.head_dim, cfg.n_rep
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        # KV-cache (sadece üretimde)
        self.cache_boyut: Optional[tuple] = None
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(self, x, cos, sin, start_pos: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = rope_uygula(q, cos, sin)
        k = rope_uygula(k, cos, sin)

        if self.cache_boyut is not None:
            if self.cache_k is None:   # ilk kullanımda, gerçek dtype/cihazla ayır
                mb, ms = self.cache_boyut
                shape = (mb, ms, self.n_kv_heads, self.head_dim)
                self.cache_k = torch.zeros(shape, dtype=k.dtype, device=k.device)
                self.cache_v = torch.zeros(shape, dtype=v.dtype, device=v.device)
            self.cache_k[:B, start_pos:start_pos + T] = k
            self.cache_v[:B, start_pos:start_pos + T] = v
            k = self.cache_k[:B, :start_pos + T]
            v = self.cache_v[:B, :start_pos + T]

        # (B, kafa, T, dh)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # GQA: her K/V kafasını n_rep Q kafası paylaşır.
        # Yeni torch'ta SDPA bunu kopyasız yapar; eskisinde K/V'yi açıkça çoğalt.
        gqa = {}
        if self.n_rep > 1:
            if _SDPA_GQA:
                gqa = {"enable_gqa": True}
            else:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

        # Nedensel maske
        if start_pos == 0:
            mask, causal = None, T > 1
        elif T == 1:
            mask, causal = None, False            # tek yeni token her şeyi görebilir
        else:                                     # cache varken çok token: j <= start_pos + i
            mask = torch.ones(T, start_pos + T, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=start_pos)
            causal = False

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0, **gqa,
        )
        o = o.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(o)


# ---------------------------------------------------------------------------
# FFN (SwiGLU)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """FFN(x) = ( SiLU(x W1) ⊙ (x W3) ) W2"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)   # kapı (gate)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)   # yukarı (up)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)   # aşağı (down)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer bloğu ve model
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = FeedForward(cfg)

    def forward(self, x, cos, sin, start_pos: int = 0):
        h = x + self.attention(self.attention_norm(x), cos, sin, start_pos)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = False   # bellek <-> hız takası (eğitimde açılabilir)

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.agirlik_bagla()

        # RoPE tabloları: kaydedilmez (persistent=False), her zaman formülden üretilir
        cos, sin = rope_tablosu(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual'a yazan projeksiyonlar: std / sqrt(2L) -> derinlikte varyans patlamasın
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def agirlik_bagla(self):
        """Tied embeddings: çıkış katmanı = embedding tablosu (aynı tensör)."""
        if self.cfg.tied_embeddings:
            self.output.weight = self.tok_embeddings.weight

    def rope_yenile(self):
        """
        RoPE tablolarını modelin cihazında fp32 olarak yeniden hesapla.
        model.to(bfloat16) sonrası veya meta cihazdan yüklemeden sonra çağır:
        büyük konumlarda (m·ω) açıların bf16'da hassasiyeti kaybolmasın.
        """
        cfg = self.cfg
        cos, sin = rope_tablosu(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
        dev = self.tok_embeddings.weight.device
        self.rope_cos, self.rope_sin = cos.to(dev), sin.to(dev)
        return self

    def parametre_sayisi(self) -> int:
        # parameters() paylaşılan (tied) ağırlığı bir kez sayar
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None,
                start_pos: int = 0, sadece_son: bool = False):
        """
        tokens:     (B, T) long
        targets:    (B, T) long, -100 = yok say  ->  çapraz entropi kaybı
        sadece_son: sadece son konumun logit'ini hesapla (üretimde V=128K için büyük tasarruf)
        Dönüş: (logits (B,T,V) veya (B,1,V) fp32, loss veya None)
        """
        B, T = tokens.shape
        assert start_pos + T <= self.cfg.max_seq_len, "dizi max_seq_len'i aşıyor"
        h = self.tok_embeddings(tokens)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = checkpoint(layer, h, cos, sin, start_pos, use_reentrant=False)
            else:
                h = layer(h, cos, sin, start_pos)
        h = self.norm(h)

        if targets is None:
            if sadece_son:
                h = h[:, -1:]
            return self.output(h).float(), None
        logits = self.output(h).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100)
        return logits, loss

    # ---------------- Üretim ----------------
    def cache_kur(self, max_batch: int, max_seq_len: int):
        for layer in self.layers:
            a = layer.attention
            a.cache_boyut, a.cache_k, a.cache_v = (max_batch, max_seq_len), None, None

    def cache_sil(self):
        for layer in self.layers:
            a = layer.attention
            a.cache_boyut, a.cache_k, a.cache_v = None, None, None

    @torch.inference_mode()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_p: float = 0.9, eos_id: Optional[int] = None) -> torch.Tensor:
        """
        prompt: (B, T0) long. KV-cache ile otoregresif üretim:
        önce prompt bir kerede işlenir (prefill), sonra her adımda tek token.
        """
        B, T0 = prompt.shape
        toplam = min(T0 + max_new_tokens, self.cfg.max_seq_len)
        self.cache_kur(B, toplam)
        try:
            tokens = prompt
            logits, _ = self(prompt, start_pos=0, sadece_son=True)
            bitti = torch.zeros(B, dtype=torch.bool, device=prompt.device)
            for pos in range(T0, toplam):
                nxt = ornekle(logits[:, -1], temperature, top_p)
                if eos_id is not None:
                    nxt = torch.where(bitti, torch.full_like(nxt, eos_id), nxt)
                    bitti |= nxt == eos_id
                tokens = torch.cat([tokens, nxt[:, None]], dim=1)
                if eos_id is not None and bitti.all():
                    break
                if pos + 1 < toplam:
                    logits, _ = self(nxt[:, None], start_pos=pos, sadece_son=True)
            return tokens
        finally:
            self.cache_sil()


def ornekle(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sıcaklık + nucleus (top-p) örnekleme. temperature=0 -> greedy."""
    if temperature <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temperature, dim=-1)
    p_sirali, idx = torch.sort(p, dim=-1, descending=True)
    kum = torch.cumsum(p_sirali, dim=-1)
    p_sirali[kum - p_sirali > top_p] = 0.0          # en olası çekirdeği tut
    p_sirali /= p_sirali.sum(-1, keepdim=True)
    secim = torch.multinomial(p_sirali, 1)
    return idx.gather(-1, secim).squeeze(-1)
