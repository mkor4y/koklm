"""
Gerçek Llama 3 ağırlıklarını KökLM mimarisine yükleme.

İki kaynak desteklenir:
  1. Meta'nın orijinal formatı (consolidated.00.pth + params.json)
     -> Parametre adları KökLM ile birebir aynı, doğrudan yüklenir.
  2. Hugging Face formatı (model.safetensors + config.json)
     -> Adlar eşlenir ve Wq/Wk satırları permüte edilir (aşağıda).

RoPE düzeni farkı (neden permütasyon gerekiyor):
  Meta / KökLM: her kafada (x_0,x_1), (x_2,x_3), ... çiftleri döner          ("interleaved")
  HF          : her kafada (x_i, x_{i+dh/2}) çiftleri döner                 ("rotate_half")
  HF dönüştürücüsü Wq ve Wk'nin satırlarını kafa içinde yeniden sıralar:
      Meta satırı 2i+j  ->  HF satırı j·(dh/2)+i        (j ∈ {0,1})
  Aynı permütasyon hem Q'ya hem K'ya uygulandığı için q·k çarpımı değişmez;
  burada tersini uygulayıp Meta düzenine geri dönüyoruz.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import Transformer


def hf_den_meta_permutasyon(w: torch.Tensor, n_kafa: int) -> torch.Tensor:
    """HF (rotate_half) satır düzeni -> Meta (interleaved) satır düzeni."""
    satir, sutun = w.shape
    dh = satir // n_kafa
    return w.view(n_kafa, 2, dh // 2, sutun).transpose(1, 2).reshape(satir, sutun)


def meta_dan_hf_permutasyon(w: torch.Tensor, n_kafa: int) -> torch.Tensor:
    """Meta (interleaved) satır düzeni -> HF (rotate_half) satır düzeni."""
    satir, sutun = w.shape
    dh = satir // n_kafa
    return w.view(n_kafa, dh // 2, 2, sutun).transpose(1, 2).reshape(satir, sutun)


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------
def hf_config_cevir(hf: dict, isim: str = "hf") -> ModelConfig:
    rs = hf.get("rope_scaling")
    scaling = None
    if rs and rs.get("rope_type", rs.get("type")) == "llama3":
        scaling = {k: rs[k] for k in ("factor", "low_freq_factor", "high_freq_factor",
                                      "original_max_position_embeddings")}
    head_dim = hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"]
    assert head_dim * hf["num_attention_heads"] == hf["hidden_size"], \
        "KökLM dh = d / H varsayar"
    return ModelConfig(
        isim=isim,
        vocab_size=hf["vocab_size"],
        dim=hf["hidden_size"],
        n_layers=hf["num_hidden_layers"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
        ffn_dim=hf["intermediate_size"],
        norm_eps=hf.get("rms_norm_eps", 1e-5),
        rope_theta=hf.get("rope_theta", 10_000.0),
        rope_scaling=scaling,
        max_seq_len=hf.get("max_position_embeddings", 8192),
        tied_embeddings=hf.get("tie_word_embeddings", False),
    )


def hf_state_dict_cevir(sd: dict, cfg: ModelConfig) -> dict:
    """HF LlamaForCausalLM state_dict -> KökLM state_dict"""
    def al(k):
        return sd[k] if k in sd else sd[k.removeprefix("model.")]

    out = {"tok_embeddings.weight": al("model.embed_tokens.weight"),
           "norm.weight": al("model.norm.weight")}
    for i in range(cfg.n_layers):
        p, q = f"model.layers.{i}.", f"layers.{i}."
        out[q + "attention.wq.weight"] = hf_den_meta_permutasyon(
            al(p + "self_attn.q_proj.weight"), cfg.n_heads)
        out[q + "attention.wk.weight"] = hf_den_meta_permutasyon(
            al(p + "self_attn.k_proj.weight"), cfg.n_kv_heads)
        out[q + "attention.wv.weight"] = al(p + "self_attn.v_proj.weight")
        out[q + "attention.wo.weight"] = al(p + "self_attn.o_proj.weight")
        out[q + "feed_forward.w1.weight"] = al(p + "mlp.gate_proj.weight")
        out[q + "feed_forward.w3.weight"] = al(p + "mlp.up_proj.weight")
        out[q + "feed_forward.w2.weight"] = al(p + "mlp.down_proj.weight")
        out[q + "attention_norm.weight"] = al(p + "input_layernorm.weight")
        out[q + "ffn_norm.weight"] = al(p + "post_attention_layernorm.weight")
    out["output.weight"] = sd.get("lm_head.weight", out["tok_embeddings.weight"]) \
        if not cfg.tied_embeddings else out["tok_embeddings.weight"]
    return out


def hf_yukle(kaynak: str, dtype: torch.dtype = torch.bfloat16,
             device: str = "cpu") -> Transformer:
    """
    kaynak: yerel klasör veya Hub adı (örn. "meta-llama/Llama-3.2-1B").
    Gerekenler: pip install "koklm[hf]"   (+ Hub'da model erişim izni)
    """
    from safetensors.torch import load_file

    yol = Path(kaynak)
    if not yol.exists():
        from huggingface_hub import snapshot_download
        yol = Path(snapshot_download(kaynak, allow_patterns=["*.json", "*.safetensors"]))
    cfg = hf_config_cevir(json.loads((yol / "config.json").read_text()), isim=yol.name)
    sd: dict = {}
    for f in sorted(yol.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    return _modele_yukle(cfg, hf_state_dict_cevir(sd, cfg), dtype, device)


# ---------------------------------------------------------------------------
# Meta orijinal formatı
# ---------------------------------------------------------------------------
def meta_config_cevir(p: dict, isim: str = "meta") -> ModelConfig:
    from .config import LLAMA3_ROPE_SCALING
    return ModelConfig(
        isim=isim,
        vocab_size=p["vocab_size"],
        dim=p["dim"],
        n_layers=p["n_layers"],
        n_heads=p["n_heads"],
        n_kv_heads=p.get("n_kv_heads", p["n_heads"]),
        multiple_of=p.get("multiple_of", 256),
        ffn_dim_multiplier=p.get("ffn_dim_multiplier"),
        norm_eps=p.get("norm_eps", 1e-5),
        rope_theta=p.get("rope_theta", 500_000.0),
        rope_scaling=LLAMA3_ROPE_SCALING if p.get("use_scaled_rope") else None,
        max_seq_len=131_072 if p.get("use_scaled_rope") else 8192,
        tied_embeddings=False,
    )


def meta_yukle(klasor: str, dtype: torch.dtype = torch.bfloat16,
               device: str = "cpu") -> Transformer:
    """klasor: consolidated.00.pth ve params.json içeren Meta indirmesi."""
    yol = Path(klasor)
    cfg = meta_config_cevir(json.loads((yol / "params.json").read_text()), isim=yol.name)
    sd = torch.load(yol / "consolidated.00.pth", map_location="cpu", weights_only=True)
    if "output.weight" not in sd:                     # bazı küçük modellerde paylaşımlı
        cfg.tied_embeddings = True
        sd["output.weight"] = sd["tok_embeddings.weight"]
    return _modele_yukle(cfg, sd, dtype, device)


def _modele_yukle(cfg, sd, dtype, device) -> Transformer:
    with torch.device("meta"):
        model = Transformer(cfg)
    model = model.to_empty(device=device)
    model.agirlik_bagla()             # to_empty paylaşımlı ağırlığı koparabilir
    model.load_state_dict(sd, strict=True)
    model = model.to(dtype)
    return model.rope_yenile().eval()  # RoPE'yi fp32 ve doğru cihazda yeniden hesapla
