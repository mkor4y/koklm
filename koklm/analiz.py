"""
Parametre ve bellek analizi — ağırlıkları oluşturmadan, kapalı formülle. torch gerektirmez.

    N = V·d + L·[2d² + 2d·(Hkv·dh) + 3·d·f + 2d] + d + (tied değilse) d·V
"""
from .config import ModelConfig


def parametre_dokumu(cfg: ModelConfig):
    """Dönüş: (blok kalemleri, blok toplamı, model kalemleri, toplam)"""
    d, kv, f, V, L = cfg.dim, cfg.kv_dim, cfg.ffn_dim, cfg.vocab_size, cfg.n_layers
    blok = {
        "Wq  (d x d)": d * d,
        "Wk  (d x Hkv*dh)": d * kv,
        "Wv  (d x Hkv*dh)": d * kv,
        "Wo  (d x d)": d * d,
        "W1 gate (d x f)": d * f,
        "W3 up   (d x f)": d * f,
        "W2 down (f x d)": f * d,
        "2 x RMSNorm (d)": 2 * d,
    }
    blok_top = sum(blok.values())
    ust = {
        "Token embedding (V x d)": V * d,
        f"{L} blok x {blok_top:,}": L * blok_top,
        "Son RMSNorm (d)": d,
        "Çıkış katmanı (d x V)": 0 if cfg.tied_embeddings else d * V,
    }
    return blok, blok_top, ust, sum(ust.values())


def toplam_parametre(cfg: ModelConfig) -> int:
    return parametre_dokumu(cfg)[3]


def flop_token_basina(cfg: ModelConfig, T: int) -> tuple[float, float]:
    """
    Token başına yaklaşık FLOP (ileri, eğitim):
      ileri ≈ 2N + 2·L·T·d   (ikinci terim: attention skorları QKᵀ ve AV)
      eğitim ≈ 3 × ileri      (geri geçiş ≈ 2 × ileri)
    """
    N = toplam_parametre(cfg)
    ileri = 2 * N + 2 * cfg.n_layers * T * cfg.dim
    return ileri, 3 * ileri


def kv_cache_bayt(cfg: ModelConfig, T: int, bayt: float = 2) -> float:
    """KV-cache = 2 (K ve V) · L · Hkv · dh · T · bayt"""
    return 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * T * bayt


def rapor(cfg: ModelConfig, T: int = 2048) -> str:
    blok, blok_top, ust, toplam = parametre_dokumu(cfg)
    s = [f"=== {cfg.isim} ===",
         f"V={cfg.vocab_size:,}  d={cfg.dim}  L={cfg.n_layers}  H={cfg.n_heads}  "
         f"Hkv={cfg.n_kv_heads}  dh={cfg.head_dim}  f={cfg.ffn_dim}  tied={cfg.tied_embeddings}",
         "", "Tek blok:"]
    for k, v in blok.items():
        s.append(f"  {k:<22} {v:>16,}  ({100 * v / blok_top:5.1f}%)")
    s += [f"  {'BLOK TOPLAM':<22} {blok_top:>16,}", "", "Model:"]
    for k, v in ust.items():
        s.append(f"  {k:<28} {v:>16,}  ({100 * v / toplam:5.1f}%)")
    s += [f"  {'TOPLAM':<28} {toplam:>16,}  ≈ {toplam / 1e9:.2f} milyar", "", "Ağırlık belleği:"]
    for ad, bayt in [("fp32", 4), ("bf16", 2), ("int8", 1), ("int4", 0.5)]:
        s.append(f"  {ad:<5} {toplam * bayt / 2**30:8.2f} GiB")
    s.append(f"  AdamW ile eğitim (~16 B/param, aktivasyon hariç): {toplam * 16 / 2**30:.1f} GiB")
    ileri, egitim = flop_token_basina(cfg, T)
    s += ["", f"Hesap (T={T}):",
          f"  ileri  ≈ {ileri / 1e9:.2f} GFLOP/token",
          f"  eğitim ≈ {egitim / 1e9:.2f} GFLOP/token",
          f"  KV-cache (bf16, T={T}) = {kv_cache_bayt(cfg, T) / 2**20:.1f} MiB"]
    return "\n".join(s)
