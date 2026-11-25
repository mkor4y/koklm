"""
Milyar parametreyi kalem kalem gör — ağırlıkları bellekte oluşturmadan, formülle.

Kullanım:
    python parametre_say.py          # tüm ayarlar
    python parametre_say.py 1B       # tek ayar
"""
import sys
from model.config import AYARLAR, ModelConfig


def parametre_dokumu(cfg: ModelConfig):
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


def yazdir(cfg: ModelConfig):
    blok, blok_top, ust, toplam = parametre_dokumu(cfg)
    print(f"\n=== {cfg.isim} ===")
    print(f"V={cfg.vocab_size:,}  d={cfg.dim}  L={cfg.n_layers}  H={cfg.n_heads}  "
          f"Hkv={cfg.n_kv_heads}  dh={cfg.head_dim}  f={cfg.ffn_dim}  "
          f"tied={cfg.tied_embeddings}")
    print("\nTek blok:")
    for k, v in blok.items():
        print(f"  {k:<22} {v:>16,}  ({100*v/blok_top:5.1f}%)")
    print(f"  {'BLOK TOPLAM':<22} {blok_top:>16,}")
    print("\nModel:")
    for k, v in ust.items():
        print(f"  {k:<28} {v:>16,}  ({100*v/toplam:5.1f}%)")
    print(f"  {'TOPLAM':<28} {toplam:>16,}  ≈ {toplam/1e9:.2f} milyar")
    print("\nAğırlık belleği:")
    for ad, bayt in [("fp32", 4), ("bf16", 2), ("int8", 1), ("int4", 0.5)]:
        print(f"  {ad:<5} {toplam*bayt/2**30:8.2f} GiB")
    # Eğitimde Adam ile kaba tahmin: fp32 ağırlık+grad+2 moment = 16 bayt/param
    print(f"  Adam ile eğitim (~16 B/param, aktivasyonlar hariç): "
          f"{toplam*16/2**30:.1f} GiB")


if __name__ == "__main__":
    adlar = sys.argv[1:] or list(AYARLAR)
    for a in adlar:
        yazdir(AYARLAR[a])
