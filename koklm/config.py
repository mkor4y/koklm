"""
Model ayarları (hiperparametreler). Bu dosya torch'a bağımlı değildir.

Mimari, Meta'nın açık kaynak Llama 3 ailesinden esinlenmiştir:
  - "1B"    -> Llama 3.2 1B ile aynı boyutlar (~1.24 milyar parametre)
  - "8B"    -> Llama 3 8B ile aynı boyutlar   (~8.03 milyar parametre)
  - "kucuk" -> tek GPU / CPU'da dakikalar içinde denenebilecek minik model
"""
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ModelConfig:
    isim: str
    vocab_size: int          # V  : sözlük boyutu
    dim: int                 # d  : gizli boyut
    n_layers: int            # L  : transformer blok sayısı
    n_heads: int             # H  : sorgu (query) kafası sayısı
    n_kv_heads: int          # Hkv: key/value kafası sayısı (GQA)
    multiple_of: int = 256   # FFN boyutu bu sayının katına yuvarlanır
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    rope_scaling: Optional[dict] = None   # Llama 3.1+ uzun bağlam ölçeklemesi
    max_seq_len: int = 8192
    tied_embeddings: bool = False         # çıkış katmanı = embedding^T
    dropout: float = 0.0                  # Llama 0 kullanır
    ffn_dim: Optional[int] = None         # None -> Llama formülüyle hesaplanır

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim, kafa sayısına bölünmeli"
        assert self.n_heads % self.n_kv_heads == 0, "H, Hkv'ye bölünmeli"
        assert self.head_dim % 2 == 0, "RoPE için kafa boyutu çift olmalı"
        if self.ffn_dim is None:
            self.ffn_dim = ffn_boyutu(self.dim, self.multiple_of, self.ffn_dim_multiplier)

    @property
    def head_dim(self) -> int:       # dh = d / H
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:         # Hkv * dh
        return self.n_kv_heads * self.head_dim

    @property
    def n_rep(self) -> int:          # bir K/V kafasını kaç Q kafası paylaşıyor
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


def ffn_boyutu(dim: int, multiple_of: int, multiplier: Optional[float]) -> int:
    """
    Llama'nın SwiGLU gizli boyut formülü:
        h = 4d                    (klasik transformer)
        h = 2/3 * h               (SwiGLU 3 matris kullanır -> parametreyi eşitle)
        h = multiplier * h        (isteğe bağlı)
        h -> multiple_of'un katına yukarı yuvarla  (GPU verimliliği)
    """
    h = 4 * dim
    h = int(2 * h / 3)
    if multiplier is not None:
        h = int(multiplier * h)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


LLAMA3_ROPE_SCALING = {
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

AYARLAR = {
    # Minik deneme modeli (byte seviyesi tokenizer, V=256)
    "kucuk": ModelConfig(
        isim="kucuk", vocab_size=256, dim=256, n_layers=4,
        n_heads=8, n_kv_heads=2, multiple_of=64,
        rope_theta=10_000.0, max_seq_len=512, tied_embeddings=True,
    ),
    # Llama 3.2 1B ile aynı boyutlar
    "1B": ModelConfig(
        isim="1B", vocab_size=128_256, dim=2048, n_layers=16,
        n_heads=32, n_kv_heads=8, multiple_of=256, ffn_dim_multiplier=1.5,
        rope_theta=500_000.0, rope_scaling=LLAMA3_ROPE_SCALING,
        max_seq_len=131_072, tied_embeddings=True,
    ),
    # Llama 3 8B ile aynı boyutlar
    "8B": ModelConfig(
        isim="8B", vocab_size=128_256, dim=4096, n_layers=32,
        n_heads=32, n_kv_heads=8, multiple_of=1024, ffn_dim_multiplier=1.3,
        rope_theta=500_000.0, max_seq_len=8192, tied_embeddings=False,
    ),
}


def ayar_al(isim: str, **degisiklikler) -> ModelConfig:
    """Hazır ayarı al, istenen alanları değiştir: ayar_al("1B", max_seq_len=4096)"""
    d = AYARLAR[isim].to_dict()
    d.update(degisiklikler)
    if {"dim", "multiple_of", "ffn_dim_multiplier"} & degisiklikler.keys() \
            and "ffn_dim" not in degisiklikler:
        d["ffn_dim"] = None          # boyut değiştiyse FFN'i yeniden hesapla
    return ModelConfig.from_dict(d)
