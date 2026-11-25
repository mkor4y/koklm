from .config import ModelConfig, AYARLAR, ayar_al, ffn_boyutu

try:  # config ve parametre sayacı torch olmadan da kullanılabilsin
    from .model import Transformer, rope_tablosu, rope_uygula, ornekle
except ImportError:  # pragma: no cover
    pass
