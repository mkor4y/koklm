"""KökLM — milyar parametrelik LLM'leri kökünden çözmek için sıfırdan yazılmış dil modeli."""

__version__ = "0.2.0"

from .config import AYARLAR, LLAMA3_ROPE_SCALING, ModelConfig, ayar_al, ffn_boyutu

try:  # config, analiz ve BPE torch olmadan da kullanılabilsin
    from .model import Transformer, ornekle, rope_tablosu, rope_uygula
except ImportError:  # pragma: no cover
    pass

__all__ = ["AYARLAR", "LLAMA3_ROPE_SCALING", "ModelConfig", "ayar_al", "ffn_boyutu",
           "Transformer", "ornekle", "rope_tablosu", "rope_uygula", "__version__"]
