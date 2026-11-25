"""
Tokenizer seçimi.

  byte            : her UTF-8 baytı bir token (V=256). Hiçbir şey indirmeden çalışır.
  bpe:<dosya>     : KökLM'in sıfırdan BPE'si (scripts/train_bpe.py ile eğitilir).
  hf:<model>      : Hugging Face tokenizer, örn. "hf:meta-llama/Llama-3.2-1B" (V=128 256).
"""
from .bpe import BPETokenizer


class ByteTokenizer:
    vocab_size = 256
    eos_id = None   # 256'lık sözlükte ayrı özel token yok

    def encode(self, metin: str) -> list[int]:
        return list(metin.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, isim: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(isim)
        self.vocab_size = len(self.tok)
        self.eos_id = self.tok.eos_token_id
        self.bos_id = self.tok.bos_token_id

    def encode(self, metin: str) -> list[int]:
        return self.tok.encode(metin, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)


def tokenizer_al(isim: str):
    if isim == "byte":
        return ByteTokenizer()
    if isim.startswith("bpe:"):
        return BPETokenizer.yukle(isim[4:])
    if isim.startswith("hf:"):
        return HFTokenizer(isim[3:])
    raise ValueError(f"Bilinmeyen tokenizer: {isim}  (byte | bpe:<dosya> | hf:<model>)")
