"""
Tokenizer'lar.

  byte : her UTF-8 baytı bir token (V=256). Hiçbir şey indirmeden çalışır -> "kucuk" model.
  hf:<isim> : Hugging Face tokenizer, örn. "hf:meta-llama/Llama-3.2-1B" (V=128 256)
              -> `pip install transformers` ve model sayfasında erişim izni gerekir.
"""


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

    def encode(self, metin: str) -> list[int]:
        return self.tok.encode(metin, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))


def tokenizer_al(isim: str):
    if isim == "byte":
        return ByteTokenizer()
    if isim.startswith("hf:"):
        return HFTokenizer(isim[3:])
    raise ValueError(f"Bilinmeyen tokenizer: {isim}  (byte | hf:<model>)")
