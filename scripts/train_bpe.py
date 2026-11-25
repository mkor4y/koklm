"""
KökLM'in sıfırdan BPE tokenizer'ını eğit.

    python scripts/train_bpe.py --girdi metinler/ --vocab_size 8192 --cikti tokenizer.json
    python scripts/prepare_data.py --girdi metinler/ --cikti data/tr --tokenizer bpe:tokenizer.json
"""
import argparse
import time
from pathlib import Path

from koklm.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", required=True, help="metin dosyası veya klasör")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--cikti", default="tokenizer.json")
    ap.add_argument("--max_karakter", type=int, default=50_000_000,
                    help="eğitimde kullanılacak en fazla karakter (saf Python, hız için)")
    a = ap.parse_args()

    yol = Path(a.girdi)
    dosyalar = [yol] if yol.is_file() else sorted(
        p for p in yol.rglob("*") if p.suffix in {".txt", ".md"})
    metin = "\n".join(p.read_text(encoding="utf-8") for p in dosyalar)[:a.max_karakter]

    t0 = time.time()
    tok = BPETokenizer.egit(metin, a.vocab_size, ayrintili=True)
    tok.kaydet(a.cikti)
    ornek = metin[:10_000]
    oran = len(ornek.encode("utf-8")) / max(1, len(tok.encode(ornek)))
    print(f"vocab={tok.vocab_size}  süre={time.time() - t0:.1f}s  "
          f"sıkıştırma≈{oran:.2f} bayt/token  ->  {a.cikti}")


if __name__ == "__main__":
    main()
