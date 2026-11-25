"""
Metin dosyalarını token'lara çevirip eğitim için ikili dosyalara yazar.

    python prepare_data.py --girdi metinler/ --cikti data/tr --tokenizer byte
    python prepare_data.py --girdi korpus.txt --cikti data/tr --tokenizer hf:meta-llama/Llama-3.2-1B

Çıktı:
    data/tr/train.bin, data/tr/val.bin   (uint32 token dizisi, np.memmap ile okunur)
    data/tr/meta.json                    (tokenizer, vocab_size, token sayıları)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tokenizer import tokenizer_al


def dosyalari_bul(girdi: Path):
    if girdi.is_file():
        return [girdi]
    return sorted(p for p in girdi.rglob("*") if p.suffix in {".txt", ".md"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", required=True, help="metin dosyası veya klasör")
    ap.add_argument("--cikti", required=True, help="çıktı klasörü")
    ap.add_argument("--tokenizer", default="byte")
    ap.add_argument("--val_orani", type=float, default=0.01)
    args = ap.parse_args()

    tok = tokenizer_al(args.tokenizer)
    dosyalar = dosyalari_bul(Path(args.girdi))
    assert dosyalar, "metin dosyası bulunamadı"

    ids = []
    for p in dosyalar:
        ids.extend(tok.encode(p.read_text(encoding="utf-8")))
        if tok.eos_id is not None:
            ids.append(tok.eos_id)          # belgeler arası ayraç
    arr = np.array(ids, dtype=np.uint32)

    n_val = max(1, int(len(arr) * args.val_orani))
    out = Path(args.cikti)
    out.mkdir(parents=True, exist_ok=True)
    arr[:-n_val].tofile(out / "train.bin")
    arr[-n_val:].tofile(out / "val.bin")
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "dtype": "uint32",
            "train_tokens": int(len(arr) - n_val), "val_tokens": int(n_val)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
