"""
Milyar parametreyi kalem kalem gör — ağırlıkları bellekte oluşturmadan, formülle.

    python scripts/parametre_say.py            # tüm ayarlar
    python scripts/parametre_say.py 1B --T 8192
"""
import argparse

from koklm.analiz import rapor
from koklm.config import AYARLAR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ayarlar", nargs="*", default=list(AYARLAR))
    ap.add_argument("--T", type=int, default=2048, help="bağlam uzunluğu (FLOP/KV-cache için)")
    a = ap.parse_args()
    for isim in a.ayarlar:
        print(rapor(AYARLAR[isim], a.T), end="\n\n")


if __name__ == "__main__":
    main()
