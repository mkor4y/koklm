"""
Sıfırdan byte seviyesi BPE (Byte Pair Encoding) tokenizer. Saf Python.

Algoritma:
  1. Metni ön-parçalara böl (kelime + önündeki boşluk, noktalama, boşluk).
     Birleştirmeler parça sınırlarını aşmaz -> "kelime." gibi karışık token'lar oluşmaz.
  2. Her parçayı UTF-8 baytlarına çevir: başlangıç sözlüğü = 256 bayt.
  3. Tekrarla: en sık geçen komşu çifti (a, b) bul, yeni token n = 256 + k yap,
     her yerde (a, b) -> n değiştir.  Sözlük her adımda 1 büyür.
  4. Kodlama: öğrenilen birleştirmeleri öğrenilme sırasına göre (en düşük sıra önce) uygula.

Sözlük boyutu V = 256 + birleştirme sayısı + özel token sayısı.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ON_PARCA = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\w\s]+|\s+(?!\S)|\s+")
EOS = "<|eos|>"


def _ciftleri_say(parcalar: dict[tuple[int, ...], int]) -> Counter:
    sayac: Counter = Counter()
    for ids, frekans in parcalar.items():
        for cift in zip(ids, ids[1:]):
            sayac[cift] += frekans
    return sayac


def _birlestir(ids: tuple[int, ...], cift: tuple[int, int], yeni: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == cift[0] and ids[i + 1] == cift[1]:
            out.append(yeni)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, birlestirmeler: list[tuple[int, int]]):
        self.birlestirmeler = [tuple(c) for c in birlestirmeler]
        self.sira = {c: i for i, c in enumerate(self.birlestirmeler)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.birlestirmeler):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eos_id = 256 + len(self.birlestirmeler)
        self.vocab_size = self.eos_id + 1
        self._onbellek: dict[str, list[int]] = {}

    # ---------------- Eğitim ----------------
    @classmethod
    def egit(cls, metin: str, vocab_size: int, ayrintili: bool = False) -> BPETokenizer:
        n_birlestirme = vocab_size - 256 - 1          # -1: EOS
        assert n_birlestirme >= 0, "vocab_size en az 257 olmalı"
        frek = Counter(ON_PARCA.findall(metin))
        parcalar = {tuple(p.encode("utf-8")): f for p, f in frek.items()}
        birlestirmeler: list[tuple[int, int]] = []
        for k in range(n_birlestirme):
            sayac = _ciftleri_say(parcalar)
            if not sayac:
                break
            # En sık çift; eşitlikte küçük çift (deterministik)
            cift = max(sayac, key=lambda c: (sayac[c], -c[0], -c[1]))
            if sayac[cift] < 2:
                break
            yeni = 256 + k
            parcalar = {_birlestir(ids, cift, yeni): f for ids, f in parcalar.items()}
            birlestirmeler.append(cift)
            if ayrintili and (k + 1) % 500 == 0:
                print(f"  {k + 1}/{n_birlestirme} birleştirme")
        return cls(birlestirmeler)

    # ---------------- Kodlama ----------------
    def _parca_kodla(self, parca: str) -> list[int]:
        if parca in self._onbellek:
            return self._onbellek[parca]
        ids = list(parca.encode("utf-8"))
        while len(ids) >= 2:
            # Sözlükte en erken öğrenilmiş çifti bul
            cift = min(zip(ids, ids[1:]), key=lambda c: self.sira.get(c, float("inf")))
            if cift not in self.sira:
                break
            ids = list(_birlestir(tuple(ids), cift, 256 + self.sira[cift]))
        if len(self._onbellek) < 100_000:
            self._onbellek[parca] = ids
        return ids

    def encode(self, metin: str) -> list[int]:
        out: list[int] = []
        for parca in ON_PARCA.findall(metin):
            out.extend(self._parca_kodla(parca))
        return out

    def decode(self, ids) -> str:
        b = b"".join(self.vocab[int(i)] for i in ids if int(i) != self.eos_id)
        return b.decode("utf-8", errors="replace")

    # ---------------- Kaydet / yükle ----------------
    def kaydet(self, yol: str | Path):
        veri = {"tur": "koklm-bpe", "surum": 1, "on_parca": ON_PARCA.pattern,
                "ozel": {EOS: self.eos_id}, "birlestirmeler": self.birlestirmeler}
        Path(yol).write_text(json.dumps(veri), encoding="utf-8")

    @classmethod
    def yukle(cls, yol: str | Path) -> BPETokenizer:
        veri = json.loads(Path(yol).read_text(encoding="utf-8"))
        assert veri.get("tur") == "koklm-bpe", "KökLM BPE dosyası değil"
        return cls([tuple(c) for c in veri["birlestirmeler"]])
