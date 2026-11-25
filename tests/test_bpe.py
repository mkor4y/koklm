"""Sıfırdan BPE tokenizer — torch gerektirmez."""
from koklm.bpe import BPETokenizer
from koklm.tokenizer import ByteTokenizer, tokenizer_al

METIN = ("Bir varmış bir yokmuş, evvel zaman içinde kalbur saman içinde. "
         "Develer tellal iken, pireler berber iken. 2026 yılında KökLM doğdu! ") * 30


def test_gidis_donus_ve_sikistirma():
    tok = BPETokenizer.egit(METIN, vocab_size=400)
    ids = tok.encode(METIN)
    assert tok.decode(ids) == METIN
    assert len(ids) < len(METIN.encode("utf-8")) / 3      # belirgin sıkıştırma
    assert max(ids) < tok.vocab_size


def test_gorulmemis_metin_ve_unicode():
    tok = BPETokenizer.egit(METIN, vocab_size=320)
    s = "Şğüöçı İSTANBUL 🌞 日本語 \n\t  boşluklar"
    assert tok.decode(tok.encode(s)) == s


def test_kaydet_yukle(tmp_path):
    tok = BPETokenizer.egit(METIN, vocab_size=350)
    yol = tmp_path / "tok.json"
    tok.kaydet(yol)
    tok2 = tokenizer_al(f"bpe:{yol}")
    assert tok2.encode(METIN) == tok.encode(METIN)
    assert tok2.vocab_size == tok.vocab_size == tok.eos_id + 1


def test_birlestirmeler_ogrenme_sirasi():
    # İlk birleştirme en sık çift olmalı
    tok = BPETokenizer.egit("aaaa aaaa aaaa bb", vocab_size=258)
    assert tok.birlestirmeler[0] == (ord("a"), ord("a"))


def test_byte_tokenizer():
    t = ByteTokenizer()
    assert t.decode(t.encode("Kök")) == "Kök"
    assert len(t.encode("ö")) == 2
