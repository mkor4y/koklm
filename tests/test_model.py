"""Model testleri (PyTorch gerekir)."""
import math

import pytest

torch = pytest.importorskip("torch")

from koklm import AYARLAR, Transformer, ayar_al, rope_tablosu, rope_uygula  # noqa: E402
from koklm.analiz import toplam_parametre  # noqa: E402


@pytest.fixture(scope="module")
def kucuk():
    torch.manual_seed(0)
    return Transformer(AYARLAR["kucuk"]).eval()


@pytest.mark.parametrize("isim,beklenen", [("1B", 1_235_814_400), ("8B", 8_030_261_248)])
def test_milyar_parametre_meta_cihazda(isim, beklenen):
    # "meta" cihaz: tensörlerin sadece şekli var, bellek ayrılmaz -> 8B bile anında
    with torch.device("meta"):
        model = Transformer(AYARLAR[isim])
    assert model.parametre_sayisi() == beklenen == toplam_parametre(AYARLAR[isim])


def test_cikis_sekli_ve_kayip(kucuk):
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))      # girdiden bağımsız hedefler
    logits, loss = kucuk(x, y)
    assert logits.shape == (2, 16, 256)
    # Rastgele başlangıçta her token ~eşit olasılıklı -> kayıp ≈ ln(V)
    assert abs(loss.item() - math.log(256)) < 0.5


def test_nedensellik(kucuk):
    x = torch.randint(0, 256, (1, 12))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 256
    a, _ = kucuk(x)
    b, _ = kucuk(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_kv_cache_ayni_sonuc(kucuk):
    # Cache ile adım adım işlemek, tüm diziyi bir kerede işlemekle aynı olmalı
    x = torch.randint(0, 256, (2, 10))
    tam, _ = kucuk(x)
    kucuk.cache_kur(2, 10)
    try:
        ilk, _ = kucuk(x[:, :6], start_pos=0)          # prefill
        parca = [ilk]
        orta, _ = kucuk(x[:, 6:8], start_pos=6)        # cache varken çok token
        parca.append(orta)
        for t in range(8, 10):                          # tek tek
            l, _ = kucuk(x[:, t:t + 1], start_pos=t)
            parca.append(l)
    finally:
        kucuk.cache_sil()
    assert torch.allclose(tam, torch.cat(parca, dim=1), atol=1e-4)


def test_rope_goreli_konum():
    dh = 16
    cos, sin = rope_tablosu(dh, 64, 10_000.0)
    q, k = torch.randn(dh), torch.randn(dh)
    Q = q.expand(1, 64, 1, dh)
    K = k.expand(1, 64, 1, dh)
    rq, rk = rope_uygula(Q, cos, sin)[0, :, 0], rope_uygula(K, cos, sin)[0, :, 0]
    assert torch.isclose(rq[5] @ rk[2], rq[40] @ rk[37], atol=1e-4)


def test_uretim(kucuk):
    out = kucuk.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
    assert out.shape == (1, 8)


def test_uretim_vocab_siniri(kucuk):
    torch.manual_seed(0)
    out = kucuk.generate(torch.tensor([[1, 2, 3]] * 4), max_new_tokens=30,
                         temperature=5.0, top_p=1.0, vocab_siniri=10)
    assert int(out[:, 3:].max()) < 10


def test_bir_egitim_adimi_kaybi_dusurur():
    torch.manual_seed(0)
    model = Transformer(ayar_al("kucuk", n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (4, 32))
    _, l0 = model(x[:, :-1], x[:, 1:])
    for _ in range(20):
        opt.zero_grad()
        _, l = model(x[:, :-1], x[:, 1:])
        l.backward()
        opt.step()
    assert l.item() < l0.item()


def test_sadece_son_logit(kucuk):
    x = torch.randint(0, 256, (2, 9))
    tam, _ = kucuk(x)
    son, _ = kucuk(x, sadece_son=True)
    assert son.shape == (2, 1, 256)
    assert torch.allclose(tam[:, -1:], son, atol=1e-5)


def test_tied_agirlik_paylasimi(kucuk):
    assert kucuk.output.weight is kucuk.tok_embeddings.weight


def test_llama3_rope_olcekleme():
    s = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
         "original_max_position_embeddings": 32}
    _, s0 = rope_tablosu(16, 64, 10_000.0)
    _, s1 = rope_tablosu(16, 64, 10_000.0, s)
    assert torch.allclose(s0[:, 0], s1[:, 0])           # yüksek frekans aynı kalır
    assert not torch.allclose(s0[:, -1], s1[:, -1])     # düşük frekans ölçeklenir
    # En düşük frekans tam olarak factor kadar yavaşlar: sin(m·ω/8)
    w = 10_000.0 ** (-14 / 16)
    m = torch.arange(64, dtype=torch.float32)
    assert torch.allclose(s1[:, -1], torch.sin(m * w / 8), atol=1e-6)


def test_rope_yenile_fp32(kucuk):
    m = Transformer(AYARLAR["kucuk"]).to(torch.bfloat16)
    assert m.rope_cos.dtype == torch.bfloat16
    m.rope_yenile()
    assert m.rope_cos.dtype == torch.float32
