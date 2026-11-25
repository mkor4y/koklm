"""
Çalıştır:  pytest -q
"""
import pytest
import torch

from model import AYARLAR, ayar_al, Transformer, rope_tablosu, rope_uygula
from parametre_say import parametre_dokumu


@pytest.fixture(scope="module")
def kucuk():
    torch.manual_seed(0)
    return Transformer(AYARLAR["kucuk"]).eval()


def test_ffn_boyutlari():
    assert AYARLAR["1B"].ffn_dim == 8192
    assert AYARLAR["8B"].ffn_dim == 14336


@pytest.mark.parametrize("isim,beklenen", [("1B", 1_235_814_400), ("8B", 8_030_261_248)])
def test_milyar_parametre_meta_cihazda(isim, beklenen):
    # "meta" cihaz: tensörlerin sadece şekli var, bellek ayrılmaz -> 8B bile anında
    with torch.device("meta"):
        model = Transformer(AYARLAR[isim])
    assert model.parametre_sayisi() == beklenen == parametre_dokumu(AYARLAR[isim])[3]


def test_cikis_sekli_ve_kayip(kucuk):
    x = torch.randint(0, 256, (2, 16))
    logits, loss = kucuk(x, x)
    assert logits.shape == (2, 16, 256)
    # Rastgele başlangıçta kayıp ≈ ln(V)
    assert abs(loss.item() - torch.log(torch.tensor(256.0)).item()) < 0.5


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


def test_config_gidis_donus():
    cfg = AYARLAR["1B"]
    from model import ModelConfig
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
