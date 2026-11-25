"""Parametre formülleri — torch gerektirmez."""
import pytest

from koklm import AYARLAR, ModelConfig, ayar_al
from koklm.analiz import kv_cache_bayt, parametre_dokumu, rapor, toplam_parametre


def test_ffn_boyutlari():
    # Llama'nın yayımlanmış FFN boyutları
    assert AYARLAR["1B"].ffn_dim == 8192
    assert AYARLAR["8B"].ffn_dim == 14336


@pytest.mark.parametrize("isim,beklenen", [("1B", 1_235_814_400), ("8B", 8_030_261_248)])
def test_bilinen_parametre_sayilari(isim, beklenen):
    # Meta'nın açıkladığı sayılar
    assert toplam_parametre(AYARLAR[isim]) == beklenen


def test_dokum_toplami_tutarli():
    blok, blok_top, ust, toplam = parametre_dokumu(AYARLAR["1B"])
    assert sum(blok.values()) == blok_top
    assert sum(ust.values()) == toplam


def test_kv_cache_1b():
    # 2 · 16 · 8 · 64 · 8192 · 2 bayt = 256 MiB
    assert kv_cache_bayt(AYARLAR["1B"], 8192) == 256 * 2**20


def test_config_gidis_donus():
    for cfg in AYARLAR.values():
        assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_ayar_al_ffn_yeniden_hesaplar():
    assert ayar_al("1B", dim=1024).ffn_dim != AYARLAR["1B"].ffn_dim
    assert ayar_al("1B", max_seq_len=4096).ffn_dim == AYARLAR["1B"].ffn_dim


def test_rapor_calisir():
    assert "1,235,814,400" in rapor(AYARLAR["1B"])


@pytest.mark.parametrize("kw", [dict(dim=100, n_heads=3), dict(n_heads=8, n_kv_heads=3)])
def test_gecersiz_config(kw):
    base = dict(isim="x", vocab_size=10, dim=64, n_layers=1, n_heads=4, n_kv_heads=2)
    base.update(kw)
    with pytest.raises(AssertionError):
        ModelConfig(**base)
