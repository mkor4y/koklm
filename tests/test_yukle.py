"""
Gerçek Llama uyumluluğu: Hugging Face'in Llama uygulamasıyla aynı ağırlıklar
verildiğinde KökLM'in aynı logit'leri ürettiğini doğrular.
"""
import pytest

torch = pytest.importorskip("torch")

from koklm import Transformer  # noqa: E402
from koklm.yukle import (  # noqa: E402
    hf_config_cevir,
    hf_den_meta_permutasyon,
    hf_state_dict_cevir,
    meta_dan_hf_permutasyon,
)

OLCEK = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
         "high_freq_factor": 4.0, "original_max_position_embeddings": 32}


def test_rope_ayarlari_iki_bicim():
    from koklm.yukle import _rope_ayarlari
    eski = {"rope_theta": 500000.0, "rope_scaling": dict(OLCEK)}
    yeni = {"rope_parameters": dict(OLCEK, rope_theta=500000.0)}
    assert _rope_ayarlari(eski) == _rope_ayarlari(yeni)
    assert _rope_ayarlari(eski)[1]["factor"] == 8.0


def test_permutasyon_gidis_donus():
    w = torch.randn(4 * 16, 64)
    assert torch.equal(hf_den_meta_permutasyon(meta_dan_hf_permutasyon(w, 4), 4), w)


def test_permutasyon_satir_eslemesi():
    # Meta satırı 2i+j  ->  HF satırı j·(dh/2)+i
    dh, n = 8, 1
    w = torch.arange(dh, dtype=torch.float32)[:, None]
    hf = meta_dan_hf_permutasyon(w, n)[:, 0].tolist()
    assert hf == [0, 2, 4, 6, 1, 3, 5, 7]


transformers = pytest.importorskip("transformers")


def _hf_model(tied, scaling):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        rope_theta=10_000.0, max_position_embeddings=128,
        tie_word_embeddings=tied, rope_scaling=scaling,
    )
    m = transformers.LlamaForCausalLM(cfg)
    with torch.no_grad():   # norm ağırlıklarını da test etmek için 1'den uzaklaştır
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
    return m.eval()


@pytest.mark.parametrize("tied,scaling", [(True, None), (False, OLCEK)])
def test_hf_llama_ile_ayni_logit(tied, scaling):
    hf = _hf_model(tied, scaling)
    cfg = hf_config_cevir(hf.config.to_dict())
    assert (cfg.rope_scaling is not None) == (scaling is not None)
    model = Transformer(cfg).eval()
    model.load_state_dict(hf_state_dict_cevir(hf.state_dict(), cfg))
    x = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        ref = hf(x).logits
        biz, _ = model(x)
    assert torch.allclose(biz, ref, atol=1e-4), (biz - ref).abs().max()


def test_hf_klasorden_yukle(tmp_path):
    from koklm.yukle import hf_yukle

    hf = _hf_model(True, OLCEK)
    hf.save_pretrained(tmp_path, safe_serialization=True)
    model = hf_yukle(str(tmp_path), dtype=torch.float32)
    assert model.output.weight is model.tok_embeddings.weight
    x = torch.randint(0, 256, (1, 30))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], hf(x).logits, atol=1e-4)
