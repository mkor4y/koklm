# Katkı Rehberi

Katkılarınız memnuniyetle karşılanır.

## Geliştirme ortamı

```bash
pip install -e ".[dev,hf]"
pytest            # testler
ruff check .      # kod stili
```

## İlkeler

- **Önce matematik:** Yeni bir bileşen eklerken denklemini ve tensör boyutlarını
  `docs/matematik.md` içinde açıklayın.
- **Test edilebilirlik:** Her davranış bir testle doğrulanmalı. Mümkünse bilinen bir referansla
  (ör. Hugging Face Llama çıktısı veya yayımlanmış parametre sayısı) karşılaştırın.
- **Llama uyumluluğu:** Mimari değişiklikleri `tests/test_yukle.py`'deki uyumluluk testlerini
  bozmamalı; bozuyorsa yeni bir ayar olarak eklenmeli.

## Pull request

1. Ayrı bir dal açın, değişikliği küçük ve odaklı tutun.
2. `pytest` ve `ruff check .` temiz geçmeli.
3. Kullanıcıyı etkileyen değişiklikleri `CHANGELOG.md`'ye ekleyin.
