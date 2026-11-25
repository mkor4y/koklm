# Değişiklik Günlüğü

## [0.2.0] — 2026-11-26

### Eklendi
- `koklm` Python paketi ve `pyproject.toml`; `pip install -e .` ile kurulum
- Gerçek Llama 3 ağırlıklarını yükleme (Meta orijinal ve Hugging Face formatları)
- Hugging Face Llama ile birebir aynı logit'leri doğrulayan testler
- Sıfırdan byte seviyesi BPE tokenizer (`koklm/bpe.py`, `scripts/train_bpe.py`)
- FLOP ve KV-cache analizi (`koklm/analiz.py`)
- Eğitim metriklerinin `log.jsonl` dosyasına yazılması
- GitHub Actions: ruff + pytest (Python 3.10 ve 3.12)

### İyileştirildi
- Attention, torch ≥ 2.5'te K/V'yi kopyalamadan GQA yapıyor
- Üretimde yalnızca son konumun logit'i hesaplanıyor (128K sözlükte büyük bellek tasarrufu)
- RoPE tabloları bf16'ya dönüştürmeden sonra fp32 olarak yeniden hesaplanıyor
- Tokenizer sözlüğü modelden farklıysa model sözlüğü otomatik uyarlanıyor
- Yetersiz veri için açıklayıcı hata mesajı

## [0.1.0] — 2026-11-26

### Eklendi
- Llama 3 tarzı Transformer: RMSNorm, RoPE (Llama 3.1 ölçeklemesi), GQA, SwiGLU, KV-cache
- Eğitim, veri hazırlama ve üretim betikleri
- 1B ve 8B ayarları, parametre dökümü, matematik dokümanı
