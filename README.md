# KökLM

**Milyar parametrelik LLM'leri kökünden çözmek için sıfırdan yazdığım dil modeli.**

KökLM, büyük dil modellerini kara kutu olmaktan çıkarmak için PyTorch ile sıfırdan
yazılmış, Llama 3 tarzı bir dil modelidir.

Mimari, Meta'nın açık kaynak **Llama 3** ailesinden esinlenmiştir:
decoder-only transformer · RMSNorm (pre-norm) · RoPE (Llama 3.1 ölçeklemesi) ·
Grouped-Query Attention · SwiGLU · KV-cache. Her bileşenin matematiği
[`docs/matematik.md`](docs/matematik.md) içinde.

## Ayarlar

| Ayar | Parametre | Boyutlar (d / L / H / Hkv / f) | Referans |
|---|---|---|---|
| `kucuk` | 2 885 888 | 256 / 4 / 8 / 2 / 704 | tek GPU, dakikalar |
| `1B` | 1 235 814 400 | 2048 / 16 / 32 / 8 / 8192 | Llama 3.2 1B |
| `8B` | 8 030 261 248 | 4096 / 32 / 32 / 8 / 14336 | Llama 3 8B |

```bash
python parametre_say.py 1B      # 1,24 milyar parametreyi kalem kalem gör (torch gerekmez)
```

## Kurulum

```bash
pip install -r requirements.txt
pytest -q
```

## Kullanım

```bash
# 1) Veri: metinleri token'a çevir
python prepare_data.py --girdi metinler/ --cikti data/tr --tokenizer byte

# 2) Eğitim (tek GPU)
python train.py --model kucuk --data data/tr

#    Çok GPU, 1B
torchrun --standalone --nproc_per_node=8 train.py --model 1B --data data/tr \
    --seq_len 2048 --batch_size 8 --grad_accum 8 --compile --grad_checkpoint

# 3) Üretim
python generate.py --ckpt checkpoints/en_iyi.pt --prompt "Bir varmış bir yokmuş"
```

## Donanım kestirimi

| | `kucuk` | `1B` | `8B` |
|---|---|---|---|
| Ağırlık (bf16) | 6 MB | 2,3 GiB | 15 GiB |
| AdamW ile eğitim (~16 B/param, aktivasyon hariç) | 46 MB | ~18 GiB | ~120 GiB |
| Uygun donanım | herhangi bir GPU / CPU | 1× A100 80GB veya 8× 24GB | çoklu A100/H100 (FSDP) |

## Dosyalar

```
model/config.py      hiperparametreler, FFN boyut formülü, hazır ayarlar
model/model.py       RMSNorm, RoPE, GQA attention, SwiGLU, Transformer, KV-cache, üretim
train.py             AdamW, ısınma + kosinüs LR, bf16, grad biriktirme, DDP, compile, checkpoint
prepare_data.py      metin -> uint32 token dosyaları
tokenizer.py         byte tokenizer / Hugging Face tokenizer
generate.py          checkpoint'ten metin üretimi (top-p örnekleme)
parametre_say.py     parametre dökümü + bellek kestirimi
docs/matematik.md    denklemler ve türetimler
tests/               şekil, nedensellik, KV-cache, RoPE, meta-cihazda 1B/8B sayımı
```

## Lisans

MIT — bkz. [LICENSE](LICENSE).
