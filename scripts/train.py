"""
Eğitim döngüsü.

Tek GPU:
    python scripts/train.py --model kucuk --data data/tr
Çok GPU (aynı makine, 8 GPU):
    torchrun --standalone --nproc_per_node=8 scripts/train.py --model 1B --data data/tr --compile

İçerik: AdamW, ısınma + kosinüs öğrenme oranı, bf16 autocast, gradyan biriktirme,
gradyan kırpma, gradient checkpointing, torch.compile, DDP, checkpoint kaydet/devam et.
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from koklm import ModelConfig, Transformer, ayar_al


def argumanlar():
    ap = argparse.ArgumentParser()
    # model / veri
    ap.add_argument("--model", default="kucuk", help="kucuk | 1B | 8B")
    ap.add_argument("--data", required=True, help="prepare_data.py çıktı klasörü")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true", help="out_dir/son.pt'den devam et")
    # batch: token/adım = batch_size * seq_len * grad_accum * gpu_sayısı
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=4)
    # optimizasyon
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # sistem
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
class Veri:
    """Token dosyasından rastgele (x, y) pencereleri. y = x'in bir sağa kaydırılmışı."""

    def __init__(self, klasor: str, seq_len: int, batch_size: int, device: str):
        meta = json.loads((Path(klasor) / "meta.json").read_text())
        self.meta, self.klasor = meta, Path(klasor)
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        for bolum in ("train", "val"):
            n = meta[f"{bolum}_tokens"]
            if n <= seq_len + 1:
                raise ValueError(f"{bolum} verisi {n} token; seq_len={seq_len} için çok az. "
                                 "Daha fazla metin ekleyin veya --seq_len'i küçültün.")

    def batch(self, bolum: str):
        # Her seferinde yeniden memmap: uzun eğitimlerde bellek sızıntısını önler
        data = np.memmap(self.klasor / f"{bolum}.bin", dtype=np.uint32, mode="r")
        ix = np.random.randint(0, len(data) - self.seq_len - 1, self.batch_size)
        pencere = np.stack([data[i:i + self.seq_len + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(np.ascontiguousarray(pencere[:, :-1]))
        y = torch.from_numpy(np.ascontiguousarray(pencere[:, 1:]))
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)


# ---------------------------------------------------------------------------
# Öğrenme oranı: doğrusal ısınma + kosinüs azalma
# ---------------------------------------------------------------------------
def ogrenme_orani(adim: int, a) -> float:
    if adim < a.warmup:
        return a.lr * (adim + 1) / a.warmup
    if adim >= a.max_steps:
        return a.min_lr
    oran = (adim - a.warmup) / max(1, a.max_steps - a.warmup)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * oran)) * (a.lr - a.min_lr)


def optimizer_kur(model: Transformer, a, device_type: str):
    """2B+ ağırlıklara weight decay, norm ve 1B vektörlere değil."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    gruplar = [{"params": decay, "weight_decay": a.weight_decay},
               {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(gruplar, lr=a.lr, betas=(a.beta1, a.beta2),
                             fused=(device_type == "cuda"))


def main():
    a = argumanlar()

    # ---- Dağıtık kurulum ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    ana = rank == 0
    device_type = "cuda" if device.startswith("cuda") else device
    torch.manual_seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = getattr(torch, a.dtype)
    ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(a.dtype == "float16"))

    # ---- Veri ve model ----
    veri = Veri(a.data, a.seq_len, a.batch_size, device)
    out_dir = Path(a.out_dir)
    if ana:
        out_dir.mkdir(parents=True, exist_ok=True)

    son_ckpt = out_dir / "son.pt"
    ckpt = torch.load(son_ckpt, map_location="cpu") if a.resume and son_ckpt.exists() else None
    if ckpt is not None:
        cfg = ModelConfig.from_dict(ckpt["config"])
    else:
        degisiklik = {"max_seq_len": a.seq_len}
        tok_V = veri.meta["vocab_size"]
        if tok_V != ayar_al(a.model).vocab_size:
            # Sözlüğü tokenizer'a uydur; 64'ün katı GPU matris çarpımlarında daha hızlı
            degisiklik["vocab_size"] = 64 * ((tok_V + 63) // 64)
        cfg = ayar_al(a.model, **degisiklik)
    assert veri.meta["vocab_size"] <= cfg.vocab_size, \
        f"tokenizer sözlüğü ({veri.meta['vocab_size']}) modelden büyük ({cfg.vocab_size})"

    model = Transformer(cfg).to(device)
    model.grad_checkpoint = a.grad_checkpoint
    optimizer = optimizer_kur(model, a, device_type)
    adim, en_iyi_val = 0, float("inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        adim, en_iyi_val = ckpt["adim"], ckpt["en_iyi_val"]
        del ckpt

    if ana:
        tok_adim = a.batch_size * a.seq_len * a.grad_accum * world
        print(f"model={cfg.isim}  parametre={model.parametre_sayisi():,}  cihaz={device}x{world}")
        print(f"adım başına token={tok_adim:,}  toplam≈{tok_adim * a.max_steps / 1e9:.2f}B token")

    egitim_modeli = torch.compile(model) if a.compile else model
    if ddp:
        egitim_modeli = DDP(egitim_modeli, device_ids=[local_rank])

    @torch.no_grad()
    def degerlendir():
        model.eval()
        sonuc = {}
        for bolum in ("train", "val"):
            kayiplar = torch.zeros(a.eval_iters)
            for i in range(a.eval_iters):
                x, y = veri.batch(bolum)
                with ctx:
                    _, loss = model(x, y)
                kayiplar[i] = loss.item()
            sonuc[bolum] = kayiplar.mean().item()
        model.train()
        return sonuc

    log_dosyasi = out_dir / "log.jsonl"

    def logla(**kayit):
        with open(log_dosyasi, "a", encoding="utf-8") as f:
            f.write(json.dumps(kayit) + "\n")

    def kaydet(yol: Path):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": cfg.to_dict(), "adim": adim, "en_iyi_val": en_iyi_val,
                    "veri_meta": veri.meta, "args": vars(a)}, yol)

    # ---- Eğitim döngüsü ----
    x, y = veri.batch("train")
    t0 = time.time()
    while adim < a.max_steps:
        lr = ogrenme_orani(adim, a)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if adim % a.eval_interval == 0 and adim > 0 and ana:
            k = degerlendir()
            print(f"[değerlendirme] adım {adim}: train {k['train']:.4f}  val {k['val']:.4f}  "
                  f"(perplexity {math.exp(k['val']):.1f})")
            logla(adim=adim, train_kayip=k["train"], val_kayip=k["val"])
            if k["val"] < en_iyi_val:
                en_iyi_val = k["val"]
                kaydet(out_dir / "en_iyi.pt")
            kaydet(son_ckpt)

        for mikro in range(a.grad_accum):
            if ddp:  # gradyanları sadece son mikro-adımda GPU'lar arası senkronla
                egitim_modeli.require_backward_grad_sync = (mikro == a.grad_accum - 1)
            with ctx:
                _, loss = egitim_modeli(x, y)
                loss = loss / a.grad_accum
            x, y = veri.batch("train")       # GPU hesaplarken bir sonrakini hazırla
            scaler.scale(loss).backward()

        if a.grad_clip > 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if adim % a.log_interval == 0 and ana:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = a.batch_size * a.seq_len * a.grad_accum * world * a.log_interval / max(dt, 1e-9)
            gn = f"  |g| {norm.item():.2f}" if a.grad_clip > 0 else ""
            logla(adim=adim, kayip=loss.item() * a.grad_accum, lr=lr, token_s=tok_s)
            print(f"adım {adim:6d}  kayıp {loss.item() * a.grad_accum:.4f}  lr {lr:.2e}{gn}  "
                  f"{tok_s:,.0f} token/s")
        adim += 1

    if ana:
        kaydet(son_ckpt)
        print("bitti ->", son_ckpt)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
