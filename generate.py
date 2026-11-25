"""
Eğitilmiş modelden metin üret.

    python generate.py --ckpt checkpoints/en_iyi.pt --prompt "Bir varmış bir yokmuş"
"""
import argparse

import torch

from model import ModelConfig, Transformer
from tokenizer import tokenizer_al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=1, help="kaç örnek")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, a.dtype) if device != "cpu" else torch.float32

    ckpt = torch.load(a.ckpt, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device=device, dtype=dtype).eval()

    tok = tokenizer_al(ckpt["veri_meta"]["tokenizer"])
    ids = tok.encode(a.prompt) or [0]
    prompt = torch.tensor([ids] * a.n, dtype=torch.long, device=device)

    out = model.generate(prompt, a.max_new_tokens, a.temperature, a.top_p, tok.eos_id)
    for i, satir in enumerate(out.tolist()):
        print(f"--- örnek {i + 1} ---")
        print(tok.decode(satir))


if __name__ == "__main__":
    main()
