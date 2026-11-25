"""
Metin üret: kendi eğittiğin KökLM checkpoint'inden ya da gerçek Llama 3 ağırlıklarından.

    # Kendi modelin
    python scripts/generate.py --ckpt checkpoints/en_iyi.pt --prompt "Bir varmış"

    # Llama 3.2 1B ağırlıklarını KökLM mimarisinde çalıştır (pip install "koklm[hf]")
    python scripts/generate.py --hf meta-llama/Llama-3.2-1B --prompt "The capital of Turkey is"

    # Meta'nın orijinal indirmesi (consolidated.00.pth + params.json + tokenizer.model)
    python scripts/generate.py --meta ~/.llama/checkpoints/Llama3.2-1B \
        --tokenizer hf:meta-llama/Llama-3.2-1B --prompt "..."
"""
import argparse

import torch

from koklm import ModelConfig, Transformer
from koklm.tokenizer import tokenizer_al


def cihaz_sec():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    kaynak = ap.add_mutually_exclusive_group(required=True)
    kaynak.add_argument("--ckpt", help="train.py checkpoint'i")
    kaynak.add_argument("--hf", help="Hugging Face Llama modeli (Hub adı veya klasör)")
    kaynak.add_argument("--meta", help="Meta orijinal checkpoint klasörü")
    ap.add_argument("--tokenizer", help="varsayılanı kaynaktan çıkarılır")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=1, help="kaç örnek")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = cihaz_sec()
    dtype = getattr(torch, a.dtype) if device != "cpu" else torch.float32

    if a.ckpt:
        ckpt = torch.load(a.ckpt, map_location="cpu")
        model = Transformer(ModelConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        model = model.to(device=device, dtype=dtype).rope_yenile().eval()
        tok_adi = a.tokenizer or ckpt["veri_meta"]["tokenizer"]
    elif a.hf:
        from koklm.yukle import hf_yukle
        model = hf_yukle(a.hf, dtype=dtype, device=device)
        tok_adi = a.tokenizer or f"hf:{a.hf}"
    else:
        from koklm.yukle import meta_yukle
        model = meta_yukle(a.meta, dtype=dtype, device=device)
        if not a.tokenizer:
            ap.error("--meta ile --tokenizer verin (örn. hf:meta-llama/Llama-3.2-1B)")
        tok_adi = a.tokenizer

    print(f"model: {model.cfg.isim}  parametre: {model.parametre_sayisi():,}  cihaz: {device}")
    tok = tokenizer_al(tok_adi)
    ids = tok.encode(a.prompt)
    bos = getattr(tok, "bos_id", None)
    if bos is not None:
        ids = [bos] + ids
    if not ids:
        ids = [0]
    prompt = torch.tensor([ids] * a.n, dtype=torch.long, device=device)

    out = model.generate(prompt, a.max_new_tokens, a.temperature, a.top_p, tok.eos_id)
    for i, satir in enumerate(out.tolist()):
        print(f"\n--- örnek {i + 1} ---")
        print(tok.decode(satir))


if __name__ == "__main__":
    main()
