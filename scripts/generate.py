"""
Metin üret ya da sohbet et: kendi KökLM checkpoint'inden ya da gerçek Llama 3 ağırlıklarından.

    # Kendi modelin
    python scripts/generate.py --ckpt checkpoints/en_iyi.pt --prompt "Bir varmış"

    # Llama ağırlıklarıyla (önce scripts/llama_donustur.py ya da doğrudan --hf)
    python scripts/generate.py --ckpt checkpoints/llama32_1b_instruct.pt --sohbet
    python scripts/generate.py --hf meta-llama/Llama-3.2-1B --prompt "The capital of Turkey is"

    # Meta'nın orijinal indirmesi (consolidated.00.pth + params.json)
    python scripts/generate.py --meta ~/.llama/checkpoints/Llama3.2-1B \
        --tokenizer hf:meta-llama/Llama-3.2-1B --prompt "..."
"""
import argparse

import torch

from koklm.tokenizer import tokenizer_al


def cihaz_sec():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_sec(istek: str, device: str):
    if device == "cpu":
        return torch.float32
    if istek == "auto":
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            return torch.float16          # örn. Colab T4
        return torch.bfloat16
    return getattr(torch, istek)


def main():
    ap = argparse.ArgumentParser()
    kaynak = ap.add_mutually_exclusive_group(required=True)
    kaynak.add_argument("--ckpt", help="KökLM checkpoint'i (train.py ya da llama_donustur.py)")
    kaynak.add_argument("--hf", help="Hugging Face Llama modeli (Hub adı veya klasör)")
    kaynak.add_argument("--meta", help="Meta orijinal checkpoint klasörü")
    ap.add_argument("--tokenizer", help="varsayılanı kaynaktan çıkarılır")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--sohbet", action="store_true", help="Instruct modelle etkileşimli sohbet")
    ap.add_argument("--sistem", default="Sen yardımsever bir asistansın. Türkçe cevap ver.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=1, help="kaç örnek")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = cihaz_sec()
    dtype = dtype_sec(a.dtype, device)

    if a.ckpt:
        from koklm.yukle import checkpoint_yukle
        model, ck = checkpoint_yukle(a.ckpt, dtype=dtype, device=device)
        tok_adi = a.tokenizer or ck["veri_meta"]["tokenizer"]
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

    print(f"model: {model.cfg.isim}  parametre: {model.parametre_sayisi():,}  "
          f"cihaz: {device}  dtype: {str(dtype).split('.')[-1]}")
    tok = tokenizer_al(tok_adi)
    dur = getattr(tok, "dur_ids", None) or tok.eos_id

    def uret(ids, n=1):
        prompt = torch.tensor([ids] * n, dtype=torch.long, device=device)
        out = model.generate(prompt, a.max_new_tokens, a.temperature, a.top_p, dur,
                             vocab_siniri=tok.vocab_size)
        return [row[len(ids):] for row in out.tolist()]

    if a.sohbet:
        if not hasattr(tok, "sohbet_kodla"):
            ap.error("--sohbet için Hugging Face tokenizer'lı bir Instruct model gerekir")
        mesajlar = [{"role": "system", "content": a.sistem}] if a.sistem else []
        print("Sohbet başladı. Çıkmak için boş satır ya da 'çık' yaz.\n")
        while True:
            try:
                soru = input("sen > ").strip()
            except EOFError:
                break
            if not soru or soru.lower() in {"çık", "cik", "exit", "quit"}:
                break
            mesajlar.append({"role": "user", "content": soru})
            yeni = uret(tok.sohbet_kodla(mesajlar))[0]
            cevap = tok.decode(yeni).strip()
            print(f"KökLM > {cevap}\n")
            mesajlar.append({"role": "assistant", "content": cevap})
        return

    ids = tok.encode(a.prompt)
    bos = getattr(tok, "bos_id", None)
    if bos is not None:
        ids = [bos] + ids
    if not ids:
        ids = [0]
    for i, yeni in enumerate(uret(ids, a.n)):
        print(f"\n--- örnek {i + 1} ---")
        print(a.prompt + tok.decode(yeni))


if __name__ == "__main__":
    main()
