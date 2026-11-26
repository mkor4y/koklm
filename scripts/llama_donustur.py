"""
Gerçek Llama ağırlıklarını KökLM checkpoint'ine dönüştür.

Ağırlıklar Hugging Face'ten iner; model sayfasında Meta'nın lisansını kabul edip
`huggingface-cli login` ile giriş yapmış olmalısın.

    pip install -e ".[hf]"
    python scripts/llama_donustur.py --hf meta-llama/Llama-3.2-1B-Instruct \\
        --cikti checkpoints/llama32_1b_instruct.pt --dogrula

Çıktı dosyası:
    python scripts/generate.py --ckpt checkpoints/llama32_1b_instruct.pt --sohbet
    python scripts/train.py --baslangic checkpoints/llama32_1b_instruct.pt --data data/tr ...

--dogrula: aynı cümle üzerinde KökLM ile Hugging Face'in kendi Llama uygulamasının
çıktılarını karşılaştırır.
"""
import argparse
import time

import torch

from koklm.yukle import hf_yukle, koklm_kaydet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Hugging Face model adı veya yerel klasör")
    ap.add_argument("--cikti", default="checkpoints/llama32_1b.pt")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--dogrula", action="store_true", help="HF Llama ile çıktıları karşılaştır")
    a = ap.parse_args()

    dtype = getattr(torch, a.dtype)
    t0 = time.time()
    model = hf_yukle(a.hf, dtype=dtype, device="cpu")
    c = model.cfg
    print(f"yüklendi: {a.hf}  ({time.time() - t0:.0f} sn)")
    print(f"  parametre: {model.parametre_sayisi():,}")
    print(f"  d={c.dim} L={c.n_layers} H={c.n_heads} Hkv={c.n_kv_heads} f={c.ffn_dim} "
          f"V={c.vocab_size} tied={c.tied_embeddings} rope_scaling={bool(c.rope_scaling)}")

    if a.dogrula:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vdtype = dtype if dev == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(a.hf)
        ref = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=vdtype).to(dev).eval()
        ids = tok("Türkiye'nin başkenti Ankara'dır ve", return_tensors="pt").input_ids.to(dev)
        biz = model.to(device=dev, dtype=vdtype).rope_yenile()
        with torch.no_grad():
            l_ref = ref(ids).logits.float()
            l_biz, _ = biz(ids)
        fark = (l_ref - l_biz).abs().max().item()
        ayni = (l_ref.argmax(-1) == l_biz.argmax(-1)).float().mean().item()
        print(f"doğrulama ({dev}, {str(vdtype).split('.')[-1]}): en büyük logit farkı {fark:.3g}, "
              f"aynı tahmin oranı %{100 * ayni:.1f}")
        del ref
        model = model.to(device="cpu", dtype=dtype).rope_yenile()
        if dev == "cuda":
            torch.cuda.empty_cache()

    koklm_kaydet(model, a.cikti, tokenizer=f"hf:{a.hf}")
    print(f"kaydedildi: {a.cikti}")


if __name__ == "__main__":
    main()
