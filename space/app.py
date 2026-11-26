"""KökLM canlı sohbet: Llama 3.2 1B ağırlıkları, KökLM'in kendi mimarisinde (CPU)."""
import os

import gradio as gr
import torch

from koklm.model import ornekle
from koklm.tokenizer import tokenizer_al
from koklm.yukle import hf_yukle

MODEL = os.environ.get("MODEL", "unsloth/Llama-3.2-1B-Instruct")
SISTEM = ("Sen KökLM'sin: Llama 3.2 1B ağırlıklarıyla çalışan, sıfırdan yazılmış bir dil modeli "
          "mimarisi. Kısa, açık ve doğru cevap ver. "
          "Kullanıcı hangi dilde yazarsa o dilde cevap ver.")
torch.set_num_threads(os.cpu_count() or 2)

print(f"yükleniyor: {MODEL}")
model = hf_yukle(MODEL, dtype=torch.float32, device="cpu")
tok = tokenizer_al(f"hf:{MODEL}")
DUR = set(tok.dur_ids)
print(f"hazır: {model.parametre_sayisi():,} parametre")


def metin(icerik):
    if isinstance(icerik, str):
        return icerik
    if isinstance(icerik, list):
        return " ".join(p.get("text", "") for p in icerik if isinstance(p, dict))
    return str(icerik or "")


@torch.inference_mode()
def sohbet(mesaj, gecmis, max_yeni=256, sicaklik=0.6):
    mesajlar = [{"role": "system", "content": SISTEM}]
    for m in (gecmis or [])[-8:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            mesajlar.append({"role": m["role"], "content": metin(m.get("content"))})
    mesajlar.append({"role": "user", "content": metin(mesaj)})
    ids = tok.sohbet_kodla(mesajlar)
    T0 = len(ids)
    toplam = min(T0 + int(max_yeni), model.cfg.max_seq_len)
    model.cache_kur(1, toplam)
    try:
        logits, _ = model(torch.tensor([ids]), start_pos=0, sadece_son=True)
        cikti = []
        for pos in range(T0, toplam):
            son = logits[:, -1]
            if tok.vocab_size < son.size(-1):
                son[:, tok.vocab_size:] = float("-inf")
            nxt = ornekle(son, float(sicaklik), 0.9)
            t = int(nxt[0])
            if t in DUR:
                break
            cikti.append(t)
            yield tok.decode(cikti)
            logits, _ = model(nxt[:, None], start_pos=pos, sadece_son=True)
    finally:
        model.cache_sil()


ACIKLAMA = (f"**{model.parametre_sayisi():,} parametre** · Meta Llama 3.2 1B Instruct ağırlıkları, "
            "[KökLM](https://github.com/mkor4y/koklm)'in kendi PyTorch koduyla çalışıyor. "
            "Ücretsiz CPU'da çalıştığı için cevaplar yavaş akar; model küçük olduğundan "
            "İngilizce cevaplar Türkçeden daha tutarlıdır.")
ORNEKLER = ["Transformer mimarisinde attention ne işe yarar?",
            "Explain RoPE in two sentences.",
            "Bana kısa bir masal anlat."]
ek = [gr.Slider(32, 512, 256, step=32, label="En fazla yeni token"),
      gr.Slider(0.1, 1.2, 0.6, step=0.1, label="Sıcaklık")]
tema = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan")
kw = dict(title="KökLM · canlı sohbet", description=ACIKLAMA, examples=ORNEKLER,
          additional_inputs=ek, additional_inputs_accordion="Ayarlar")
try:
    demo = gr.ChatInterface(sohbet, type="messages", theme=tema, **kw)
except TypeError:
    demo = gr.ChatInterface(sohbet, **kw)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
