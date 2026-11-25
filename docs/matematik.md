# KökLM — Matematik

Gösterim: `B` batch, `T` dizi uzunluğu, `d` gizli boyut, `H` sorgu kafası, `Hkv` key/value kafası,
`dh = d/H` kafa boyutu, `f` FFN boyutu, `V` sözlük, `L` blok sayısı.
Ağırlık kuralı: `y = x W`, `W ∈ ℝ^(giriş × çıkış)`.

---

## 0. Büyük resim

Model tek bir şeyi öğrenir: önceki token'lara bakarak bir sonrakinin olasılık dağılımı.

```
P(x_1, …, x_T) = ∏_t P(x_t | x_1, …, x_{t-1})
```

Eğitim hedefi (çapraz entropi / negatif log-olabilirlik):

```
ℒ = −(1/T) Σ_t log softmax(z_t)[x_{t+1}]
```

Akış:

```
token id'leri (B,T)
   │  E[x]                       embedding tablosu E ∈ ℝ^(V×d)
   ▼
h₀ (B,T,d)
   │  L kez:
   │    h  ← h + Attn(RMSNorm(h))
   │    h  ← h + FFN(RMSNorm(h))
   ▼
RMSNorm → z = h W_out  (B,T,V)   (tied: W_out = Eᵀ)
```

---

## 1. Embedding

`h₀[b,t] = E[x_{b,t}]` — bir tablodan satır seçmek. Matematiksel olarak one-hot vektör ile
çarpmaya eşittir: `onehot(x) · E`. Parametre: `V·d`.

1B modelde `V = 128 256`, `d = 2048` → **262 668 288** parametre (modelin %21'i!).
Küçük modellerde sözlük, parametrelerin ciddi bir kısmını yer. Bu yüzden 1B'de
**tied embeddings** kullanılır: çıkış katmanı `W_out = Eᵀ`, ayrı matris yok.

---

## 2. RMSNorm

```
RMSNorm(x) = x / √( (1/d) Σ_i x_i² + ε ) ⊙ g
```

LayerNorm'dan farkı: ortalamayı çıkarmaz ve bias yoktur. Sadece ölçeği sabitler.
Daha ucuzdur, pratikte aynı işi görür. Parametre: `g ∈ ℝ^d`.

**Pre-norm:** Normalizasyon alt katmana girmeden önce yapılır, residual yol ise
normalize edilmez. Böylece `h_L = h_0 + Σ (katman çıktıları)` şeklinde temiz bir toplam
yolu kalır ve gradyan derin ağda kaybolmadan akar.

---

## 3. RoPE (Rotary Position Embedding)

Attention kendi başına sıra bilmez (permütasyona duyarsız). Konumu, Q ve K vektörlerini
döndürerek ekleriz. Vektörü `dh/2` tane 2B çifte böl; `i`. çifti, `m` konumunda
`m·ω_i` açısıyla döndür:

```
ω_i = θ^(−2i/dh)        θ = 500 000 (Llama 3)

[q'_{2i}  ]   [cos mω_i  −sin mω_i] [q_{2i}  ]
[q'_{2i+1}] = [sin mω_i   cos mω_i] [q_{2i+1}]
```

**Neden işe yarar:** Dönme matrisleri için `R(a)ᵀ R(b) = R(b − a)`. O halde

```
⟨R(m)q, R(n)k⟩ = qᵀ R(m)ᵀ R(n) k = qᵀ R(n − m) k
```

Skor yalnızca **göreli konum** `n − m`'ye bağlıdır. (Test: `test_rope_goreli_konum`.)
Düşük `i` → hızlı dönen çiftler (yakın ilişki), yüksek `i` → yavaş dönen (uzak bağlam).
Büyük θ, en yavaş frekansı daha da yavaşlatır → uzun bağlam.

**Llama 3.1+ ölçeklemesi (1B ayarında açık):** 8K ile eğitilmiş modeli 128K'ya
uzatmak için her frekansın dalga boyuna `λ_i = 2π/ω_i` bakılır (`s = 32`, `C = 8192`):

```
λ_i < C/4   →  ω_i            (yüksek frekans: yerel bilgi, dokunma)
λ_i > C/1   →  ω_i / s        (düşük frekans: 32 kat yavaşlat)
arada       →  (1−α)·ω_i/s + α·ω_i,   α = (C/λ_i − 1) / (4 − 1)
```

Parametre: **0** (tamamen sabit formül).

---

## 4. Attention (GQA — Grouped Query Attention)

```
Q = X W_q  ∈ ℝ^(T × H·dh)        W_q ∈ ℝ^(d × d)
K = X W_k  ∈ ℝ^(T × Hkv·dh)      W_k ∈ ℝ^(d × Hkv·dh)
V = X W_v  ∈ ℝ^(T × Hkv·dh)      W_v ∈ ℝ^(d × Hkv·dh)
```

Her kafa `j` için (K,V kafası `⌊j / (H/Hkv)⌋` paylaşılır):

```
A_j = softmax( Q_j K_jᵀ / √dh + M )        M_{ts} = 0 (s ≤ t),  −∞ (s > t)
O_j = A_j V_j
Çıktı = concat(O_1 … O_H) W_o              W_o ∈ ℝ^(d × d)
```

- **√dh ile bölme:** `q, k` bileşenleri varyans 1 ise `q·k`'nın varyansı `dh` olur.
  Bölmezsek softmax doyar (tek bir 1, gerisi 0) ve gradyan ölür.
- **Nedensel maske M:** `−∞`, softmax'ta `e^{−∞} = 0` verir → geleceğe bakış yok.
  (Test: `test_nedensellik`.)
- **GQA:** 1B'de `H = 32`, `Hkv = 8` → 4 sorgu kafası bir K/V kafasını paylaşır.
  K/V parametreleri ve çıkarımdaki **KV-cache** belleği 4 kat küçülür, kalite neredeyse aynı.

Parametre (blok başına): `d² + 2·d·Hkv·dh + d² = 2d² + 2d·(Hkv·dh)`
1B: `2·2048² + 2·2048·512` = **10 485 760**

Hesap maliyeti: skor matrisi `T × T` → attention `O(T²·d)`. Uzun bağlamın pahalı olma sebebi.

---

## 5. FFN — SwiGLU

```
FFN(x) = ( SiLU(x W_1) ⊙ x W_3 ) W_2       SiLU(u) = u · σ(u)
W_1, W_3 ∈ ℝ^(d × f),   W_2 ∈ ℝ^(f × d)
```

`x W_3` bir "kapı" ile çarpılıyor: ağ, hangi özelliklerin geçeceğini girdiye göre seçiyor.

**f nasıl seçilir?** Klasik FFN `f = 4d` ile 2 matris kullanır → `8d²` parametre.
SwiGLU 3 matris kullandığı için eşitlemek adına `f = (2/3)·4d`. Sonra çarpan ve yuvarlama:

```
1B: 4·2048 = 8192 → ·2/3 = 5461 → ·1.5 = 8191 → 256'nın katı → f = 8192
8B: 4·4096 = 16384 → ·2/3 = 10922 → ·1.3 = 14198 → 1024'ün katı → f = 14336
```

Parametre: `3·d·f`. 1B: `3·2048·8192` = **50 331 648** → bloğun **%83'ü** FFN!
Yani "bilgi"nin çoğu attention'da değil FFN'de saklanır.

---

## 6. Toplam parametre formülü

```
N = V·d                                  (embedding)
  + L · [ 2d² + 2d·Hkv·dh + 3·d·f + 2d ]  (bloklar)
  + d                                    (son norm)
  + (tied değilse) d·V                   (çıkış)
```

| | 1B | 8B |
|---|---|---|
| V | 128 256 | 128 256 |
| d | 2048 | 4096 |
| L | 16 | 32 |
| H / Hkv | 32 / 8 | 32 / 8 |
| f | 8192 | 14 336 |
| tied | evet | hayır |
| **N** | **1 235 814 400** | **8 030 261 248** |

Her iki sayı da Meta'nın yayımladığı değerlerle birebir aynı (`test_bilinen_parametre_sayilari`).

---

## 7. Hesap ve bellek kestirimi

- **İleri geçiş:** token başına ≈ `2N` FLOP (her parametre bir çarpma + bir toplama).
- **Eğitim:** ileri + geri ≈ `6N` FLOP/token. 1B modeli 1 trilyon token ile eğitmek:
  `6 · 1.24e9 · 1e12 ≈ 7.4e21` FLOP.
- **Bellek (ağırlık):** fp32 4 B, bf16 2 B, int4 0.5 B / parametre.
- **Adam ile eğitim:** ağırlık + gradyan + 2 moment ≈ 16 B/parametre (+ aktivasyonlar).
- **KV-cache:** `2 · L · Hkv · dh · T · bayt`. 1B, T = 8192, bf16:
  `2·16·8·64·8192·2` ≈ 256 MiB. GQA olmasaydı (Hkv = 32) 4 katı.

---

## 8. Eğitimin matematiği (`scripts/train.py`)

**Kayıp:** `ℒ = −(1/BT) Σ log p(x_{t+1} | x_{≤t})`. Rastgele başlangıçta her token eşit
olasılıklı → `ℒ₀ ≈ ln V` (byte modeli: ln 256 ≈ 5.55). Perplexity = `e^ℒ`.

**Başlangıç ağırlıkları:** `W ~ 𝒩(0, 0.02²)`. Residual'a yazan `W_o` ve `W_2` için
`σ = 0.02/√(2L)`: her blok residual'a 2 terim ekler, `2L` bağımsız terimin toplamının
varyansı `2L·σ²` olur; böylece derinlik arttıkça çıktının ölçeği sabit kalır.

**AdamW:**

```
m_t = β₁ m_{t−1} + (1−β₁) g_t           (momentum)
v_t = β₂ v_{t−1} + (1−β₂) g_t²          (ölçek)
m̂ = m_t/(1−β₁ᵗ),  v̂ = v_t/(1−β₂ᵗ)       (sapma düzeltme)
θ ← θ − η ( m̂/(√v̂ + ε) + λ θ )          (λ: weight decay, sadece matrislere)
```

β₁ = 0.9, β₂ = 0.95, λ = 0.1 (Llama değerleri).

**Öğrenme oranı:** ilk `w` adım doğrusal ısınma `η = η_max·(t+1)/w`, sonra kosinüs:
`η = η_min + ½(1 + cos(π·p))(η_max − η_min)`, `p ∈ [0,1]` ilerleme oranı.

**Gradyan kırpma:** `‖g‖ > c` ise `g ← g·c/‖g‖` (c = 1). Ani patlamaları keser.

**Gradyan biriktirme:** `k` mikro-batch'in kaybı `1/k` ile ölçeklenip toplanır →
matematiksel olarak `k` kat büyük batch ile aynı gradyan.

**Gradient checkpointing:** ileri geçişte blok içi aktivasyonlar saklanmaz, geri
geçişte yeniden hesaplanır. Bellek `O(L)` blok çıktısına iner, hesap ~%33 artar.

---

## 9. Tokenizer: Byte Pair Encoding (`koklm/bpe.py`)

Model sayılarla çalışır; metni sayılara çeviren şey tokenizer'dır.

1. Başlangıç sözlüğü 256 bayt → her UTF-8 metin temsil edilebilir, "bilinmeyen token" yok.
2. Metni ön-parçalara böl (kelime + önündeki boşluk, sayı, noktalama).
3. Tekrarla: en sık komşu çifti `(a, b)` bul, yeni token `n = 256 + k` ata, her yerde birleştir.
4. Kodlarken birleştirmeleri öğrenilme sırasına göre uygula.

Her birleştirme diziyi kısaltır: token başına düşen bayt artar → aynı bağlam penceresine
daha çok metin sığar, ama `V` büyüdükçe embedding (`V·d`) ve çıkış katmanı büyür.
Bu, sözlük boyutunun temel takasıdır. Llama 3: `V = 128 256`.

---

## 10. Gerçek Llama ağırlıkları ve RoPE permütasyonu (`koklm/yukle.py`)

Meta'nın orijinal kodu RoPE çiftlerini `(x_{2i}, x_{2i+1})` olarak alır; Hugging Face
`(x_i, x_{i+dh/2})` ("rotate_half"). HF dönüştürücüsü bunu dengelemek için her kafada
`W_q` ve `W_k` satırlarını yeniden sıralar:

```
Meta satırı 2i + j   →   HF satırı j·(dh/2) + i,      j ∈ {0, 1}
```

Bu bir permütasyon matrisi `P` ile `W_q^HF = P W_q`, `W_k^HF = P W_k` demektir. `P` dik
(`PᵀP = I`) ve aynı anda hem Q'ya hem K'ya uygulandığından

```
(P q)ᵀ R'(n−m) (P k) = qᵀ R(n−m) k        (R' = P R Pᵀ: rotate_half düzeninde dönme)
```

yani attention skorları değişmez. KökLM, `Pᵀ`'yi uygulayarak Meta düzenine geri döner ve
Hugging Face'in Llama'sıyla **aynı logit'leri** üretir (`tests/test_yukle.py`).

---

## 11. Sonraki adımlar

1. **FSDP:** 8B ölçeğinde parametre/optimizer durumunu GPU'lara bölmek.
2. **Türkçe ön-eğitim:** `kucuk` → ~100M → 1B ölçek deneyleri, ölçekleme yasaları.
3. **İnce ayar:** sohbet formatı (SFT), LoRA.
4. **Nicemleme:** int8/int4 çıkarım.
