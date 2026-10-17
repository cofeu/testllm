# TestLLM

Küçük, Türkçe instruction-tuned bir dil modeli. Mimari **encoder-decoder
(seq2seq)**'dir: encoder çift yönlü olarak bağlamı kodlar, decoder nedensel
self-attention + **cross-attention** ile cevabı token-token üretir. Bileşenler
**Llama + GPT hibrididir** (varsayılan Llama tarzı, `norm_type`/`activation`/
`use_rope` ile GPT tarzına geçilebilir). İki bağımsız implementasyon içerir;
**kanonik olan PyTorch yoludur**.

## Gerçek kod yapısı

Kanonik PyTorch implementasyonu artık modüler bir pakettir:
**`src/py/normal/`**. Eski tek dosya (`src/pt/decoder_model.py`) yalnızca
geriye dönük uyumluluk için bu paketi yeniden dışa aktaran ince bir shim'dir.

| Modül | Açıklama |
|---|---|
| `src/py/normal/config.py` | `TrainConfig`, `GenerationConfig`. |
| `src/py/normal/tokenizer.py` | `BPETokenizer` (byte-level, özel chat token'ları, kelime-memoizasyonlu `encode`). |
| `src/py/normal/prompt.py` | `ChatPromptFormatter`, `clean_text`. |
| `src/py/normal/model.py` | `EncoderDecoderLM`: encoder + decoder blokları, `Attention` (nedensel/cross), RMSNorm/LayerNorm, RoPE/öğrenilmiş pozisyon, GQA, SwiGLU/GELU, QK-Norm, self+cross KV cache, z-loss, gradient checkpointing, gelişmiş örnekleme. |
| `src/py/normal/corpus.py` | Veri okuma, temizleme, instruction korpusu oluşturma. |
| `src/py/normal/dataset.py` | `Seq2SeqDataset` — kaynak/hedef çiftleri, teacher forcing, pad maskeleri. |
| `src/py/normal/train.py` | AdamW (fused), cosine LR, grad accumulation, eğitim, checkpoint, resume; CLI (`python -m src.py.normal.train`). |
| `src/py/normal/generate.py` | `generate_text` + CLI (`python -m src.py.normal.generate`). |
| `src/py/normal/__main__.py` | `python -m src.py.normal` giriş noktası. |

Diğer dizinler:

| Dizin / dosya | Bağımlılık | Açıklama |
|---|---|---|
| `src/pt/decoder_model.py` | torch | **Shim**: `src.py.normal`'ı yeniden dışa aktarır; `python -m src.pt.decoder_model` hâlâ çalışır. |
| `src/py/custom/` | yalnızca Python stdlib | Sıfırdan yazılmış saf-Python transformer (matris, softmax, backward, AdamW elle). Bağımsızdır. |
| `src/py/config.py` | torch | `TrainConfig`'i genişleten `Config` + CLI arg çözümleyicileri. |
| `src/py/{train,generate,checkpoint}.py` | torch | İnce re-export (eski import yollarını korur). |
| `src/pt/mini_encoder.py` | torch | Bağımsız eğitici kodlayıcı demosu; `python -m src.pt.mini_encoder`. |

## Veri

| Dosya | İçerik |
|---|---|
| `data/data.jsonl` | Elle yazılmış küçük Türkçe diyalog seti. |
| `data/data_large.jsonl` | 100.000 diyalog (`TFLai/Turkish-Alpaca` + `merve/turkish_instructions`). |
| `data/tokenizer.json` | Önbelleklenmiş BPE tokenizer (vocab 2048). |
| `data/chatml_instruct.jsonl` | Eğitimde kullanılan temizlenmiş chatml çıktısı (otomatik üretilir). |

Format (her satır bir diyalog):

```json
{"turns": [["kullanıcı", "asistan"], ["kullanıcı", "asistan"]]}
```

## Kurulum

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt`: `torch`, `numpy`, `pytest`.

## Kullanım

### Kanonik PyTorch yolu (önerilen, GPU/CUDA/AMP destekli)

```
.venv/bin/python -m src.py.normal
```

Bu komut sırayla: `data/data.jsonl` + `data/data_large.jsonl` yükler →
`build_assistant_only_corpus` ile temizler → tokenizer'ı (`data/tokenizer.json`
varsa yükler, yoksa eğitip kaydeder) → modeli eğitir → örnek üretir →
checkpoint'i yeniden yükleyip üretimi doğrular. Checkpoint:
`decoder_model_ckpt.pt` (proje kökü).

Varsayılan model (Llama tarzı): vocab 2048, d_model 256, 8 encoder + 8 decoder
katmanı, 8 kafa, 2 KV kafası (GQA, `n_rep=4`), ff 1024 (SwiGLU), blok 256
(**RoPE**), `tie_weights`. Encoder-decoder olduğu için decoder katmanında ek
cross-attention vardır.

### CLI: eğitim ve üretim

```
# eğitim (argümanlarla; --resume ile kaldığı yerden devam)
.venv/bin/python -m src.py.normal.train --steps 4000 --qk-norm --batch-size 32
.venv/bin/python -m src.py.normal.train --resume

# GPT tarzı varyant (LayerNorm + GELU + öğrenilmiş pozisyon + bias)
.venv/bin/python -m src.py.normal.train --norm layernorm --activation gelu --no-rope

# üretim (tek seferlik veya etkileşimli)
.venv/bin/python -m src.py.normal.generate --prompt "Merhaba" --sampler typical
.venv/bin/python -m src.py.normal.generate --interactive
```

Öne çıkan eğitim argümanları: `--d-model --n-layers --n-head --n-kv-head
--ff-dim --block-size --lr --warmup --grad-accum --dropout --qk-norm --no-rope
--norm {rmsnorm,layernorm} --activation {swiglu,gelu} --grad-checkpointing
--resume --compile`.

### Saf Python yolu (bağımlılıksız, eğitim amaçlı)

```
python3 src/py/custom/test_grads.py                     # sonlu fark gradyan doğrulaması
python3 src/py/custom/train.py --steps 300 --batch 4 --d-model 64 --layers 3
python3 src/py/custom/train.py --resume --steps 500
python3 src/py/custom/generate.py --prompt "K: Merhaba\nA:" --interactive
```

### Testler

```
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

## Mimari ve performans notları

**Encoder-decoder (seq2seq) + Llama/GPT bileşenleri.** Encoder çift yönlü
self-attention kullanır (gelecek token'lar geçmişi etkiler); decoder nedensel
self-attention + encoder çıktısına **cross-attention** + FFN'den oluşur. GPT
tarzı bileşenler yapılandırma ile açılır:

| Özellik | Llama (varsayılan) | GPT |
|---|---|---|
| Normalizasyon | `norm_type="rmsnorm"` | `"layernorm"` |
| FFN aktivasyonu | `activation="swiglu"` | `"gelu"` |
| Pozisyon | `use_rope=True` (RoPE) | `False` (öğrenilmiş) |
| Bias | `bias=False` | `True` |
| Ortak bileşenler | pre-norm, dropout, weight tying, GQA, self+cross KV cache | |

- **Teacher forcing**: decoder girdisi `<bos>` ile başlar, hedefler bir
  kaydırılmış hâlidir; cevabın sonuna `<|im_end|>` (EOS) eklenir. Pad etiketleri
  `-100` olduğu için kayıp yalnızca gerçek cevap token'larından hesaplanır.
- **Pad maskeleri**: encoder dolgusu hem encoder self-attention'ında hem decoder
  cross-attention'ında maskelenir; decoder dolgusu self-attention'ında maskelenir.
  Boyut `F.scaled_dot_product_attention`'ın bool `attn_mask`'i ile verilir.
- **RoPE** (rotary positional embedding): mutlak pozisyon yerine; encoder ve
  decoder self-attention'ında uygulanır (cross-attention'da pozisyon yok).
  `use_rope=False` ile öğrenilmiş pozisyon gömlemesine düşer.
- **RoPE scaling**: `rope_scaling="linear"` + `rope_scaling_factor` ile doğrusal
  pozisyon enterpolasyonu (uzun bağlam ince-ayarı).
- **GQA + KV cache**: `n_kv_head` ile grouped-query attention; `generate()`
  decoder self-attention anahtar/değerini önbelleğe alır, encoder çıktısı bir kez
  kodlanıp cross-attention cache'i paylaşılır
  (`forward(..., caches=, return_cache=, pos_offset=)`). Cache'li ve cache'siz
  üretim deterministik seed altında aynı sonucu verir.
- **SwiGLU / GELU FFN**: `activation` ile seçilir; `default_swiglu_hidden(d_model)`
  Llama kuralıyla (~8/3·d_model, 64'ün katı) gizli boyutu verir.
- **Residual başlatma ölçekleme** yalnızca çıkış projeksiyonlarına
  (`attn.proj`, `self_attn.proj`, `cross_attn.proj`, `mlp.down`/`mlp.proj`)
  uygulanır; q/k/v projeksiyonları standart 0.02 ile başlar.
- **QK-Norm** (`qk_norm=True`): Llama 3 stabilitiesi için q/k üzerine RMSNorm.
- **z-loss** (`z_loss>0`): logit büyüklüğünü düzenleyen ek kayıp terimi; yalnızca
  `-100` olmayan (gerçek hedef) konumlarda hesaplanır.
- **Prefix-space tokenizer** (GPT-2 tarzı): kelimeye başındaki boşluk dahil
  edilir; tek başına boşluk tokeni ~%0,3'e iner ve aynı bağlama ~%27 daha fazla
  metin sığar (token/kelime 3,13 → 2,32).
- **Kaynak kırpma**: prompt blok boyutunu aşarsa **sondaki** (en güncel) bağlam
  korunur, böylece kullanıcının sorusu prompt sonunda olduğu için silinmez.
- **Tek system prompt kaynağı**: `prompt.DEFAULT_SYSTEM_PROMPT` (kısa); eğitim,
  `__main__`, CLI ve `format_chat_prompt` aynı sabiti kullanır. Çok-turlu
  geçmiş, güncel kullanıcı mesajından **önce** yerleştirilir.
- **Gradient checkpointing + grad accumulation**: `grad_checkpointing=True` ve
  `grad_accum>1` ile daha büyük model / daha az VRAM.
- **Optimizer**: CUDA'da fused AdamW, yapılandırılabilir `betas`/`eps`,
  ağırlık grubu ayrımı ve cosine + warmup LR zamanlaması.
- **Resume**: checkpoint optimizer + scaler + step durumunu saklar; `--resume` /
  `TrainConfig(resume=True)` ile kaldığı adımdan devam eder.
- **Örnekleme**: `sampler="top_p"` (varsayılan), `"typical"` veya `"mirostat"`
  (v2: `surprise > μ` olan token'lar elenir, sonra `μ -= η·(gözlenen surprise − τ)`);
  ayrıca `top_k`, `top_p`, `min_p`, `repetition_penalty`.
- **Özel token'lar**: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3,
  `<|im_start|>`=4, `<|im_end|>`=5, bayt aralığı 6'dan başlar (`OFFSET=6`).

Geliştirme aşamasındadır.
