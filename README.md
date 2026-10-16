# TestLLM

Küçük, Türkçe instruction-tuned bir GPT dil modeli. İki bağımsız
implementasyon içerir; **kanonik olan PyTorch yoludur**.

## Gerçek kod yapısı

Kanonik PyTorch implementasyonu artık modüler bir pakettir:
**`src/py/normal/`**. Eski tek dosya (`src/pt/decoder_model.py`) yalnızca
geriye dönük uyumluluk için bu paketi yeniden dışa aktaran ince bir shim'dir.

| Modül | Açıklama |
|---|---|
| `src/py/normal/config.py` | `TrainConfig`, `GenerationConfig`. |
| `src/py/normal/tokenizer.py` | `BPETokenizer` (byte-level, özel chat token'ları, kelime-memoizasyonlu `encode`). |
| `src/py/normal/prompt.py` | `ChatPromptFormatter`, `clean_text`. |
| `src/py/normal/model.py` | `DecoderOnlyLM`: RMSNorm, RoPE, GQA, SwiGLU, KV cache, gradient checkpointing, gelişmiş örnekleme. |
| `src/py/normal/corpus.py` | Veri okuma, temizleme, instruction korpusu oluşturma. |
| `src/py/normal/dataset.py` | `InstructionDataset` — sequence packing + asistan-only loss. |
| `src/py/normal/train.py` | AdamW (fused), cosine LR, grad accumulation, eğitim, checkpoint. |
| `src/py/normal/generate.py` | `generate_text`. |
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

Varsayılan model (Llama tarzı): vocab 2048, d_model 256, 8 katman, 8 kafa,
2 KV kafası (GQA, `n_rep=4`), ff 1024 (SwiGLU), blok 256 (**RoPE**),
`tie_weights`, **~8.13M parametre**. Colab için ideal.

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

- **RoPE** (rotary positional embedding): mutlak pozisyon yerine; uzun bağlam
  genellemesi. `use_rope=False` ile öğrenilmiş pozisyon gömlemesine düşer.
- **GQA + KV cache**: `n_kv_head` ile grouped-query attention; `generate()`
  anahtar/değeri önbelleğe alır (`forward(..., caches=, return_cache=, pos_offset=)`).
- **SwiGLU + RMSNorm**: Llama tarzı FFN ve normalizasyon.
- **Sequence packing**: `InstructionDataset` tüm örnekleri dolgusuz tek akışa
  dizer; pencere başına token bütçesi tam dolu kalır.
- **Asistan-only loss**: yalnızca asistan token'ları etiketlenir; system/user ve
  prompt token'ları `-100` olur (`F.cross_entropy(ignore_index=-100)`).
- **Gradient checkpointing + grad accumulation**: `grad_checkpointing=True` ve
  `grad_accum>1` ile daha büyük model / daha az VRAM.
- **Optimizer**: CUDA'da fused AdamW, yapılandırılabilir `betas`/`eps`,
  ağırlık grubu ayrımı ve cosine + warmup LR zamanlaması.
- **Örnekleme**: `sampler="top_p"` (varsayılan), `"typical"` veya `"mirostat"`
  (v2); ayrıca `top_k`, `top_p`, `min_p`, `repetition_penalty`.
- **Özel token'lar**: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3,
  `<|im_start|>`=4, `<|im_end|>`=5, bayt aralığı 6'dan başlar (`OFFSET=6`).

Geliştirme aşamasındadır.
