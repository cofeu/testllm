"""Hiperparametreler: mimari, optimizasyon ve uretim ayarlari.

Kanonik tanimlar burada yasar; `src/pt/decoder_model.py` ve `src/py/config.py`
bu modulu yeniden disa aktarir.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # ---- mimari ----
    vocab_size: int = 32
    d_model: int = 128
    n_layers: int = 8
    n_head: int = 8
    n_kv_head: int = 0          # 0 -> GQA yok (n_head'e esit)
    ff_dim: int = 512           # SwiGLU gizli boyutu
    block_size: int = 128
    dropout: float = 0.10
    tie_weights: bool = True
    bias: bool = False
    use_rope: bool = True       # mutlak pozisyon yerine RoPE
    rope_theta: float = 10000.0
    grad_checkpointing: bool = False  # bellek icin aktivasyon yeniden hesaplama

    # ---- optimizasyon ----
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    label_smoothing: float = 0.0
    batch_size: int = 12
    grad_accum: int = 1
    epochs: int = 200           # toplam optimizasyon adimi
    fused_adamw: bool = True    # CUDA'da fused AdamW (varsa)

    # ---- degerlendirme / kayit ----
    val_frac: float = 0.1
    eval_every: int = 50
    eval_iters: int = 20
    save_every: int = 100
    log_every: int = 10

    # ---- veri akisi ----
    pack_sequences: bool = True  # ornekleri bloklara paketle (dolgusuz)

    # ---- calisma ortami ----
    seed: int = 7
    device: str = "auto"
    amp: bool = True
    compile: bool = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 0.28
    top_k: int = 12
    top_p: float = 0.82
    repetition_penalty: float = 1.02
    min_p: float = 0.03
    sampler: str = "top_p"       # "top_p" | "typical" | "mirostat"
    typical_p: float = 0.95
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
