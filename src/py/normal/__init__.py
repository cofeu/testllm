"""Kanonik PyTorch implementasyonu (Llama + GPT hibrit).

Llama: RMSNorm, RoPE, GQA, SwiGLU, pre-norm. GPT: LayerNorm, GELU, ogrenilmis
pozisyon, bias, dropout, weight tying -- `norm_type`/`activation`/`use_rope`/`bias`
ile secilir. Modern: QK-Norm (Llama 3), RoPE scaling, z-loss, KV cache,
gradient checkpointing, sequence packing, typical/mirostat ornekleme, resume.

Modul yapisi:
  config     : TrainConfig, GenerationConfig
  tokenizer  : BPETokenizer
  prompt     : ChatPromptFormatter, clean_text
  model      : DecoderOnlyLM (RMSNorm/LayerNorm, RoPE, GQA, SwiGLU/GELU, KV cache)
  corpus     : veri okuma/temizleme/korpus olusturma
  dataset    : InstructionDataset (sequence packing)
  train      : optimizer, LR schedule, egitim, checkpoint, resume
  generate   : generate_text

Calistirma:
  python -m src.py.normal            # tam pipeline (egit + uret)
  python -m src.py.normal.train      # CLI egitim (--resume vb.)
  python -m src.py.normal.generate   # CLI uretim (--interactive)
"""

from .config import GenerationConfig, TrainConfig, train_config_defaults
from .corpus import (
    build_assistant_only_corpus,
    build_chatml_dataset_file,
    build_instruction_corpus,
    build_long_clean_chat_corpus,
    load_chatml_dataset,
    load_jsonl_dialogues,
    make_synthetic_data,
)
from .dataset import InstructionDataset
from .generate import generate_text
from .model import (
    Block,
    CausalSelfAttention,
    DecoderOnlyLM,
    MLP,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary,
    default_swiglu_hidden,
    make_ffn,
    make_norm,
)
from .prompt import (
    ChatPromptFormatter,
    DEFAULT_SYSTEM_PROMPT,
    clean_text,
    format_chat_prompt,
    format_instruction_example,
)
from .tokenizer import BPETokenizer
from .train import (
    configure_optimizer,
    estimate_loss,
    load_checkpoint,
    load_tokenizer_from_checkpoint,
    lr_at,
    pick_device,
    save_checkpoint,
    seed_everything,
    train_model,
)

__all__ = [
    # config
    "TrainConfig",
    "GenerationConfig",
    "train_config_defaults",
    # tokenizer
    "BPETokenizer",
    # prompt
    "ChatPromptFormatter",
    "DEFAULT_SYSTEM_PROMPT",
    "clean_text",
    "format_chat_prompt",
    "format_instruction_example",
    # model
    "DecoderOnlyLM",
    "CausalSelfAttention",
    "SwiGLU",
    "MLP",
    "Block",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "make_norm",
    "make_ffn",
    "default_swiglu_hidden",
    # corpus
    "load_jsonl_dialogues",
    "build_instruction_corpus",
    "build_assistant_only_corpus",
    "build_long_clean_chat_corpus",
    "build_chatml_dataset_file",
    "load_chatml_dataset",
    "make_synthetic_data",
    # dataset
    "InstructionDataset",
    # train
    "seed_everything",
    "pick_device",
    "save_checkpoint",
    "load_checkpoint",
    "load_tokenizer_from_checkpoint",
    "configure_optimizer",
    "lr_at",
    "estimate_loss",
    "train_model",
    # generate
    "generate_text",
]
