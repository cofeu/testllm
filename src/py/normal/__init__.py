"""Kanonik PyTorch implementasyonu (Llama tarzi).

Modul yapisi:
  config     : TrainConfig, GenerationConfig
  tokenizer  : BPETokenizer
  prompt     : ChatPromptFormatter, clean_text
  model      : DecoderOnlyLM (RMSNorm, RoPE, GQA, SwiGLU, KV cache)
  corpus     : veri okuma/temizleme/korpus olusturma
  dataset    : InstructionDataset (sequence packing)
  train      : optimizer, LR schedule, egitim, checkpoint
  generate   : generate_text

Calistirma:
  python -m src.py.normal
"""

from .config import GenerationConfig, TrainConfig
from .corpus import (
    build_assistant_only_corpus,
    build_chatml_dataset_file,
    build_instruction_corpus,
    build_long_clean_chat_corpus,
    build_strict_instruction_corpus,
    build_text_sequences,
    load_chatml_dataset,
    load_jsonl_dialogues,
    make_synthetic_data,
    make_text_data,
)
from .dataset import InstructionDataset
from .generate import generate_text
from .model import (
    Block,
    CausalSelfAttention,
    DecoderOnlyLM,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary,
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
    "Block",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    # corpus
    "load_jsonl_dialogues",
    "build_instruction_corpus",
    "build_assistant_only_corpus",
    "build_long_clean_chat_corpus",
    "build_strict_instruction_corpus",
    "build_chatml_dataset_file",
    "load_chatml_dataset",
    "build_text_sequences",
    "make_text_data",
    "make_synthetic_data",
    # dataset
    "InstructionDataset",
    # train
    "seed_everything",
    "pick_device",
    "save_checkpoint",
    "load_checkpoint",
    "configure_optimizer",
    "lr_at",
    "estimate_loss",
    "train_model",
    # generate
    "generate_text",
]
