"""Compatibility layer: normal path intentionally mirrors the PT prototype.

This package re-exports the canonical classes from src.pt.decoder_model so both
code paths share the same tokenizer, prompt formatter, config, and model logic.
"""

from src.pt.decoder_model import BPETokenizer, ChatPromptFormatter, DecoderOnlyLM, GenerationConfig, TrainConfig

Config = TrainConfig
GPT = DecoderOnlyLM

__all__ = [
    "BPETokenizer",
    "ChatPromptFormatter",
    "Config",
    "DecoderOnlyLM",
    "GPT",
    "GenerationConfig",
    "TrainConfig",
]
