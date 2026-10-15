"""Compatibility layer: the normal model is a thin alias to the PT model."""

from src.pt.decoder_model import DecoderOnlyLM

GPT = DecoderOnlyLM

__all__ = ["GPT", "DecoderOnlyLM"]
