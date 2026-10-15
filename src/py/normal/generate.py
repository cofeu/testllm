"""Compatibility wrapper around the PT generation path."""

from src.pt.decoder_model import ChatPromptFormatter, generate_text

__all__ = ["ChatPromptFormatter", "generate_text"]
