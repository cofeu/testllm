"""Compatibility wrapper: normal tokenizer is the canonical PT tokenizer."""

from src.pt.decoder_model import BPETokenizer

ByteTokenizer = BPETokenizer

__all__ = ["BPETokenizer", "ByteTokenizer"]
