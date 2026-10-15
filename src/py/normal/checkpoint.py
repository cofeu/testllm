"""Compatibility wrapper: checkpoint IO is implemented by the PT path."""

from src.pt.decoder_model import load_checkpoint, save_checkpoint

__all__ = ["save_checkpoint", "load_checkpoint"]
