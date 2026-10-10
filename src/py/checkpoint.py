"""Model agirliklarini pickle ile kaydet/yukle (standart kutuphane)."""

import os
import pickle


def save_checkpoint(path, model, cfg, step, extra=None, tokenizer=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": cfg.to_dict(),
        "step": step,
        "state": model.state_dict(),
        "extra": extra or {},
        "tokenizer": tokenizer.to_dict() if tokenizer is not None else None,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)
