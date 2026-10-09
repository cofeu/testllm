from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


@dataclass
class Config:
    vocab_size: int = 1024
    block_size: int = 48
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    ffn_hidden: int = 256

    lr: float = 5e-4
    min_lr: float = 1e-4
    warmup: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.0
    tie_weights: bool = False
    use_rope: bool = False

    batch_size: int = 4
    max_steps: int = 300
    eval_every: int = 25
    log_every: int = 10
    save_every: int = 50

    seed: int = 1337
    data_path: str = resolve_path("data/data.jsonl")
    ckpt_path: str = resolve_path("checkpoints/ckpt.pkl")
    val_frac: float = 0.1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        item = {}
        for k, v in d.items():
            if k not in known:
                continue
            if k in {"data_path", "ckpt_path"} and isinstance(v, str):
                item[k] = resolve_path(v)
            else:
                item[k] = v
        return cls(**item)
