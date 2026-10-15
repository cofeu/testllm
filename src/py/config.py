from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.pt.decoder_model import TrainConfig

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    if path is None:
        return ""
    return str(path)


@dataclass
class Config(TrainConfig):
    data_path: str = field(default_factory=lambda: resolve_path("data/data.jsonl"))
    ckpt_path: str = field(default_factory=lambda: resolve_path("checkpoints/torch_ckpt.pt"))
    val_frac: float = 0.1
    min_lr: float = 3e-5
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 250
    save_every: int = 250
    eval_iters: int = 50
    device: str = "auto"
    compile: bool = False
    amp: bool = True
    temperature: float = 0.55
    top_k: int = 18
    top_p: float = 0.85
    repetition_penalty: float = 1.08
    min_p: float = 0.02
    system_prompt: str = (
        "Sen Türkçe konuşan, kısa ve net cevap veren bir yardımcı asistanısın. "
        "Cevaplar kısa, anlaşılır, samimi ve işlevsel olsun."
    )
    tie_weights: bool = True
    use_rope: bool = False

    @property
    def n_heads(self):
        return self.n_head

    @n_heads.setter
    def n_heads(self, value):
        self.n_head = value

    @property
    def ffn_hidden(self):
        return self.ff_dim

    @ffn_hidden.setter
    def ffn_hidden(self, value):
        self.ff_dim = value

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, value):
        self.learning_rate = value

    @property
    def max_steps(self):
        return self.epochs

    @max_steps.setter
    def max_steps(self, value):
        self.epochs = value

    def to_dict(self):
        payload = asdict(self)
        payload["lr"] = self.learning_rate
        payload["max_steps"] = self.epochs
        payload["n_heads"] = self.n_head
        payload["ffn_hidden"] = self.ff_dim
        return payload

    @classmethod
    def from_dict(cls, d):
        payload = {}
        for k, v in (d or {}).items():
            key = k
            if k == "lr":
                key = "learning_rate"
            elif k == "max_steps":
                key = "epochs"
            elif k == "n_heads":
                key = "n_head"
            elif k == "ffn_hidden":
                key = "ff_dim"
            payload[key] = v
        return cls(**payload)


def apply_args(cfg, args, arch=True):
    alias_map = {
        "steps": "max_steps",
        "max_steps": "max_steps",
        "lr": "lr",
        "batch": "batch_size",
        "data": "data_path",
        "ckpt": "ckpt_path",
        "ckpt_path": "ckpt_path",
        "heads": "n_heads",
        "n_heads": "n_heads",
        "ffn": "ffn_hidden",
        "ffn_hidden": "ffn_hidden",
        "block": "block_size",
        "d_model": "d_model",
        "layers": "n_layers",
        "n_layers": "n_layers",
        "vocab_size": "vocab_size",
        "seed": "seed",
        "log_every": "log_every",
        "eval_every": "eval_every",
        "save_every": "save_every",
    }
    for arg_name, target in alias_map.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            setattr(cfg, target, getattr(args, arg_name))
    for field in getattr(cfg, "__dataclass_fields__", {}):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    for field in ("data_path", "ckpt_path"):
        if hasattr(args, field) and getattr(args, field) is not None:
            setattr(cfg, field, resolve_path(getattr(args, field)))
    if getattr(args, "data", None):
        cfg.data_path = resolve_path(args.data)
    ckpt_value = getattr(args, "ckpt_path", None) or getattr(args, "ckpt", None)
    if ckpt_value:
        cfg.ckpt_path = resolve_path(ckpt_value)
    if getattr(args, "batch", None) is not None:
        cfg.batch_size = args.batch
    return cfg


__all__ = ["Config", "resolve_path", "apply_args"]
