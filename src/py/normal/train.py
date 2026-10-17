"""Egitim dongusu, optimizer, LR zamanlamasi ve checkpoint IO."""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .corpus import make_synthetic_data
from .dataset import InstructionDataset
from .model import DecoderOnlyLM
from .tokenizer import BPETokenizer


def seed_everything(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, cfg: TrainConfig, path: str, optimizer=None, step: int | None = None):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base = getattr(model, "_orig_mod", model)
    payload = {"state_dict": base.state_dict(), "config": dict(cfg.__dict__)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = int(step)
    torch.save(payload, str(save_path))


def load_checkpoint(path: str, device: torch.device | None = None):
    device = device or pick_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    known = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k in known})
    model = DecoderOnlyLM(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def configure_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"betas": cfg.betas, "eps": cfg.eps}
    if cfg.fused_adamw and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, lr=cfg.learning_rate, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, **kwargs)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup > 0 and step < cfg.warmup:
        return cfg.learning_rate * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.epochs - cfg.warmup)
    prog = min(1.0, max(0.0, prog))
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, data, cfg, device):
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = data.get_batch("val", cfg.batch_size, device)
        if cfg.amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train_model(cfg: TrainConfig, device: torch.device | None = None, ckpt_path: str | None = None, tokenizer: BPETokenizer | None = None, texts=None):
    device = device or pick_device()
    seed_everything(cfg.seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = DecoderOnlyLM(cfg).to(device)
    optim = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if (cfg.amp and device.type == "cuda") else None

    ckpt_path = ckpt_path or os.path.join(os.path.dirname(__file__), "normal_ckpt.pt")
    start_step = 1
    if cfg.resume and os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        if "optimizer" in payload:
            optim.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0)) + 1
        print(f"resume: {ckpt_path} (step {start_step - 1}'den devam)", flush=True)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    data = InstructionDataset(tokenizer, cfg, texts) if (tokenizer is not None and texts) else None
    if data is not None:
        print(f"veri: {data.n_examples} ornek, train {data.n_train} token, val {data.n_val} token, "
              f"vocab={cfg.vocab_size}", flush=True)
        print(f"     learnable {data.learnable_tokens}/{data.total_tokens} token, "
              f"packing doluluk ~{data.packing_ratio:.1%}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} parametre, d_model={cfg.d_model}, layers={cfg.n_layers}, "
          f"rope={cfg.use_rope}", flush=True)

    accum = max(1, cfg.grad_accum)

    for step in range(start_step, cfg.epochs + 1):
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group["lr"] = lr
        model.train()
        optim.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(accum):
            if data is not None:
                x, y = data.get_batch("train", cfg.batch_size, device)
            else:
                x, y = make_synthetic_data(cfg)
                x, y = x.to(device), y.to(device)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss / accum).backward()
            else:
                _, loss = model(x, y)
                (loss / accum).backward()
            running += loss.item()

        if cfg.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        if step % cfg.log_every == 0 or step == start_step:
            print(f"step={step} loss={running / accum:.4f} lr={lr:.2e} device={device}", flush=True)

        if data is not None and (step % cfg.eval_every == 0 or step == cfg.epochs):
            val = estimate_loss(model, data, cfg, device)
            print(f"eval step={step} val_loss={val:.4f}", flush=True)

        if step % cfg.save_every == 0 or step == cfg.epochs:
            save_checkpoint(model, cfg, ckpt_path, optimizer=optim, step=step)

    save_checkpoint(model, cfg, ckpt_path, optimizer=optim, step=cfg.epochs)
    print(f"checkpoint saved: {ckpt_path}")
    return model, cfg


def main(argv=None):
    """CLI:  python -m src.py.normal.train --steps 2000 --resume ..."""
    import argparse
    import json

    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data"

    ap = argparse.ArgumentParser(description="Llama+GPT tarzi model egitimi")
    ap.add_argument("--data", default=str(data_dir / "data_large.jsonl"))
    ap.add_argument("--extra-data", default=str(data_dir / "data.jsonl"))
    ap.add_argument("--ckpt", default=str(root / "decoder_model_ckpt.pt"))
    ap.add_argument("--tokenizer", default=str(data_dir / "tokenizer.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--norm", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    ap.add_argument("--activation", choices=["swiglu", "gelu"], default="swiglu")
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--no-rope", action="store_true")
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    from .corpus import build_assistant_only_corpus, load_jsonl_dialogues

    dialogues = load_jsonl_dialogues(args.data)
    if args.extra_data and os.path.exists(args.extra_data):
        dialogues += load_jsonl_dialogues(args.extra_data)
    corpus = build_assistant_only_corpus(dialogues, max_examples=100000)
    print(f"korpus: {len(corpus)} ornek", flush=True)

    texts = [c["user"] for c in corpus] + [c["assistant"] for c in corpus]
    if os.path.exists(args.tokenizer):
        with open(args.tokenizer, "r", encoding="utf-8") as f:
            tokenizer = BPETokenizer.from_dict(json.load(f))
    else:
        tokenizer = BPETokenizer.from_corpus(texts, vocab_size=2048)
        with open(args.tokenizer, "w", encoding="utf-8") as f:
            json.dump(tokenizer.to_dict(), f, ensure_ascii=False)

    cfg = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        ff_dim=args.ff_dim,
        block_size=args.block_size,
        dropout=args.dropout,
        use_rope=not args.no_rope,
        norm_type=args.norm,
        activation=args.activation,
        qk_norm=args.qk_norm,
        grad_checkpointing=args.grad_checkpointing,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.steps,
        pack_sequences=not args.no_pack,
        resume=args.resume,
        compile=args.compile,
        seed=args.seed,
    )

    device = pick_device() if args.device is None else torch.device(args.device)
    train_model(cfg, device=device, ckpt_path=args.ckpt, tokenizer=tokenizer, texts=corpus)


if __name__ == "__main__":
    main()
