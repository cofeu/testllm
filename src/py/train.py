"""Egitim dongusu: veri -> forward/backward -> AdamW -> checkpoint.

Kullanim (proje kokunden):
    python3 src/py/train.py --steps 300
"""

import argparse
import os
import random
import time

from checkpoint import load_checkpoint, save_checkpoint
from config import Config, resolve_path
from dataset import build_dataset, load_dialogues, split_train_val
from model import TransformerLM
from optim import AdamW, clip_grad_norm, lr_at, scale_inplace_grads
from tokenizer import BPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Saf Python LLM egitimi")
    p.add_argument("--data", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--ffn", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    return p.parse_args()


def apply_args(cfg, args, arch=True):
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.save_every is not None:
        cfg.save_every = args.save_every

    if getattr(args, "data", None):
        cfg.data_path = resolve_path(args.data)
    if getattr(args, "ckpt", None):
        cfg.ckpt_path = resolve_path(args.ckpt)
    if getattr(args, "batch", None) is not None:
        cfg.batch_size = args.batch
    if getattr(args, "lr", None) is not None:
        cfg.lr = args.lr
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed

    if arch:
        if getattr(args, "block", None) is not None:
            cfg.block_size = args.block
        if getattr(args, "d_model", None) is not None:
            cfg.d_model = args.d_model
        if getattr(args, "layers", None) is not None:
            cfg.n_layers = args.layers
        if getattr(args, "heads", None) is not None:
            cfg.n_heads = args.heads
        if getattr(args, "ffn", None) is not None:
            cfg.ffn_hidden = args.ffn
    return cfg


def evaluate(model, windows, cfg, max_items=32):
    if not windows:
        return float("nan")
    total = 0.0
    count = 0
    for x, y in windows[:max_items]:
        loss, _ = model.forward(x, y)
        total += loss
        count += 1
    return total / max(1, count)


def main():
    args = parse_args()
    cfg = Config()
    if args.ckpt:
        cfg.ckpt_path = resolve_path(args.ckpt)

    ckpt = None
    if args.resume:
        if not os.path.exists(cfg.ckpt_path):
            raise SystemExit(f"resume icin checkpoint yok: {cfg.ckpt_path}")
        ckpt = load_checkpoint(cfg.ckpt_path)
        cfg = Config.from_dict(ckpt["config"])
        cfg = apply_args(cfg, args, arch=False)
        start_step = ckpt["step"]
        print(f"resume: {cfg.ckpt_path} (step={start_step})")
    else:
        cfg = apply_args(cfg, args, arch=True)
        start_step = 0

    rng = random.Random(cfg.seed)
    dialogues = load_dialogues(cfg.data_path)
    tok = BPETokenizer.from_corpus(dialogues, cfg.vocab_size)
    assert cfg.vocab_size == tok.vocab_size, "vocab_size tokenizer ile ayni olmali"
    windows = build_dataset(tok, dialogues, cfg.block_size)
    train_w, val_w = split_train_val(windows, cfg.val_frac, cfg.seed)
    print(
        f"veri: {len(dialogues)} diyalog, {len(windows)} pencere "
        f"(train {len(train_w)}, val {len(val_w)}) | blok={cfg.block_size}"
    )

    model = TransformerLM(cfg, random.Random(cfg.seed))
    if ckpt is not None:
        model.load_state_dict(ckpt["state"])

    params = list(model.parameters())
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if ckpt is not None:
        extra = ckpt.get("extra") or {}
        if "opt_m" in extra:
            opt.m.update(extra["opt_m"])
            opt.v.update(extra["opt_v"])
            print("adam durumu yuklendi")
    print(
        f"model: {model.num_params():,} parametre | "
        f"d={cfg.d_model} katman={cfg.n_heads}x{cfg.n_layers} ffn={cfg.ffn_hidden}"
    )

    order = list(range(len(train_w)))
    step = start_step
    t0 = time.time()
    loss_acc = 0.0
    loss_n = 0

    while step < cfg.max_steps:
        rng.shuffle(order)
        for bi in range(0, len(order), cfg.batch_size):
            if step >= cfg.max_steps:
                break
            batch = [train_w[i] for i in order[bi : bi + cfg.batch_size]]
            lr = lr_at(step, cfg)
            opt.lr = lr
            model.zero_grad()
            for x, y in batch:
                loss, _ = model.forward(x, y)
                model.backward()
                loss_acc += loss
                loss_n += 1
            scale_inplace_grads(params, 1.0 / len(batch))
            gn = clip_grad_norm(params, cfg.grad_clip)
            opt.step(step + 1)
            opt.zero_grad()
            step += 1

            if step % cfg.log_every == 0:
                avg = loss_acc / max(1, loss_n)
                dt = time.time() - t0
                print(
                    f"adim {step:4d}/{cfg.max_steps} | loss {avg:.4f} | "
                    f"lr {lr:.2e} | |g| {gn:.3f} | {dt:.1f}s"
                )
                loss_acc = 0.0
                loss_n = 0
                t0 = time.time()

            if step % cfg.eval_every == 0:
                vl = evaluate(model, val_w, cfg)
                print(f"         val loss {vl:.4f}")

            if step % cfg.save_every == 0 or step >= cfg.max_steps:
                vl = evaluate(model, val_w, cfg)
                save_checkpoint(
                    cfg.ckpt_path, model, cfg, step,
                    {"val_loss": vl, "opt_m": opt.m, "opt_v": opt.v},
                )
                print(f"         checkpoint -> {cfg.ckpt_path}")

    vl = evaluate(model, val_w, cfg)
    save_checkpoint(
        cfg.ckpt_path, model, cfg, step,
        {"val_loss": vl, "opt_m": opt.m, "opt_v": opt.v},
    )
    print(f"tamamlandi: {step} adim, son val loss {vl:.4f}, ckpt {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
