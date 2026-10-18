"""Egitim dongusu, optimizer, LR zamanlamasi ve checkpoint IO."""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig, train_config_defaults
from .corpus import make_synthetic_data
from .dataset import Seq2SeqDataset
from .model import EncoderDecoderLM
from .prompt import SYSTEM_PROMPTS
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


def _git_commit() -> str | None:
    """Depo commit hash'i (yoksa None)."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def save_checkpoint(
    model: torch.nn.Module,
    cfg: TrainConfig,
    path: str,
    optimizer=None,
    step: int | None = None,
    scaler=None,
    tokenizer: BPETokenizer | None = None,
    system_prompt: str | None = None,
    system_mode: str | None = None,
    meta: dict | None = None,
):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base = getattr(model, "_orig_mod", model)
    payload = {"state_dict": base.state_dict(), "config": dict(cfg.__dict__)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if tokenizer is not None:
        payload["tokenizer"] = tokenizer.to_dict()
        payload["tokenizer_summary"] = {
            "vocab_size": tokenizer.vocab_size,
            "version": tokenizer.VERSION,
            "pretokenization": tokenizer.PRETOKENIZATION,
            "n_merges": len(tokenizer.merges),
        }
    if step is not None:
        payload["step"] = int(step)
    # Egitimde kullanilan sistem-prompt'u checkpoint'e yaz: generate varsayilani
    # bu olsun, boylece --system-mode safe ile egitilen model dogru prompt'u gorur.
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if system_mode is not None:
        payload["system_mode"] = system_mode
    meta = dict(meta or {})
    meta.setdefault("git_commit", _git_commit())
    payload["meta"] = meta
    torch.save(payload, str(save_path))


def load_checkpoint(path: str, device: torch.device | None = None):
    device = device or pick_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    known = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k in known})
    model = EncoderDecoderLM(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def load_tokenizer_from_checkpoint(path: str) -> BPETokenizer | None:
    """Checkpoint'e gomulu tokenizer'i dondurur (yoksa None)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    data = payload.get("tokenizer")
    if not data:
        return None
    return BPETokenizer.from_dict(data, allow_legacy=True)


def load_checkpoint_metadata(path: str) -> dict:
    """Checkpoint'teki egitim meta verisini (sistem-prompt, mod, veri, commit) dondurur.

    Not: `torch.load` tum arsivi cozer (optimizer dahil). Generate gibi birden fazla
    bilgi gerektiren yerlerde bunun yerine `load_checkpoint_bundle` kullanin (tek okuma).
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "system_prompt": payload.get("system_prompt"),
        "system_mode": payload.get("system_mode"),
        "meta": payload.get("meta") or {},
        "tokenizer_summary": payload.get("tokenizer_summary") or {},
        "step": payload.get("step"),
    }


def load_checkpoint_bundle(path: str, device: torch.device | None = None):
    """Checkpoint'i BIR kez okur; (model, cfg, metadata, tokenizer) dondurur.

    `generate` bu sayede checkpoint'i 3 kez okumaz.
    """
    device = device or pick_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    known = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k in known})
    model = EncoderDecoderLM(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {
        "system_prompt": payload.get("system_prompt"),
        "system_mode": payload.get("system_mode"),
        "meta": payload.get("meta") or {},
        "tokenizer_summary": payload.get("tokenizer_summary") or {},
        "step": payload.get("step"),
    }
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"], allow_legacy=True) if payload.get("tokenizer") else None
    return model, cfg, metadata, tokenizer


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
    """Val kumesini bir kez dolasip token-agirlikli ortalama kaybi dondurur.

    Her batch tek bir rastgele kovadan gelmez; tum val ornekleri tam olarak bir
    kez gorulur (deterministik). Ortalama, dolgu disi hedef token sayisina gore
    agirliklandirilir; boylece val_loss ve perplexity tekrarlanabilir olur.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt_in, tgt_out, in data.eval_batches("val", cfg.batch_size, device):
        loss = forward_loss(model, src, tgt_in, tgt_out, cfg, device)
        n = int((tgt_out != Seq2SeqDataset.IGNORE).sum())
        if n == 0:
            continue
        total_loss += loss.item() * n
        total_tokens += n
    if was_training:
        model.train()
    return total_loss / max(1, total_tokens)


def forward_loss(model, src, tgt_in, tgt_out, cfg, device):
    """AMP'li tek ileri gecis + kayip. Pad maskeleri PAD id'sinden turetilir."""
    pad = BPETokenizer.PAD
    src_pad = src == pad
    tgt_pad = tgt_in == pad
    if cfg.amp and device.type == "cuda":
        with torch.amp.autocast("cuda"):
            loss = model(src, tgt_in, tgt_out, src_pad=src_pad, tgt_pad=tgt_pad)[1]
    else:
        loss = model(src, tgt_in, tgt_out, src_pad=src_pad, tgt_pad=tgt_pad)[1]
    return loss


def train_model(cfg: TrainConfig, device: torch.device | None = None, ckpt_path: str | None = None, tokenizer: BPETokenizer | None = None, texts=None, system_prompt: str | None = None, system_mode: str | None = None, meta: dict | None = None, keep_best: bool = False, early_stop: int = 0):
    device = device or pick_device()
    # Veri kurulumu deterministik olsun diye taban tohum; resume'da batch akisini
    # farklilastirmak icin start_step kadar kaydirilir (asagida).
    seed_everything(cfg.seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = EncoderDecoderLM(cfg).to(device)
    optim = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if (cfg.amp and device.type == "cuda") else None

    ckpt_path = ckpt_path or os.path.join(os.path.dirname(__file__), "normal_ckpt.pt")
    start_step = 1
    resumed_best = None
    resumed_bad = 0
    if cfg.resume and os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(payload["state_dict"])
        if "optimizer" in payload:
            optim.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        if tokenizer is None and payload.get("tokenizer"):
            tokenizer = BPETokenizer.from_dict(payload["tokenizer"], allow_legacy=True)
        prev_meta = payload.get("meta") or {}
        resumed_best = prev_meta.get("best_val")
        resumed_bad = int(prev_meta.get("bad_evals", 0))
        start_step = int(payload.get("step", 0)) + 1
        print(f"resume: {ckpt_path} (step {start_step - 1}'den devam)", flush=True)

    if start_step > cfg.epochs:
        print(f"egitim zaten tamam (step {start_step - 1} >= epochs {cfg.epochs}); atlaniyor.", flush=True)
        model.eval()
        return model, cfg

    if start_step > 1:
        # Ayni batch akisini tekrar oynatmayi onle: tohumu adima gore kaydir.
        seed_everything(cfg.seed + start_step)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    data = Seq2SeqDataset(tokenizer, cfg, texts) if (tokenizer is not None and texts) else None
    if data is not None:
        avg_src = data.total_src_tokens / max(1, data.n_examples)
        avg_tgt = data.total_tgt_tokens / max(1, data.n_examples)
        print(f"veri: {data.n_examples} ornek, train {data.n_train}, val {data.n_val}, "
              f"vocab={cfg.vocab_size}", flush=True)
        print(f"     ortalama src={avg_src:.1f} tgt={avg_tgt:.1f} token (block={cfg.block_size})", flush=True)
    base_model = getattr(model, "_orig_mod", model)
    n_params = sum(p.numel() for p in model.parameters())
    rope_state = "RoPE" if base_model.uses_rope else "learned-pos"
    if cfg.use_rope and not base_model.uses_rope:
        rope_state += " (use_rope istendi ama head_dim tek -> ogrenilmis pozisyon)"
    ff_dim = base_model.encoder[0].mlp.hidden if base_model.encoder else cfg.ff_dim
    print(f"model: {n_params} parametre, d_model={cfg.d_model}, "
          f"enc/dec layers={cfg.n_layers}/{cfg.n_layers}, ff_dim={ff_dim}, pos={rope_state}", flush=True)

    accum = max(1, cfg.grad_accum)

    # En iyi val / erken durdurma durumu resume'da korunur (aksi halde ilk eval
    # daha kotu olsa bile .best ezilir ve erken durdurma sayaci sifirlanir).
    best_val = float(resumed_best) if resumed_best is not None else float("inf")
    best_path = ckpt_path + ".best" if keep_best else None
    if keep_best and best_path and os.path.exists(best_path):
        best_meta = (torch.load(best_path, map_location="cpu", weights_only=True).get("meta") or {})
        if best_meta.get("best_val") is not None:
            best_val = min(best_val, float(best_meta["best_val"]))
    bad_evals = resumed_bad
    run_meta = dict(meta or {})

    def _save(target, step):
        run_meta["best_val"] = None if best_val == float("inf") else best_val
        run_meta["bad_evals"] = bad_evals
        save_checkpoint(model, cfg, target, optimizer=optim, step=step, scaler=scaler,
                        tokenizer=tokenizer, system_prompt=system_prompt,
                        system_mode=system_mode, meta=run_meta)

    last_step = start_step - 1
    for step in range(start_step, cfg.epochs + 1):
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group["lr"] = lr
        model.train()
        optim.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(accum):
            if data is not None:
                src, tgt_in, tgt_out = data.get_batch("train", cfg.batch_size, device)
            else:
                src, tgt_in, tgt_out = make_synthetic_data(cfg)
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            loss = forward_loss(model, src, tgt_in, tgt_out, cfg, device)
            if scaler is not None:
                scaler.scale(loss / accum).backward()
            else:
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

        last_step = step
        if step % cfg.log_every == 0 or step == start_step:
            print(f"step={step} loss={running / accum:.4f} lr={lr:.2e} device={device}", flush=True)

        if data is not None and (step % cfg.eval_every == 0 or step == cfg.epochs):
            val = estimate_loss(model, data, cfg, device)
            if cfg.label_smoothing == 0.0 and cfg.z_loss == 0.0:
                # exp(loss) yalnizca saf cross-entropy'de perplexity'dir.
                print(f"eval step={step} val_loss={val:.4f} ppl={math.exp(val):.2f}", flush=True)
            else:
                print(f"eval step={step} val_loss={val:.4f}", flush=True)
            if val < best_val - 1e-6:
                best_val = val
                bad_evals = 0
                if best_path is not None:
                    _save(best_path, step)
                    print(f"  yeni en iyi val_loss={best_val:.4f} -> {best_path}", flush=True)
            else:
                bad_evals += 1
                if early_stop and bad_evals >= early_stop:
                    print(f"erken durdurma: {bad_evals} degerlendirmedir iyilesme yok.", flush=True)
                    break

        if step % cfg.save_every == 0 or step == cfg.epochs:
            _save(ckpt_path, step)

    _save(ckpt_path, last_step)
    print(f"checkpoint saved: {ckpt_path}")
    return model, cfg


def add_train_args(ap) -> None:
    """Egitim CLI argumanlari; varsayilanlar TrainConfig'ten (tek kaynak)."""
    d = train_config_defaults()
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data"

    ap.add_argument("--data", default=str(data_dir / "data_large.jsonl"))
    ap.add_argument(
        "--extra-data",
        action="append",
        default=None,
        metavar="PATH",
        help="ek veri dosyasi; birden fazla kez verilebilir (varsayilan: data/data.jsonl)",
    )
    ap.add_argument(
        "--system-mode",
        choices=sorted(SYSTEM_PROMPTS),
        default=None,
        help="sistem-prompt modu (varsayilan: default; --resume'da checkpoint'ten devralinir): "
             "safe=censored (reddeden), open=uncensored (dolaysiz)",
    )
    ap.add_argument("--ckpt", default=str(root / "decoder_model_ckpt.pt"))
    ap.add_argument("--tokenizer", default=str(data_dir / "tokenizer.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-examples", type=int, default=100000,
                    help="egitim ornegi ust siniri (eleme sonrasi uygulanir)")

    ap.add_argument("--steps", type=int, default=8000, help="toplam optimizasyon adimi (epochs)")
    ap.add_argument("--batch-size", type=int, default=d["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=d["grad_accum"])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=d["n_layers"])
    ap.add_argument("--n-head", type=int, default=d["n_head"])
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=d["ff_dim"], help="0 -> otomatik (Llama kurali / 4*d_model)")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=d["dropout"])
    ap.add_argument("--lr", type=float, default=d["learning_rate"])
    ap.add_argument("--min-lr", type=float, default=d["min_lr"])
    ap.add_argument("--warmup", type=int, default=d["warmup"])
    ap.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    ap.add_argument("--grad-clip", type=float, default=d["grad_clip"])
    ap.add_argument("--betas", type=float, nargs=2, default=list(d["betas"]))
    ap.add_argument("--eps", type=float, default=d["eps"])
    ap.add_argument("--label-smoothing", type=float, default=d["label_smoothing"])
    ap.add_argument("--norm", choices=["rmsnorm", "layernorm"], default=d["norm_type"])
    ap.add_argument("--activation", choices=["swiglu", "gelu"], default=d["activation"])
    ap.add_argument("--rope-theta", type=float, default=d["rope_theta"])
    ap.add_argument("--rope-scaling", choices=["none", "linear"], default=d["rope_scaling"])
    ap.add_argument("--rope-scaling-factor", type=float, default=d["rope_scaling_factor"])
    ap.add_argument("--z-loss", type=float, default=d["z_loss"])
    ap.add_argument("--qk-norm", action="store_true", default=d["qk_norm"])
    ap.add_argument("--bias", action="store_true", default=d["bias"])
    ap.add_argument("--no-rope", action="store_true")
    ap.add_argument("--no-tie", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-fused", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true", default=d["grad_checkpointing"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--keep-best", action="store_true", help="en iyi val kaybini ayri dosyaya (.best) kaydet")
    ap.add_argument("--early-stop", type=int, default=0, help="iyilesme olmadan bu kadar eval sonra dur (0=kapali)")
    ap.add_argument("--seed", type=int, default=d["seed"])

    ap.add_argument("--val-frac", type=float, default=d["val_frac"])
    ap.add_argument("--eval-every", type=int, default=d["eval_every"])
    ap.add_argument("--save-every", type=int, default=d["save_every"])
    ap.add_argument("--log-every", type=int, default=d["log_every"])


def config_from_args(args, vocab_size: int) -> TrainConfig:
    return TrainConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        ff_dim=args.ff_dim,
        block_size=args.block_size,
        dropout=args.dropout,
        tie_weights=not args.no_tie,
        bias=args.bias,
        use_rope=not args.no_rope,
        rope_theta=args.rope_theta,
        rope_scaling=args.rope_scaling,
        rope_scaling_factor=args.rope_scaling_factor,
        norm_type=args.norm,
        activation=args.activation,
        qk_norm=args.qk_norm,
        z_loss=args.z_loss,
        grad_checkpointing=args.grad_checkpointing,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        betas=tuple(args.betas),
        eps=args.eps,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.steps,
        fused_adamw=not args.no_fused,
        resume=args.resume,
        val_frac=args.val_frac,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        amp=not args.no_amp,
        compile=args.compile,
        seed=args.seed,
    )


def main(argv=None):
    """CLI:  python -m src.py.normal.train --steps 2000 --resume ..."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Llama+GPT tarzi model egitimi")
    add_train_args(ap)
    args = ap.parse_args(argv)

    from .corpus import build_assistant_only_corpus, load_jsonl_records
    from .prompt import system_prompt_for

    dialogues = load_jsonl_records(args.data)
    extra_paths = args.extra_data if args.extra_data is not None else [str(Path(__file__).resolve().parents[3] / "data" / "data.jsonl")]
    used_paths = [args.data]
    for extra_path in extra_paths:
        if extra_path and os.path.exists(extra_path):
            dialogues += load_jsonl_records(extra_path)
            used_paths.append(extra_path)
    # Resume'da kullanici mod belirtmediyse checkpoint'teki modu devral (aksi halde
    # safe ile egitilen model resume'da sessizce 'default'a duser).
    system_mode = args.system_mode
    if system_mode is None and args.resume and os.path.exists(args.ckpt):
        system_mode = load_checkpoint_metadata(args.ckpt).get("system_mode")
    if system_mode is None:
        system_mode = "default"
    system_prompt = system_prompt_for(system_mode)
    corpus = build_assistant_only_corpus(
        dialogues, system_prompt=system_prompt, max_examples=args.max_examples
    )
    print(f"korpus: {len(corpus)} ornek", flush=True)

    texts = [c["user"] for c in corpus] + [c["assistant"] for c in corpus]
    raw = None
    if os.path.exists(args.tokenizer):
        with open(args.tokenizer, "r", encoding="utf-8") as f:
            raw = json.load(f)
    if raw is not None and raw.get("pretokenization") == BPETokenizer.PRETOKENIZATION:
        tokenizer = BPETokenizer.from_dict(raw)
    else:
        # Eski (kimliksiz/uyumsuz) dosyayi damgalamak yerine yeniden egit: merge'ler
        # yeni semayla ogrenilmemis olabilir, aksi halde surum kontrolu anlamsizlasir.
        if raw is not None:
            print("tokenizer uyumsuz sema (pretokenization); yeniden egitiliyor...", flush=True)
        vocab = int(raw.get("vocab_size", 2048)) if raw else 2048
        tokenizer = BPETokenizer.from_corpus(texts, vocab_size=vocab)
        with open(args.tokenizer, "w", encoding="utf-8") as f:
            json.dump(tokenizer.to_dict(), f, ensure_ascii=False)

    cfg = config_from_args(args, tokenizer.vocab_size)
    device = pick_device() if args.device is None else torch.device(args.device)
    meta = {
        "data_files": used_paths,
        "system_mode": system_mode,
        "n_examples": len(corpus),
        "vocab_size": tokenizer.vocab_size,
    }
    train_model(
        cfg, device=device, ckpt_path=args.ckpt, tokenizer=tokenizer, texts=corpus,
        system_prompt=system_prompt, system_mode=system_mode, meta=meta,
        keep_best=args.keep_best, early_stop=args.early_stop,
    )


if __name__ == "__main__":
    main()
