"""Egitilmis modelden token token uretim (temperature + top-k orneklemesi).

Kullanim (proje kokunden):
    python3 src/py/generate.py --prompt "K: Merhaba\nA:"
    python3 src/py/generate.py --interactive
"""

import argparse
import math
import random

from checkpoint import load_checkpoint
from config import Config, resolve_path
from dataset import load_dialogues
from model import TransformerLM
from tokenizer import BPETokenizer


def format_prompt(user_text, assistant_prefix="A:"):
    text = (user_text or "").strip()
    if not text:
        text = "Merhaba"
    return f"K: {text}\n{assistant_prefix}: "


def sample_next(row, temperature, top_k, rng, repetition_penalty=1.0, recent=None):
    if not row:
        return 0
    if temperature <= 0:
        return max(range(len(row)), key=lambda i: row[i])

    if recent is None:
        recent = []
    scaled = [v / temperature for v in row]
    if top_k and top_k < len(scaled):
        kth = sorted(scaled, reverse=True)[top_k - 1]
        scaled = [v if v >= kth else -1e30 for v in scaled]
    if repetition_penalty is not None and repetition_penalty > 1.0:
        for i in recent:
            if 0 <= i < len(scaled):
                scaled[i] /= repetition_penalty
    m = max(scaled)
    e = [math.exp(v - m) for v in scaled]
    s = sum(e)
    if s <= 0:
        return max(range(len(row)), key=lambda i: row[i])
    r = (rng.random() if rng is not None else 0.5) * s
    acc = 0.0
    for i, vi in enumerate(e):
        acc += vi
        if r <= acc:
            return i
    return len(e) - 1


def generate(model, tok, prompt, cfg, max_new=80, temperature=0.7, top_k=40, rng=None, repetition_penalty=1.1):
    rng = rng or random.Random(2024)
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        clean_prompt = format_prompt("Merhaba")
    if not clean_prompt.startswith("K:"):
        clean_prompt = format_prompt(clean_prompt)
    ids = [tok.BOS] + [i for i in tok.encode(clean_prompt) if tok.OFFSET <= i < tok.vocab_size]
    out_ids = []
    recent = []
    for _ in range(max_new):
        ctx = ids[-cfg.block_size:]
        _, logits = model.forward(ctx)
        nxt = sample_next(logits[-1], temperature, top_k, rng, repetition_penalty, recent)
        if nxt == tok.EOS:
            break
        if nxt < tok.OFFSET or nxt >= tok.vocab_size:
            continue
        ids.append(nxt)
        out_ids.append(nxt)
        recent.append(nxt)
        if len(recent) > 12:
            recent.pop(0)
    return tok.decode(ids), tok.decode(out_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt.pkl")
    p.add_argument("--prompt", default="K: Merhaba\nA:")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()
    args.prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")

    args.ckpt = resolve_path(args.ckpt)
    ckpt = load_checkpoint(args.ckpt)
    cfg = Config.from_dict(ckpt["config"])
    tokenizer_payload = ckpt.get("tokenizer")
    if tokenizer_payload is not None:
        tok = BPETokenizer.from_dict(tokenizer_payload)
    else:
        dialogues = load_dialogues(cfg.data_path)
        tok = BPETokenizer.from_corpus(dialogues, cfg.vocab_size)
    model = TransformerLM(cfg)
    model.eval()
    model.load_state_dict(ckpt["state"])
    print(f"model yuklendi: {ckpt['step']} adim, {model.num_params():,} parametre\n")

    if args.interactive:
        print("cikis icin 'quit' yazin")
        while True:
            try:
                user = input("K: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user.lower() in {"quit", "exit", "cik"}:
                break
            _, reply = generate(
                model, tok, "K: " + user + "\nA:", cfg,
                args.max_new, args.temperature, args.top_k,
                random.Random(args.seed),
                args.repetition_penalty,
            )
            print("A:", reply.strip())
    else:
        full, reply = generate(
            model, tok, args.prompt, cfg,
            args.max_new, args.temperature, args.top_k,
            random.Random(args.seed),
            args.repetition_penalty,
        )
        print(full)


if __name__ == "__main__":
    main()
