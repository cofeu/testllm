"""Metin uretimi: prompt'u token'lar, uretim yapar, ozel token'lari suzup cozer."""

import torch

from .config import GenerationConfig
from .model import EncoderDecoderLM
from .tokenizer import BPETokenizer


def generate_text(model: EncoderDecoderLM, tokenizer: BPETokenizer, prompt: str, generation: GenerationConfig | None = None, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generation = generation or GenerationConfig()
    prompt_text = str(prompt or "")
    if not prompt_text.strip():
        return ""
    if not prompt_text.endswith("\n"):
        prompt_text = prompt_text + "\n"

    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        return ""

    if len(prompt_ids) > model.cfg.block_size:
        prompt_ids = prompt_ids[-model.cfg.block_size:]

    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
        min_p=generation.min_p,
        repetition_penalty=generation.repetition_penalty,
        device=device,
        pad_id=tokenizer.PAD,
        bos_id=tokenizer.BOS,
        eos_id=tokenizer.EOS,
        unk_id=tokenizer.UNK,
        forbidden_ids={tokenizer.PAD, tokenizer.BOS, tokenizer.UNK, tokenizer.IM_START},
        stop_ids={tokenizer.IM_END},
        sampler=generation.sampler,
        typical_p=generation.typical_p,
        mirostat_tau=generation.mirostat_tau,
        mirostat_eta=generation.mirostat_eta,
    )
    # Encoder-decoder'da cikti yalnizca uretilen cevaptir (prompt encoder'a gider).
    output_ids = generated[0].tolist()
    visible_ids = [idx for idx in output_ids if idx >= tokenizer.OFFSET]
    if not visible_ids:
        return ""

    decoded = tokenizer.decode(visible_ids)
    return decoded.replace("\x00", "").strip()


def main(argv=None):
    """CLI:  python -m src.py.normal.generate --prompt "Merhaba" [--interactive]"""
    import argparse
    from pathlib import Path

    from .prompt import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPTS, format_chat_prompt, system_prompt_for
    from .train import load_checkpoint, load_checkpoint_metadata, pick_device

    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Modelden metin uret")
    ap.add_argument("--ckpt", default=str(root / "decoder_model_ckpt.pt"))
    ap.add_argument("--prompt", default="Merhaba, bugün nasıl yardımcı olabilirim?")
    ap.add_argument("--system", default=None, help="ozel sistem prompt'u (verilirse --mode ve checkpoint varsayilanini ezer)")
    ap.add_argument("--mode", choices=sorted(SYSTEM_PROMPTS), default=None,
                    help="hazir sistem-prompt modu: safe=censored, open=uncensored (varsayilan: checkpoint'te kayitli mod)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--min-p", type=float, default=0.02)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--sampler", choices=["top_p", "typical", "mirostat"], default="top_p")
    ap.add_argument("--device", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args(argv)

    device = pick_device() if args.device is None else torch.device(args.device)
    model, _ = load_checkpoint(args.ckpt, device=device)

    # Sistem-prompt onceligi: --system > --mode > checkpoint'te kayitli > default.
    ckpt_meta = {}
    if Path(args.ckpt).exists():
        try:
            ckpt_meta = load_checkpoint_metadata(str(args.ckpt))
        except Exception:
            ckpt_meta = {}
    if args.system is not None:
        system_prompt = args.system
    elif args.mode is not None:
        system_prompt = system_prompt_for(args.mode)
    else:
        system_prompt = ckpt_meta.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        if ckpt_meta.get("system_prompt"):
            print(f"(checkpoint sistem-modu: {ckpt_meta.get('system_mode') or 'ozel'})")
    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        sampler=args.sampler,
    )
    tokenizer = _tokenizer_from_checkpoint_or(args.ckpt, root)
    history = []

    def ask(text):
        prompt = format_chat_prompt(text, system_prompt=system_prompt, history=history)
        out = generate_text(model, tokenizer, prompt, generation=gen, device=device)
        print(f"asistan> {out}")
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": out})
        del history[:-8]
        return out

    if args.interactive:
        print("Cikmak icin 'q' yazin.")
        while True:
            try:
                text = input("siz> ").strip()
            except EOFError:
                break
            if text.lower() in {"q", "quit", "exit"}:
                break
            if text:
                ask(text)
    else:
        ask(args.prompt)


def _tokenizer_from_checkpoint_or(ckpt, root):
    """Once checkpoint'e gomulu tokenizer'i, yoksa data/tokenizer.json'i dener."""
    import json
    from pathlib import Path

    from .train import load_tokenizer_from_checkpoint

    if ckpt and Path(ckpt).exists():
        try:
            tok = load_tokenizer_from_checkpoint(str(ckpt))
        except Exception:
            tok = None
        if tok is not None:
            return tok

    for candidate in (Path(root) / "data" / "tokenizer.json",):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return BPETokenizer.from_dict(json.load(f), allow_legacy=True)
    raise FileNotFoundError("tokenizer bulunamadi; once egitim calistirin")


if __name__ == "__main__":
    main()
