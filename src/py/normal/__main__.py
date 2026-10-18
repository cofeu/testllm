"""Kanonik giris noktasi:  python -m src.py.normal

Veri yukle -> temizle -> tokenizer (onbellek) -> egit -> uret -> checkpoint dogrula.
"""

import json
import os
from pathlib import Path

from .config import GenerationConfig, TrainConfig
from .corpus import build_assistant_only_corpus, load_jsonl_records
from .generate import generate_text
from .model import EncoderDecoderLM
from .prompt import DEFAULT_SYSTEM_PROMPT, format_chat_prompt
from .tokenizer import BPETokenizer
from .train import load_checkpoint, pick_device, train_model

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
CKPT_PATH = ROOT / "decoder_model_ckpt.pt"


def build_tokenizer(corpus, tok_path):
    texts = [item["user"] for item in corpus if isinstance(item, dict)] + [
        item["assistant"] for item in corpus if isinstance(item, dict)
    ]
    raw = None
    if os.path.exists(tok_path):
        with open(tok_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    if raw is not None and raw.get("pretokenization") == BPETokenizer.PRETOKENIZATION:
        tokenizer = BPETokenizer.from_dict(raw)
        print(f"tokenizer yuklendi: {tok_path} (vocab={tokenizer.vocab_size})", flush=True)
    else:
        # Eski/uyumsuz tokenizer'i damgalamak yerine yeniden egit (merge semasi farkli olabilir).
        if raw is not None:
            print("tokenizer uyumsuz sema (pretokenization); yeniden egitiliyor...", flush=True)
        vocab = int(raw.get("vocab_size", 2048)) if raw else 2048
        tokenizer = BPETokenizer.from_corpus(texts, vocab_size=vocab)
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer.to_dict(), f, ensure_ascii=False)
        print(f"tokenizer egitildi ve kaydedildi: {tok_path} (vocab={tokenizer.vocab_size})", flush=True)
    return tokenizer


def load_corpus(system_prompt):
    dialogues = load_jsonl_records(str(DATA_DIR / "data.jsonl"))
    large = load_jsonl_records(str(DATA_DIR / "data_large.jsonl"))
    if large:
        dialogues = dialogues + large
    corpus = build_assistant_only_corpus(dialogues, system_prompt=system_prompt, max_examples=100000)
    if not corpus:
        corpus = [
            {"system": system_prompt, "user": "Merhaba", "assistant": "Merhaba! Size nasıl yardımcı olabilirim?"},
            {"system": system_prompt, "user": "Hava nasıl?", "assistant": "Bugün hava çok güzel. Dışarı çıkıp kısa bir yürüyüş yapabilirsin."},
        ]
    else:
        with open(DATA_DIR / "chatml_instruct.jsonl", "w", encoding="utf-8") as f:
            for item in corpus:
                msg = [
                    {"role": "system", "content": item.get("system", system_prompt)},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]},
                ]
                f.write(json.dumps({"messages": msg}, ensure_ascii=False) + "\n")
    return corpus


def main():
    system_prompt = DEFAULT_SYSTEM_PROMPT

    corpus = load_corpus(system_prompt)
    tokenizer = build_tokenizer(corpus, str(DATA_DIR / "tokenizer.json"))

    cfg = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=8,
        n_head=8,
        n_kv_head=2,
        ff_dim=0,
        block_size=256,
        dropout=0.1,
        tie_weights=True,
        use_rope=True,
        grad_checkpointing=False,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup=300,
        weight_decay=0.1,
        grad_clip=1.0,
        betas=(0.9, 0.95),
        batch_size=32,
        grad_accum=1,
        epochs=8000,
        amp=True,
        seed=7,
    )

    device = pick_device()
    model, cfg = train_model(cfg, device=device, ckpt_path=str(CKPT_PATH), tokenizer=tokenizer, texts=corpus)

    prompt = format_chat_prompt("Merhaba, bugün nasıl yardımcı olabilirim?", system_prompt=system_prompt)
    generation_cfg = GenerationConfig(
        max_new_tokens=64, temperature=0.7, top_k=40, top_p=0.9,
        repetition_penalty=1.1, min_p=0.02, sampler="top_p",
    )
    generated = generate_text(model, tokenizer, prompt, generation=generation_cfg, device=device)
    print("prompt:", prompt)
    print("generated_text:", generated)

    restored_model, _ = load_checkpoint(str(CKPT_PATH), device=device)
    restored = generate_text(restored_model, tokenizer, prompt, generation=generation_cfg, device=device)
    print("restored_text:", restored)
    print("Encoder-Decoder + RoPE + GQA + cross-attention + KV cache model ok")


if __name__ == "__main__":
    main()
