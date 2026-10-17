"""Veri okuma, temizleme ve instruction korpusu olusturma."""

import json
import os

import numpy as np
import torch

from .config import TrainConfig
from .prompt import DEFAULT_SYSTEM_PROMPT, clean_text
from .tokenizer import BPETokenizer


def _has_special_token(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in BPETokenizer.SPECIAL_STRINGS)


def load_jsonl_dialogues(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict) and "turns" in payload:
                rows.append(payload["turns"])
    return rows


def build_instruction_corpus(dialogues, system_prompt: str | None = None, max_turns: int | None = None):
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    corpus = []
    seen = set()

    for dialogue in dialogues[:max_turns] if max_turns else dialogues:
        if not isinstance(dialogue, (list, tuple)):
            continue
        for turn in dialogue:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            user, assistant = turn
            user_text = clean_text(user)
            assistant_text = clean_text(assistant)
            if len(user_text) < 3 or len(assistant_text) < 5:
                continue
            if _has_special_token(user_text) or _has_special_token(assistant_text):
                continue
            key = (user_text.lower(), assistant_text.lower())
            if key in seen:
                continue
            seen.add(key)
            corpus.append({
                "system": system_prompt,
                "user": user_text,
                "assistant": assistant_text,
            })
    return corpus


def build_assistant_only_corpus(dialogues, system_prompt: str | None = None, max_examples: int | None = None):
    examples = build_instruction_corpus(dialogues, system_prompt=system_prompt)
    if max_examples is not None:
        examples = examples[:max_examples]

    clean = []
    seen = set()
    for item in examples:
        user = clean_text(item["user"])
        assistant = clean_text(item["assistant"])
        if len(user) < 4 or len(assistant) < 4:
            continue
        if assistant.lower().startswith(user.lower()[:20]):
            continue
        if _has_special_token(user) or _has_special_token(assistant):
            continue
        key = (user.lower(), assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        clean.append({
            "system": clean_text(item.get("system", system_prompt or "")),
            "user": user,
            "assistant": assistant,
        })
    return clean


def build_long_clean_chat_corpus(dialogues, system_prompt: str | None = None, max_examples: int | None = None):
    return build_assistant_only_corpus(dialogues, system_prompt=system_prompt, max_examples=max_examples)


def build_chatml_dataset_file(data_path: str, out_path: str, system_prompt: str | None = None, max_examples: int | None = None):
    dialogues = load_jsonl_dialogues(data_path)
    corpus = build_long_clean_chat_corpus(dialogues, system_prompt=system_prompt, max_examples=max_examples)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in corpus:
            msg = [
                {"role": "system", "content": clean_text(item.get("system", system_prompt or ""))},
                {"role": "user", "content": clean_text(item["user"])},
                {"role": "assistant", "content": clean_text(item["assistant"])},
            ]
            f.write(json.dumps({"messages": msg}, ensure_ascii=False) + "\n")
    return corpus


def load_chatml_dataset(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                continue
            rows.append(messages)
    return rows


def make_synthetic_data(cfg: TrainConfig):
    rng = np.random.default_rng(cfg.seed)
    x = rng.integers(0, cfg.vocab_size, size=(cfg.batch_size, cfg.block_size), dtype=np.int64)
    y = np.roll(x, -1, axis=1)
    return torch.from_numpy(x), torch.from_numpy(y)
