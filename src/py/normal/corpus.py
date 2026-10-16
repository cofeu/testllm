"""Veri okuma, temizleme ve instruction korpusu olusturma."""

import json
import os

import numpy as np
import torch

from .config import TrainConfig
from .prompt import ChatPromptFormatter, DEFAULT_SYSTEM_PROMPT, clean_text
from .tokenizer import BPETokenizer


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
    system_prompt = system_prompt or (
        "Sen Türkçe konuşan, yardımsever, net ve güvenilir bir asistansın. "
        "Cevaplar kısa, anlaşılır, samimi ve işlevsel olsun. "
        "Kullanıcı sorusunu tekrar etme; yalnızca asistanın yanıtını üret."
    )
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
            if any(token in assistant_text.lower() for token in ["<|im_start|>", "<|im_end|>"]):
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
        if any(token in assistant.lower() for token in ["<|im_start|>", "<|im_end|>"]):
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


def build_strict_instruction_corpus(dialogues, system_prompt: str | None = None, max_examples: int | None = None):
    base = build_long_clean_chat_corpus(dialogues, system_prompt=system_prompt)
    if max_examples is not None:
        base = base[:max_examples]
    strict = []
    for item in base:
        user = clean_text(item["user"])
        assistant = clean_text(item["assistant"])
        if len(user) < 4 or len(assistant) < 4:
            continue
        if len(user) > 480 or len(assistant) > 700:
            continue
        strict.append({
            "system": clean_text(item.get("system", system_prompt or "")),
            "user": user,
            "assistant": assistant,
        })
    return strict


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


def build_text_sequences(tokenizer: BPETokenizer, dialogues, block_size: int):
    sequences = []
    for dialogue in dialogues:
        if not isinstance(dialogue, (list, tuple)):
            continue
        buffer = [tokenizer.BOS]
        for turn in dialogue:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            user, assistant = turn
            user_ids = tokenizer.encode(str(user))
            assistant_ids = tokenizer.encode(str(assistant))
            if user_ids:
                buffer.extend(user_ids)
            buffer.append(tokenizer.EOS)
            if assistant_ids:
                buffer.extend(assistant_ids)
            buffer.append(tokenizer.EOS)
        if len(buffer) >= 2:
            sequences.append(buffer)

    if not sequences:
        return []

    chunked = []
    for seq in sequences:
        for i in range(0, len(seq) - 1, 1):
            chunk = seq[i:i + block_size]
            if len(chunk) >= 2:
                chunked.append(chunk)
    return chunked


def make_synthetic_data(cfg: TrainConfig):
    rng = np.random.default_rng(cfg.seed)
    x = rng.integers(0, cfg.vocab_size, size=(cfg.batch_size, cfg.block_size), dtype=np.int64)
    y = np.roll(x, -1, axis=1)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_text_data(cfg: TrainConfig, tokenizer: BPETokenizer, texts):
    if not texts:
        return make_synthetic_data(cfg)

    rows = []
    for text in texts:
        if text is None:
            continue
        if isinstance(text, dict):
            user = clean_text(text.get("user", ""))
            assistant = clean_text(text.get("assistant", ""))
            system = clean_text(text.get("system", ""))
            if not user or not assistant:
                continue
            formatter = ChatPromptFormatter(system_prompt=system or None)
            prompt = formatter.format_prompt(user)
            assistant_prefix = "<|im_start|>assistant\n"
            if not prompt.endswith(assistant_prefix):
                prompt = prompt.rstrip() + "\n" + assistant_prefix
            seq = [tokenizer.BOS] + tokenizer.encode(prompt) + tokenizer.encode(assistant) + [tokenizer.EOS]
        elif isinstance(text, str):
            seq = [tokenizer.BOS] + tokenizer.encode(text) + [tokenizer.EOS]
        elif isinstance(text, (list, tuple)):
            seq = [tokenizer.BOS]
            for item in text:
                seq.extend(tokenizer.encode(str(item)))
                seq.append(tokenizer.EOS)
        else:
            seq = [tokenizer.BOS] + tokenizer.encode(str(text)) + [tokenizer.EOS]
        if len(seq) >= 2:
            rows.append(seq)

    if not rows:
        return make_synthetic_data(cfg)

    batch_x = []
    batch_y = []
    for seq in rows[:max(1, cfg.batch_size)]:
        if len(seq) < 2:
            continue
        x = seq[:-1]
        y = seq[1:]
        pad_len = max(0, cfg.block_size - len(x))
        x = x[: cfg.block_size] + [tokenizer.PAD] * pad_len
        y = y[: cfg.block_size] + [tokenizer.PAD] * pad_len
        batch_x.append(x)
        batch_y.append(y)

    while len(batch_x) < cfg.batch_size:
        x, y = make_synthetic_data(cfg)
        batch_x.append(x[0].tolist())
        batch_y.append(y[0].tolist())

    x_tensor = torch.tensor(batch_x[: cfg.batch_size], dtype=torch.long)
    y_tensor = torch.tensor(batch_y[: cfg.batch_size], dtype=torch.long)
    return x_tensor, y_tensor
