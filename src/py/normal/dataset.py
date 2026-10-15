"""Compatibility layer: the normal dataset path follows the PT implementation."""

import json
import random

import numpy as np
import torch

from src.pt.decoder_model import ChatPromptFormatter, clean_text

PROMPT_PREFIX = "<|im_start|>user\n"
RESPONSE_PREFIX = "<|im_start|>assistant\n"
SYSTEM_PROMPT = (
    "Sen Türkçe konuşan, kısa ve net cevap veren bir yardımcı asistanısın. "
    "Cevaplar kısa, anlaşılır ve Türkçe olsun."
)

normalize_text = clean_text


def normalize_turns(turns):
    out = []
    for pair in turns or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        user = normalize_text(pair[0])
        assistant = normalize_text(pair[1])
        if not user or not assistant:
            continue
        out.append((user, assistant))
    return out


def load_dialogues(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns = obj.get("turns", [])
                cleaned = normalize_turns(turns)
                if cleaned:
                    rows.append(cleaned)
    except OSError:
        return []
    return rows


def build_stream(tokenizer, turns, system_prompt=None):
    formatter = ChatPromptFormatter(system_prompt=system_prompt or SYSTEM_PROMPT)
    ids = [tokenizer.BOS]
    for user, assistant in normalize_turns(turns):
        text = formatter.format_messages([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])
        ids.extend(tokenizer.encode(text))
        ids.append(tokenizer.EOS)
    return ids


class ChatDataset:
    def __init__(self, tokenizer, dialogues, block_size, val_frac=0.1, seed=1337, system_prompt=None):
        self.block_size = block_size
        streams = [build_stream(tokenizer, turns, system_prompt=system_prompt) for turns in dialogues]
        rng = random.Random(seed)
        order = list(range(len(streams)))
        rng.shuffle(order)
        n_val = max(1, int(len(streams) * val_frac)) if len(streams) > 1 else 0
        val_set = set(order[:n_val])
        train_streams = [s for i, s in enumerate(streams) if i not in val_set]
        val_streams = [s for i, s in enumerate(streams) if i in val_set]
        if not train_streams:
            train_streams = streams[:1]
        self.train = self._concat(train_streams)
        self.val = self._concat(val_streams) if val_streams else self.train[:1]

    @staticmethod
    def _concat(streams):
        return np.concatenate([np.asarray(s, dtype=np.int64) for s in streams])

    def get_batch(self, split, batch_size, device):
        src = self.train if split == "train" else self.val
        n = len(src)
        if n < self.block_size + 1:
            pad = np.full(self.block_size + 1 - n, 0, dtype=np.int64)
            src = np.concatenate([src, pad])
            n = len(src)
        hi = n - self.block_size - 1
        ix = torch.randint(0, max(1, hi), (batch_size,))
        xs = np.stack([src[i : i + self.block_size] for i in ix.tolist()])
        ys = np.stack([src[i + 1 : i + 1 + self.block_size] for i in ix.tolist()])
        x = torch.from_numpy(np.ascontiguousarray(xs)).to(device)
        y = torch.from_numpy(np.ascontiguousarray(ys)).to(device)
        return x, y
