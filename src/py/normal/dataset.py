"""Instruction veri seti: ornekleri dolgusuz bloklara paketler (sequence packing).

Etiketler (labels) yalnizca asistan token'larinda gercek id olur; system/user
ve ozel prompt token'lari -100 yapilir. F.cross_entropy(ignore_index=-100)
sayesinde model sadece cevap vermeyi ogrenir, soruyu taklit etmez.

Sequence packing: tum ornekler tek bir token akisina dizildigi icin pencere
ici dolgu yoktur; egitim token butcesi tamamen dolu kalir.
"""

import numpy as np
import torch

from .config import TrainConfig
from .prompt import ChatPromptFormatter, clean_text
from .tokenizer import BPETokenizer


class InstructionDataset:
    IGNORE = -100

    def __init__(self, tokenizer: BPETokenizer, cfg: TrainConfig, texts):
        self.tok = tokenizer
        self.block_size = cfg.block_size

        base_ids = [tokenizer.BOS]
        base_labels = [self.IGNORE]
        n_examples = 0
        for item in texts or []:
            if not isinstance(item, dict):
                continue
            user = clean_text(item.get("user", ""))
            assistant = clean_text(item.get("assistant", ""))
            if not user or not assistant:
                continue
            system = clean_text(item.get("system", "")) or None
            prompt = ChatPromptFormatter(system_prompt=system).format_prompt(user)
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(assistant) + [tokenizer.IM_END]
            base_ids.extend(prompt_ids)
            base_labels.extend([self.IGNORE] * len(prompt_ids))
            base_ids.extend(answer_ids)
            base_labels.extend(answer_ids)
            n_examples += 1

        self.n_examples = n_examples
        self.total_tokens = len(base_ids)
        self.learnable_tokens = sum(1 for x in base_labels if x != self.IGNORE)
        self.packing_ratio = self.total_tokens / max(1, (self.total_tokens + self.block_size - 1) // self.block_size) / self.block_size

        train_ids, train_labels, val_ids, val_labels = self._split(base_ids, base_labels, cfg)
        self.train_x = torch.tensor(train_ids, dtype=torch.long)
        self.train_y = torch.tensor(train_labels, dtype=torch.long)
        self.val_x = torch.tensor(val_ids, dtype=torch.long)
        self.val_y = torch.tensor(val_labels, dtype=torch.long)
        self.n_train = len(train_ids)
        self.n_val = len(val_ids)

    def _split(self, ids, labels, cfg):
        need = cfg.block_size + 2
        if len(ids) < need:
            pad = need - len(ids)
            ids = ids + [self.tok.PAD] * pad
            labels = labels + [self.IGNORE] * pad
        n = len(ids)
        cut = int(n * (1.0 - cfg.val_frac))
        cut = max(cfg.block_size + 1, min(cut, n - (cfg.block_size + 1)))
        train_ids, train_labels = ids[:cut], labels[:cut]
        val_ids, val_labels = ids[cut:], labels[cut:]
        if len(val_ids) < need:
            val_ids, val_labels = train_ids, train_labels
        return train_ids, train_labels, val_ids, val_labels

    def get_batch(self, split, batch_size, device):
        arr = self.train_x if split == "train" else self.val_x
        lab = self.train_y if split == "train" else self.val_y
        hi = arr.numel() - self.block_size - 1
        ix = torch.randint(0, max(1, hi), (batch_size,))
        xs = torch.stack([arr[i:i + self.block_size] for i in ix])
        ys = torch.stack([lab[i + 1:i + 1 + self.block_size] for i in ix])
        return xs.to(device), ys.to(device)
