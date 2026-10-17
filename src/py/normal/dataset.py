"""Instruction veri seti.

Iki mod desteklenir (`cfg.pack_sequences`):

* **pack_sequences=True (sequence packing, varsayilan)**: tum ornekler tek bir
  token akisina dizilir; pencere ici dolgu yoktur, egitim token butcesi tamamen
  dolu kalir. Pencereler ornek sinirlarini asabilir.
* **pack_sequences=False**: her ornek ayri bir egitim ornegidir; blok boyutuna
  kadar doldurulur (pad + ignore). Pencereler ornek sinirini asmaz.

Etiketler (labels) yalnizca asistan token'larinda gercek id olur; system/user ve
ozel prompt token'lari -100 yapilir. F.cross_entropy(ignore_index=-100) sayesinde
model sadece cevap vermeyi ogrenir, soruyu taklit etmez.

Ornekler `cfg.seed` ile karistirilir; boylece valayi (kuyruk) tek bir kaynaktan
gelmez ve temsili kalir.
"""

import random

import torch

from .config import TrainConfig
from .prompt import ChatPromptFormatter, clean_text
from .tokenizer import BPETokenizer


class InstructionDataset:
    IGNORE = -100

    def __init__(self, tokenizer: BPETokenizer, cfg: TrainConfig, texts):
        self.tok = tokenizer
        self.block_size = cfg.block_size
        self.pack = bool(cfg.pack_sequences)
        self.mode = "packed" if self.pack else "per_example"

        examples = []
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
            ids = [tokenizer.BOS] + prompt_ids + answer_ids
            labels = [self.IGNORE] + [self.IGNORE] * len(prompt_ids) + answer_ids
            examples.append((ids, labels))

        self.n_examples = len(examples)
        random.Random(cfg.seed).shuffle(examples)

        if self.pack:
            self._build_packed(examples, cfg)
        else:
            self._build_per_example(examples, cfg)

    # ------------------------------------------------------------------
    # Insa
    # ------------------------------------------------------------------
    def _build_packed(self, examples, cfg):
        base_ids = [self.tok.BOS]
        base_labels = [self.IGNORE]
        for ids, labels in examples:
            base_ids.extend(ids)
            base_labels.extend(labels)

        self.total_tokens = len(base_ids)
        self.learnable_tokens = sum(1 for x in base_labels if x != self.IGNORE)
        n_blocks = max(1, (self.total_tokens + self.block_size - 1) // self.block_size)
        self.packing_ratio = self.total_tokens / (n_blocks * self.block_size)

        train_ids, train_labels, val_ids, val_labels = self._split(base_ids, base_labels, cfg)
        self.train_x = torch.tensor(train_ids, dtype=torch.long)
        self.train_y = torch.tensor(train_labels, dtype=torch.long)
        self.val_x = torch.tensor(val_ids, dtype=torch.long)
        self.val_y = torch.tensor(val_labels, dtype=torch.long)
        self.n_train = len(train_ids)
        self.n_val = len(val_ids)

    def _build_per_example(self, examples, cfg):
        width = self.block_size + 1
        rows_x, rows_y = [], []
        fill = 0
        learnable = 0
        for ids, labels in examples:
            if len(ids) > width:
                # Cevap sonda oldugu icin kuyrugu tut; basi kirpilan prompt -100 zaten.
                ids = ids[-width:]
                labels = labels[-width:]
            fill += min(len(ids), width)
            learnable += sum(1 for x in labels if x != self.IGNORE)
            pad = width - len(ids)
            rows_x.append((ids + [self.tok.PAD] * pad)[: self.block_size])
            rows_y.append((labels + [self.IGNORE] * pad)[1:width])
        if not rows_x:
            rows_x = [[self.tok.BOS] + [self.tok.PAD] * (self.block_size - 1)]
            rows_y = [[self.IGNORE] * self.block_size]

        self.total_tokens = sum(min(len(ids), width) for ids, _ in examples) or self.block_size
        self.learnable_tokens = learnable
        self.packing_ratio = fill / max(1, len(examples) * width)

        x = torch.tensor(rows_x, dtype=torch.long)
        y = torch.tensor(rows_y, dtype=torch.long)
        n = x.size(0)
        cut = int(n * (1.0 - cfg.val_frac))
        cut = max(1, min(cut, n - 1)) if n > 1 else n
        self.train_x, self.train_y = x[:cut], y[:cut]
        self.val_x, self.val_y = (x[cut:], y[cut:]) if cut < n else (x[:cut], y[:cut])
        self.n_train = self.train_x.size(0)
        self.n_val = self.val_x.size(0)

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

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def get_batch(self, split, batch_size, device):
        x = self.train_x if split == "train" else self.val_x
        y = self.train_y if split == "train" else self.val_y
        if self.pack:
            hi = x.numel() - self.block_size - 1
            ix = torch.randint(0, max(1, hi), (batch_size,))
            xs = torch.stack([x[i:i + self.block_size] for i in ix])
            ys = torch.stack([y[i + 1:i + 1 + self.block_size] for i in ix])
        else:
            ix = torch.randint(0, max(1, x.size(0)), (batch_size,))
            xs, ys = x[ix], y[ix]
        return xs.to(device), ys.to(device)
