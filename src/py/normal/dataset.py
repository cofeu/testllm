"""Seq2seq veri seti.

Her ornek bir (kaynak, hedef) ciftidir:
  kaynak : chat prompt'u (system + gecmis + guncel kullanici mesaji) -> encoder.
  hedef  : asistan cevabi -> decoder (teacher forcing).

Decoder girdisi `<bos>` ile baslar; hedefler bir kaydirilmis halidir ve cevabin
sonuna tam olarak bir `<|im_end|>` (EOS) eklenir. Dolgu (pad) etiketleri -100
yapilir; boylece kayip yalnizca gercek cevap token'larindan hesaplanir.

Bellek/hesap verimliligi:
  * Satirlar int32 olarak duz (flat) dizilerde + ofset dizisinde saklanir
    (int64'e gore yari bellek; blok boyutuna onceden doldurulmaz).
  * `get_batch` yalnizca o batch'teki en uzun diziye kadar doldurur (dinamik pad),
    boylece encoder/decoder hesaplarinin cogu dolguya gitmez.

Encoder dolgusu self-attention'da; encoder dolgusu decoder cross-attention'inda;
decoder dolgusu decoder self-attention'inda maskelenir (model.forward'a pad
maskeleri verilir).
"""

import random

import numpy as np
import torch

from .config import TrainConfig
from .prompt import ChatPromptFormatter, clean_text
from .tokenizer import BPETokenizer


class Seq2SeqDataset:
    IGNORE = -100

    def __init__(self, tokenizer: BPETokenizer, cfg: TrainConfig, texts):
        self.tok = tokenizer
        self.block_size = cfg.block_size

        src_rows, in_rows, out_rows = [], [], []
        for item in texts or []:
            if not isinstance(item, dict):
                continue
            user = clean_text(item.get("user", ""))
            assistant = clean_text(item.get("assistant", ""))
            if not user or not assistant:
                continue
            system = clean_text(item.get("system", "")) or None
            prompt = ChatPromptFormatter(system_prompt=system).format_prompt(user)

            src_ids = tokenizer.encode(prompt)
            if len(src_ids) > self.block_size:
                # Sondaki (en guncel) baglami koru; kullanicinin sorusu prompt'un
                # sonunda yer alir, bastan kirpmak soruyu silebilir.
                src_ids = src_ids[-self.block_size:]
            # Cevaplayi encoder'a koymuyoruz; IM_END'i BIR kez (tgt_out sonunda) ekle.
            answer = tokenizer.encode(assistant)[: self.block_size - 1]
            if not src_ids or not answer:
                continue

            tgt_in = [tokenizer.BOS] + answer
            tgt_out = answer + [tokenizer.IM_END]
            src_rows.append(src_ids)
            in_rows.append(tgt_in)
            out_rows.append(tgt_out)

        self.n_examples = len(src_rows)
        order = list(range(self.n_examples))
        random.Random(cfg.seed).shuffle(order)
        src_rows = [src_rows[i] for i in order]
        in_rows = [in_rows[i] for i in order]
        out_rows = [out_rows[i] for i in order]

        self.total_src_tokens = sum(len(r) for r in src_rows)
        self.total_tgt_tokens = sum(len(r) for r in out_rows)

        n = self.n_examples
        cut = int(n * (1.0 - cfg.val_frac))
        cut = max(1, min(cut, n - 1)) if n > 1 else n
        train_idx = list(range(cut))
        val_idx = list(range(cut, n)) or train_idx

        self._store = {
            "train": self._pack([src_rows[i] for i in train_idx],
                                [in_rows[i] for i in train_idx],
                                [out_rows[i] for i in train_idx]),
            "val": self._pack([src_rows[i] for i in val_idx],
                              [in_rows[i] for i in val_idx],
                              [out_rows[i] for i in val_idx]),
        }
        self.n_train = len(train_idx)
        self.n_val = len(val_idx)

    @staticmethod
    def _flat(rows):
        rows = rows or [[]]
        lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        data = np.empty(int(offsets[-1]), dtype=np.int32)
        for i, r in enumerate(rows):
            if r:
                data[offsets[i]:offsets[i + 1]] = r
        return data, offsets

    def _pack(self, src_rows, in_rows, out_rows):
        return {
            "src": self._flat(src_rows),
            "in": self._flat(in_rows),
            "out": self._flat(out_rows),
        }

    @staticmethod
    def _batch(store, ix, fill):
        data, offsets = store
        starts = offsets[ix]
        ends = offsets[ix + 1]
        max_len = max(1, int((ends - starts).max()))
        out = np.full((len(ix), max_len), fill, dtype=np.int64)
        for j in range(len(ix)):
            s, e = int(starts[j]), int(ends[j])
            if e > s:
                out[j, : e - s] = data[s:e]
        return out

    def get_batch(self, split, batch_size, device):
        store = self._store["train" if split == "train" else "val"]
        n = store["src"][1].size - 1
        ix = np.random.randint(0, max(1, n), size=batch_size)
        src = self._batch(store["src"], ix, self.tok.PAD)
        tgt_in = self._batch(store["in"], ix, self.tok.PAD)
        tgt_out = self._batch(store["out"], ix, self.IGNORE)
        return (
            torch.from_numpy(src).long().to(device),
            torch.from_numpy(tgt_in).long().to(device),
            torch.from_numpy(tgt_out).long().to(device),
        )
