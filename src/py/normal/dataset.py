"""Seq2seq veri seti.

Her ornek bir (kaynak, hedef) ciftidir:
  kaynak : chat prompt'u (system + gecmis + guncel kullanici mesaji) -> encoder.
  hedef  : asistan cevabi -> decoder (teacher forcing).

Decoder girdisi `<bos>` ile baslar; hedefler bir kaydirilmis halidir ve cevabin
sonuna `<|im_end|>` (EOS) eklenir. Dolgu (pad) etiketleri -100 yapilir; boylece
kayip yalnizca gercek cevap token'larindan hesaplanir.

Encoder dolgusu self-attention'da maskelenir; encoder dolgusu decoder
cross-attention'inda da maskelenir (model.forward'a pad maskeleri verilir).
"""

import random

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
            answer = (tokenizer.encode(assistant) + [tokenizer.IM_END])[: self.block_size - 1]
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

        src = self._pad(src_rows, self.tok.PAD)
        tgt_in = self._pad(in_rows, self.tok.PAD)
        tgt_out = self._pad(out_rows, self.IGNORE)

        n = self.n_examples
        cut = int(n * (1.0 - cfg.val_frac))
        cut = max(1, min(cut, n - 1)) if n > 1 else n
        if cut < n:
            self.train_src, self.val_src = src[:cut], src[cut:]
            self.train_in, self.val_in = tgt_in[:cut], tgt_in[cut:]
            self.train_out, self.val_out = tgt_out[:cut], tgt_out[cut:]
        else:
            self.train_src, self.val_src = src, src
            self.train_in, self.val_in = tgt_in, tgt_in
            self.train_out, self.val_out = tgt_out, tgt_out
        self.n_train = self.train_src.size(0)
        self.n_val = self.val_src.size(0)

    def _pad(self, rows, fill):
        if not rows:
            rows = [[self.tok.PAD]]
        width = self.block_size
        padded = [(r[:width] + [fill] * (width - len(r)))[:width] for r in rows]
        return torch.tensor(padded, dtype=torch.long)

    def get_batch(self, split, batch_size, device):
        if split == "train":
            src, tgt_in, tgt_out = self.train_src, self.train_in, self.train_out
        else:
            src, tgt_in, tgt_out = self.val_src, self.val_in, self.val_out
        ix = torch.randint(0, max(1, src.size(0)), (batch_size,))
        return src[ix].to(device), tgt_in[ix].to(device), tgt_out[ix].to(device)
