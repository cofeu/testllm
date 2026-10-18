"""Seq2seq veri seti.

Her ornek bir (kaynak, hedef) ciftidir:
  kaynak : chat prompt'u (system + gecmis + guncel kullanici mesaji) -> encoder.
  hedef  : asistan cevabi -> decoder (teacher forcing).

Decoder girdisi `<bos>` ile baslar; hedefler bir kaydirilmis halidir ve cevabin
sonuna tam olarak bir `<|im_end|>` (EOS) eklenir. Cevap blok boyutuna sigmiyorsa
kirpilir ve bu durumda EOS EKLENMEZ (model yarida kesmeyi ogrenmesin).

Bellek/hesap verimliligi:
  * Satirlar int32 olarak duz (flat) dizilerde + ofset dizisinde saklanir
    (int64'e gore yari bellek; blok boyutuna onceden doldurulmaz).
  * Uzunluga gore gruplama (length bucketing): ornekler `max(src_len, tgt_len)`
    8'lik kovalara ayrilir; `get_batch` once rastgele bir kova secer (ornek
    sayisina gore agirlikli), sonra o kovadan ornekler. Boylece ayni batch'e
    benzer uzunluktaki ornekler duser ve tek uzun ornek kisa ornekleri onun
    boyuna cekmez.
  * Her tensor kendi batch maksimumuna (8'in katina yuvarlanmis) doldurulur;
    src ve tgt bagimsiz doldurulur, dolgu minimize edilir.

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

PAD_MULTIPLE = 8


class Seq2SeqDataset:
    IGNORE = -100

    def __init__(self, tokenizer: BPETokenizer, cfg: TrainConfig, texts, bucket_width: int = 8):
        self.tok = tokenizer
        self.block_size = cfg.block_size
        self.bucket_width = max(1, int(bucket_width))

        src_rows, in_rows, out_rows = [], [], []
        for item in texts or []:
            if not isinstance(item, dict):
                continue
            user = clean_text(item.get("user", ""), keep_newlines=True)
            assistant = clean_text(item.get("assistant", ""), keep_newlines=True)
            if not user or not assistant:
                continue
            system = clean_text(item.get("system", ""), keep_newlines=True) or None
            history = item.get("history") or None
            prompt = ChatPromptFormatter(system_prompt=system).format_prompt(user, history=history)

            src_ids = tokenizer.encode(prompt)
            if len(src_ids) > self.block_size:
                # Sondaki (en guncel) baglami koru; kullanicinin sorusu prompt'un
                # sonunda yer alir, bastan kirpmak soruyu silebilir.
                src_ids = src_ids[-self.block_size:]

            answer = tokenizer.encode(assistant)
            truncated = len(answer) > self.block_size - 1
            answer = answer[: self.block_size - 1]
            if not src_ids or not answer:
                continue

            tgt_in = [tokenizer.BOS] + answer
            # Kirpilan cevaba EOS eklemeyiz: aksi halde model yarida bitirmeyi ogrenir.
            tgt_out = list(answer) if truncated else answer + [tokenizer.IM_END]
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
        self._buckets, self._bucket_of = {}, {}
        for k, v in self._store.items():
            arrays, bucket_of = self._bucketize(v["src"][1], v["out"][1])
            self._buckets[k] = arrays
            self._bucket_of[k] = bucket_of
        self.n_train = len(train_idx)
        self.n_val = len(val_idx)

    # ------------------------------------------------------------------
    # Depolama
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Uzunluga gore kovalama
    # ------------------------------------------------------------------
    def _bucketize(self, src_offsets, out_offsets):
        """Ornekleri `max(src_len, tgt_len)` 8'lik kovalara ayirir.

        Doner: (kovalarin pozisyon dizileri listesi, pozisyon -> kova indeksi).
        """
        n = src_offsets.size - 1
        keys = np.empty(n, dtype=np.int64)
        for pos in range(n):
            s = int(src_offsets[pos + 1] - src_offsets[pos])
            t = int(out_offsets[pos + 1] - out_offsets[pos])
            keys[pos] = max(s, t) // self.bucket_width
        groups = {}
        for pos, k in enumerate(keys.tolist()):
            groups.setdefault(int(k), []).append(pos)
        arrays = []
        bucket_of = np.empty(n, dtype=np.int64)
        for bi, positions in enumerate(groups.values()):
            arr = np.array(positions, dtype=np.int64)
            arrays.append(arr)
            bucket_of[arr] = bi
        return arrays, bucket_of

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    @staticmethod
    def _round_up(max_len, cap):
        rounded = ((max_len + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        return min(cap, max(max_len, rounded))

    def _batch(self, store, ix, fill):
        data, offsets = store
        starts = offsets[ix]
        ends = offsets[ix + 1]
        raw_max = max(1, int((ends - starts).max()))
        max_len = self._round_up(raw_max, self.block_size)
        out = np.full((len(ix), max_len), fill, dtype=np.int64)
        for j in range(len(ix)):
            s, e = int(starts[j]), int(ends[j])
            if e > s:
                out[j, : e - s] = data[s:e]
        return out

    def get_batch(self, split, batch_size, device):
        key = "train" if split == "train" else "val"
        store = self._store[key]
        n = store["src"][1].size - 1
        # Rastgele bir ornek sec, onun kovasini kullan (kova secimi ornek
        # sayisina gore agirlikli olur), sonra o kovadan batch'i doldur.
        probe = int(np.random.randint(0, max(1, n)))
        bucket = self._buckets[key][int(self._bucket_of[key][probe])]
        ix = bucket[np.random.randint(0, bucket.size, size=batch_size)]
        src = self._batch(store["src"], ix, self.tok.PAD)
        tgt_in = self._batch(store["in"], ix, self.tok.PAD)
        tgt_out = self._batch(store["out"], ix, self.IGNORE)
        return (
            torch.from_numpy(src).long().to(device),
            torch.from_numpy(tgt_in).long().to(device),
            torch.from_numpy(tgt_out).long().to(device),
        )

    def eval_batches(self, split, batch_size, device):
        """Deterministik degerlendirme: tum split'i kovalara gore sirayla, bir kez dolasir.

        `get_batch`'in aksine ornekleme yapmaz; ayni veri uzerinde tekrarlanabilir
        sonuc verir ve her ornegi tam olarak bir kez gorur.
        """
        key = "train" if split == "train" else "val"
        store = self._store[key]
        bs = max(1, int(batch_size))
        for bucket in self._buckets[key]:
            for start in range(0, int(bucket.size), bs):
                ix = bucket[start:start + bs]
                src = self._batch(store["src"], ix, self.tok.PAD)
                tgt_in = self._batch(store["in"], ix, self.tok.PAD)
                tgt_out = self._batch(store["out"], ix, self.IGNORE)
                yield (
                    torch.from_numpy(src).long().to(device),
                    torch.from_numpy(tgt_in).long().to(device),
                    torch.from_numpy(tgt_out).long().to(device),
                )

