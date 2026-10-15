"""Byte-level + BPE tokenizer.

- Temel byte tokenlar her zaman mevcuttur.
- BPE, veriden öğrenilen sıkı birleşimler ekler.
- Mevcut model API'si korunur: encode/decode, BOS/EOS/UNK/PAD sabitleri.
"""

import re
from collections import Counter


class BPETokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    OFFSET = 4

    def __init__(self, vocab_size=1024, merges=None):
        self.specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.vocab_size = max(260, int(vocab_size))
        self.merges = list(merges or [])
        self._id_to_token = {
            self.PAD: self.specials[0],
            self.BOS: self.specials[1],
            self.EOS: self.specials[2],
            self.UNK: self.specials[3],
        }
        self._token_to_id = {v: k for k, v in self._id_to_token.items()}
        for b in range(256):
            token = bytes([b])
            idx = self.OFFSET + b
            self._id_to_token[idx] = token
            self._token_to_id[token] = idx
        next_id = max(self._id_to_token) + 1
        for a, b in self.merges:
            merged = a + b
            if merged in self._token_to_id:
                continue
            self._id_to_token[next_id] = merged
            self._token_to_id[merged] = next_id
            next_id += 1
        self.vocab_size = max(self.vocab_size, next_id)

    @classmethod
    def from_corpus(cls, corpus, vocab_size=1024):
        vocab_size = max(260, int(vocab_size))
        if corpus is None:
            return cls(vocab_size=vocab_size, merges=[])

        texts = []
        if isinstance(corpus, str):
            texts = [corpus]
        elif isinstance(corpus, (bytes, bytearray)):
            texts = [corpus.decode("utf-8", errors="replace")]
        else:
            for item in corpus:
                if item is None:
                    continue
                if isinstance(item, dict):
                    for turn in item.get("turns", []):
                        if isinstance(turn, (list, tuple)) and len(turn) == 2:
                            user, assistant = turn
                            texts.extend([str(user), str(assistant)])
                    continue
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, (list, tuple)) and len(sub) == 2:
                            user, assistant = sub
                            texts.extend([str(user), str(assistant)])
                        else:
                            texts.append(str(sub))
                    continue
                texts.append(str(item))

        words = []
        for text in texts:
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            words.extend(bytes(part, "utf-8") for part in re.findall(r"\S+", text))

        if not words:
            return cls(vocab_size=vocab_size, merges=[])

        pieces = [[bytes([b]) for b in word] for word in words]
        merges = []
        while len(pieces[0]) + len(merges) < vocab_size - cls.OFFSET:
            counts = Counter()
            for word in pieces:
                for i in range(len(word) - 1):
                    counts[(word[i], word[i + 1])] += 1
            if not counts:
                break
            best_pair, best_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_count <= 0:
                break
            a, b = best_pair
            merged = a + b
            new_pieces = []
            changed = False
            for word in pieces:
                out = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                        out.append(merged)
                        i += 2
                        changed = True
                    else:
                        out.append(word[i])
                        i += 1
                new_pieces.append(out)
            if not changed:
                break
            merges.append((a, b))
            pieces = new_pieces
            if len(merges) + 256 + 4 >= vocab_size:
                break

        return cls(vocab_size=vocab_size, merges=merges)

    def _merge_once(self, tokens, a, b):
        out = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def encode(self, text):
        if text is None:
            return []
        if isinstance(text, (bytes, bytearray)):
            data = bytes(text)
        else:
            data = text.encode("utf-8")
        tokens = [bytes([b]) for b in data]
        for a, b in self.merges:
            tokens = self._merge_once(tokens, a, b)
        ids = []
        for tok in tokens:
            ids.append(self._token_to_id.get(tok, self.UNK))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for idx in ids:
            if idx < self.OFFSET:
                continue
            tok = self._id_to_token.get(idx)
            if isinstance(tok, bytes):
                buf.extend(tok)
            elif isinstance(tok, str):
                continue
        return bytes(buf).decode("utf-8", errors="replace")

    def is_special(self, i):
        return i < self.OFFSET

    def to_dict(self):
        return {"vocab_size": self.vocab_size, "merges": list(self.merges)}

    @classmethod
    def from_dict(cls, d):
        data = d or {}
        return cls(vocab_size=data.get("vocab_size", 1024), merges=data.get("merges", []))

    def to_dict(self):
        return {"vocab_size": self.vocab_size, "merges": [[list(a), list(b)] for a, b in self.merges]}

    @classmethod
    def from_dict(cls, data):
        merges = []
        for pair in data.get("merges", []):
            if len(pair) != 2:
                continue
            a, b = pair
            merges.append((bytes(a), bytes(b)))
        return cls(vocab_size=data.get("vocab_size", 1024), merges=merges)


class ByteTokenizer(BPETokenizer):
    def __init__(self):
        super().__init__(vocab_size=260, merges=[])
