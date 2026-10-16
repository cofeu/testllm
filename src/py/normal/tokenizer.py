"""Byte-level BPE tokenizer (UTF-8 bayt tabanli, ozel chat token'lari ile).

Kanonik implementasyon. `encode` kelime-memoizasyonu kullanir; BPE egitimi
artimli (incremental) calisir.
"""

import re
from collections import Counter, defaultdict


class BPETokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    IM_START = 4   # <|im_start|>
    IM_END = 5     # <|im_end|>
    OFFSET = 6
    SPECIAL_STRINGS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>"]

    def __init__(self, vocab_size=1024, merges=None):
        self.specials = list(self.SPECIAL_STRINGS)
        self.vocab_size = max(self.OFFSET + 256, int(vocab_size))
        self.merges = list(merges or [])
        self._ranks = None
        self._word_cache = {}
        self._id_to_token = {i: s for i, s in enumerate(self.specials)}
        self._token_to_id = {s: i for i, s in enumerate(self.specials)}
        self._special_strings = set(self.specials)
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

        # ---- artimli (incremental) BPE: O(merges x etkilenen kelime) ----
        word_freq = {}
        for word in words:
            key = tuple(bytes([b]) for b in word)
            if key:
                word_freq[key] = word_freq.get(key, 0) + 1

        word_list = [list(k) for k in word_freq]
        freqs = list(word_freq.values())
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for wi, w in enumerate(word_list):
            f = freqs[wi]
            for i in range(len(w) - 1):
                pair = (w[i], w[i + 1])
                pair_counts[pair] += f
                pair_words[pair].add(wi)

        merges = []
        max_merges = max(0, vocab_size - cls.OFFSET - 256)
        while len(merges) < max_merges and pair_counts:
            best_count = max(pair_counts.values())
            if best_count <= 0:
                break
            best_pair = max(p for p, c in pair_counts.items() if c == best_count)
            a, b = best_pair
            merged = a + b

            affected = list(pair_words.pop(best_pair, ()))
            pair_counts.pop(best_pair, None)
            for wi in affected:
                w = word_list[wi]
                f = freqs[wi]
                for i in range(len(w) - 1):
                    pair = (w[i], w[i + 1])
                    c = pair_counts.get(pair, 0) - f
                    if c <= 0:
                        pair_counts.pop(pair, None)
                        pair_words.get(pair, set()).discard(wi)
                    else:
                        pair_counts[pair] = c
                new_word = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(w[i])
                        i += 1
                word_list[wi] = new_word
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_counts[pair] += f
                    pair_words[pair].add(wi)
            merges.append((a, b))

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

    def _bpe(self, syms):
        """Sembol listesini (bytes) rank-oncelikli greedy BPE ile birlestirir."""
        if not self.merges or len(syms) < 2:
            return list(syms)
        if self._ranks is None or len(self._ranks) != len(self.merges):
            self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        ranks = self._ranks
        syms = list(syms)
        while len(syms) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            a, b = self.merges[best_rank]
            syms[best_idx:best_idx + 2] = [a + b]
        return syms

    def _encode_word(self, word: str):
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        syms = self._bpe([bytes([b]) for b in word.encode("utf-8")])
        ids = [self._token_to_id.get(tok, self.UNK) for tok in syms]
        self._word_cache[word] = ids
        return ids

    def _encode_plain(self, text: str):
        ids = []
        for chunk in re.findall(r"\S+|\s+", text):
            if chunk.isspace():
                base = self.OFFSET
                ids.extend(base + b for b in chunk.encode("utf-8"))
            else:
                ids.extend(self._encode_word(chunk))
        return ids

    def _encode_bytes(self, data):
        if not data:
            return []
        syms = self._bpe([bytes([b]) for b in data])
        return [self._token_to_id.get(tok, self.UNK) for tok in syms]

    def encode(self, text):
        if text is None:
            return []
        if isinstance(text, (bytes, bytearray)):
            return self._encode_bytes(bytes(text))
        data = str(text)
        if not data:
            return []
        present = [s for s in self.specials if s in data]
        if present:
            present.sort(key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(s) for s in present) + ")"
            parts = re.split(pattern, data)
        else:
            parts = [data]
        ids = []
        for part in parts:
            if part in self._special_strings:
                ids.append(self._token_to_id[part])
            elif part:
                ids.extend(self._encode_plain(part))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for idx in ids:
            if idx < self.OFFSET:
                continue
            tok = self._id_to_token.get(idx)
            if isinstance(tok, bytes):
                buf.extend(tok)
        return bytes(buf).decode("utf-8", errors="replace")

    def is_special(self, i):
        return i < self.OFFSET

    def to_dict(self):
        return {"vocab_size": self.vocab_size, "merges": [[list(a), list(b)] for a, b in self.merges]}

    @classmethod
    def from_dict(cls, data):
        payload = data or {}
        merges = []
        for pair in payload.get("merges", []):
            if len(pair) != 2:
                continue
            a, b = pair
            merges.append((bytes(a), bytes(b)))
        return cls(vocab_size=payload.get("vocab_size", 1024), merges=merges)
