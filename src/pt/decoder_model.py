import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass
class TrainConfig:
    # mimari
    vocab_size: int = 32
    d_model: int = 128
    n_layers: int = 8
    n_head: int = 8
    n_kv_head: int = 0  # 0 -> GQA yok (n_head'e esit)
    ff_dim: int = 512
    block_size: int = 96
    dropout: float = 0.10
    tie_weights: bool = True
    bias: bool = False
    use_rope: bool = False

    # optimizasyon
    learning_rate: float = 1.6e-4
    min_lr: float = 1.6e-5
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 12
    grad_accum: int = 1
    epochs: int = 200  # toplam optimizasyon adimi

    # degerlendirme / kayit
    val_frac: float = 0.1
    eval_every: int = 50
    eval_iters: int = 20
    save_every: int = 100
    log_every: int = 10

    # calisma ortami
    seed: int = 7
    device: str = "auto"
    amp: bool = True
    compile: bool = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 0.28
    top_k: int = 12
    top_p: float = 0.82
    repetition_penalty: float = 1.02
    min_p: float = 0.03


def seed_everything(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ChatPromptFormatter:
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or (
            "Sen Türkçe konuşan, yardımsever, net ve güvenilir bir yardımcı asistanısın. "
            "Yalnızca asistanın cevabını yaz, kullanıcı sorusunu veya sistem metnini tekrar etme. "
            "Cevaplar kısa, anlaşılır, doğal ve Türkçe olsun; hiçbir etiketi, başlığı veya tekrarı yeniden üretme."
        )

    def format_messages(self, messages):
        if not messages:
            return f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        blocks = [f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>"]
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = clean_text(str(msg.get("content", "")).strip())
            if not content:
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            blocks.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        if not messages or str(messages[-1].get("role", "user")).strip().lower() != "assistant":
            blocks.append("<|im_start|>assistant\n")
        return "\n".join(blocks)

    def format_prompt(self, user_text: str, history=None):
        history = history or []
        messages = [{"role": "user", "content": user_text}]
        for entry in history:
            if isinstance(entry, dict):
                messages.append(entry)
        return self.format_messages(messages)


def format_instruction_example(user_text: str, assistant_text: str, system_prompt: str | None = None):
    formatter = ChatPromptFormatter(system_prompt=system_prompt)
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return formatter.format_messages(messages)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model, n_head'in kati olmali")
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head if cfg.n_kv_head else cfg.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head, n_kv_head'in kati olmali (GQA)")
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            past_k, past_v = cache
            if past_k is not None and past_k.size(2) > 0:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = t > 1 and k.size(2) == t
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y)), new_cache


class SwiGLU(nn.Module):
    """x -> down( silu(gate(x)) * up(x) ). GELU FFN yerine (Llama tarzi)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        hidden = cfg.ff_dim
        self.gate = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = None if cfg.use_rope else nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") and p.ndim == 2:
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        caches=None,
        return_cache: bool = False,
        pos_offset: int = 0,
    ):
        if x.dim() != 2:
            raise ValueError(f"x must be [batch, seq], got {tuple(x.shape)}")
        b, t = x.shape
        if pos_offset + t > self.cfg.block_size:
            raise ValueError(f"sequence length {pos_offset + t} exceeds block_size {self.cfg.block_size}")

        h = self.token_emb(x)
        if self.pos_emb is not None:
            pos = torch.arange(pos_offset, pos_offset + t, device=x.device).unsqueeze(0).expand(b, -1)
            h = h + self.pos_emb(pos)
        h = self.drop(h)

        new_caches = [] if return_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = None if caches is None else caches[i]
            h, updated = block(h, cache=block_cache)
            if return_cache:
                new_caches.append(updated)

        h = self.norm(h)
        logits = self.head(h)

        if targets is None:
            return (logits, new_caches) if return_cache else logits

        flat_logits = logits.reshape(-1, self.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        if loss_mask is not None:
            flat_mask = loss_mask.reshape(-1).float()
            raw = F.cross_entropy(flat_logits, flat_targets, reduction="none", ignore_index=-100)
            loss = (raw * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        else:
            valid = (flat_targets != -100).sum().clamp(min=1)
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum", ignore_index=-100) / valid
        return logits, loss

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.02,
        repetition_penalty: float = 1.2,
        device: torch.device | None = None,
        eos_id: int = 2,
        forbidden_ids=None,
        stop_ids=None,
        use_cache: bool = True,
    ):
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        device = device or next(self.parameters()).device
        self.eval()
        generated = prompt.to(device).clone()

        if forbidden_ids is None:
            forbidden_ids = {0, 1, 3}
        forbidden_ids = {int(i) for i in forbidden_ids}
        stop_set = set()
        if eos_id is not None:
            stop_set.add(int(eos_id))
        if stop_ids:
            stop_set.update(int(i) for i in stop_ids)
        forbidden_ids -= stop_set

        caches = None
        cache_len = 0
        produced = 0
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if not use_cache:
                    window = generated[:, -self.cfg.block_size:]
                    logits = self(window)[:, -1, :]
                elif caches is None or cache_len >= self.cfg.block_size:
                    window = generated[:, -self.cfg.block_size:]
                    out, caches = self(window, return_cache=True)
                    cache_len = window.size(1)
                    logits = out[:, -1, :]
                else:
                    last = generated[:, -1:]
                    out, caches = self(last, caches=caches, return_cache=True, pos_offset=cache_len)
                    cache_len += 1
                    logits = out[:, -1, :]
                next_logits = logits / max(temperature, 1e-8)

                if repetition_penalty is not None and repetition_penalty > 1.0:
                    for prev in torch.unique(generated).tolist():
                        prev = int(prev)
                        if 0 <= prev < next_logits.size(-1):
                            value = next_logits[0, prev]
                            if value > 0:
                                next_logits[0, prev] = value / repetition_penalty
                            else:
                                next_logits[0, prev] = value * repetition_penalty

                if top_k is not None and top_k > 0:
                    topv, topi = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                    mask = torch.full_like(next_logits, float("-inf"))
                    mask.scatter_(1, topi, topv)
                    next_logits = mask

                probs = torch.softmax(next_logits, dim=-1)

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative <= top_p
                    if mask.any(dim=-1, keepdim=True).item() == 0:
                        mask[..., 0] = True
                    keep = torch.zeros_like(probs, dtype=torch.bool)
                    keep.scatter_(1, sorted_indices, mask)
                    probs = probs.masked_fill(~keep, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                if min_p is not None and min_p > 0.0:
                    min_p_value = min_p * probs.max(dim=-1, keepdim=True).values
                    probs = torch.where(probs >= min_p_value, probs, torch.zeros_like(probs))
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                for fid in forbidden_ids:
                    if 0 <= fid < probs.size(-1):
                        probs[..., fid] = 0.0
                if produced == 0:
                    for sid in stop_set:
                        if 0 <= sid < probs.size(-1):
                            probs[..., sid] = 0.0

                total = probs.sum(dim=-1, keepdim=True)
                if float(total.item()) <= 0.0:
                    probs = torch.ones_like(probs) / probs.size(-1)
                else:
                    probs = probs / total

                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token[0].item())

                generated = torch.cat([generated, next_token], dim=1)

                if token_id in stop_set:
                    break
                produced += 1
        return generated


def make_synthetic_data(cfg: TrainConfig):
    rng = np.random.default_rng(cfg.seed)
    x = rng.integers(0, cfg.vocab_size, size=(cfg.batch_size, cfg.block_size), dtype=np.int64)
    y = np.roll(x, -1, axis=1)
    return torch.from_numpy(x), torch.from_numpy(y)


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip().replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def save_checkpoint(model: nn.Module, cfg: TrainConfig, path: str):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base = getattr(model, "_orig_mod", model)
    payload = {"state_dict": base.state_dict(), "config": dict(cfg.__dict__)}
    torch.save(payload, str(save_path))


def load_checkpoint(path: str, device: torch.device | None = None):
    device = device or pick_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    known = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k in known})
    model = DecoderOnlyLM(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def configure_optimizer(model: nn.Module, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95))


def lr_at(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup > 0 and step < cfg.warmup:
        return cfg.learning_rate * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.epochs - cfg.warmup)
    prog = min(1.0, max(0.0, prog))
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


class InstructionDataset:
    """user/assistant orneklerini tek token akisina dizer, rastgele pencere keser.

    Etiketler (labels) yalnizca asistan token'larinda gercek id olur; system/user
    ve ozel prompt token'lari -100 yapilir. F.cross_entropy(ignore_index=-100)
    sayesinde model sadece cevap vermeyi ogrenir, soruyu taklit etmez.
    """

    IGNORE = -100

    def __init__(self, tokenizer, cfg: TrainConfig, texts):
        self.tok = tokenizer
        self.block_size = cfg.block_size
        base_ids = [tokenizer.BOS]
        base_labels = [self.IGNORE]
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
        self.train_ids, self.train_labels, self.val_ids, self.val_labels = self._split(
            base_ids, base_labels, cfg
        )
        self.n_train = len(self.train_ids)
        self.n_val = len(self.val_ids)

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
        ids = self.train_ids if split == "train" else self.val_ids
        labels = self.train_labels if split == "train" else self.val_labels
        arr = np.asarray(ids, dtype=np.int64)
        larr = np.asarray(labels, dtype=np.int64)
        hi = len(arr) - self.block_size - 1
        ix = np.random.randint(0, max(1, hi), size=batch_size)
        xs = np.stack([arr[i : i + self.block_size] for i in ix])
        ys = np.stack([larr[i + 1 : i + 1 + self.block_size] for i in ix])
        return (
            torch.from_numpy(xs).to(device),
            torch.from_numpy(ys).to(device),
        )


@torch.no_grad()
def estimate_loss(model, data, cfg, device):
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = data.get_batch("val", cfg.batch_size, device)
        if cfg.amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train_model(cfg: TrainConfig, device: torch.device | None = None, ckpt_path: str | None = None, tokenizer: BPETokenizer | None = None, texts=None):
    device = device or pick_device()
    seed_everything(cfg.seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = DecoderOnlyLM(cfg).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optim = configure_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if (cfg.amp and device.type == "cuda") else None

    data = InstructionDataset(tokenizer, cfg, texts) if (tokenizer is not None and texts) else None
    if data is not None:
        print(f"veri: train {data.n_train} token, val {data.n_val} token, vocab={cfg.vocab_size}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} parametre, d_model={cfg.d_model}, layers={cfg.n_layers}")

    ckpt_path = ckpt_path or os.path.join(os.path.dirname(__file__), "decoder_model_ckpt.pt")

    for step in range(1, cfg.epochs + 1):
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group["lr"] = lr
        model.train()
        optim.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(max(1, cfg.grad_accum)):
            if data is not None:
                x, y = data.get_batch("train", cfg.batch_size, device)
            else:
                x, y = make_synthetic_data(cfg)
                x, y = x.to(device), y.to(device)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss / cfg.grad_accum).backward()
            else:
                _, loss = model(x, y)
                (loss / cfg.grad_accum).backward()
            running += loss.item()

        if cfg.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        if step % cfg.log_every == 0 or step == 1:
            print(f"step={step} loss={running / max(1, cfg.grad_accum):.4f} lr={lr:.2e} device={device}", flush=True)

        if data is not None and (step % cfg.eval_every == 0 or step == cfg.epochs):
            val = estimate_loss(model, data, cfg, device)
            print(f"eval step={step} val_loss={val:.4f}", flush=True)

        if step % cfg.save_every == 0 or step == cfg.epochs:
            save_checkpoint(model, cfg, ckpt_path)

    save_checkpoint(model, cfg, ckpt_path)
    print(f"checkpoint saved: {ckpt_path}")
    return model, cfg


def generate_text(model: DecoderOnlyLM, tokenizer: BPETokenizer, prompt: str, generation: GenerationConfig | None = None, device: torch.device | None = None):
    device = device or pick_device()
    generation = generation or GenerationConfig()
    prompt_text = str(prompt or "")
    if not prompt_text.strip():
        return ""
    if not prompt_text.endswith("\n"):
        prompt_text = prompt_text + "\n"

    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        return ""

    if len(prompt_ids) > model.cfg.block_size:
        prompt_ids = prompt_ids[-model.cfg.block_size:]

    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
        min_p=generation.min_p,
        repetition_penalty=generation.repetition_penalty,
        device=device,
        eos_id=tokenizer.EOS,
        forbidden_ids={tokenizer.PAD, tokenizer.BOS, tokenizer.UNK, tokenizer.IM_START},
        stop_ids={tokenizer.IM_END},
    )
    output_ids = generated[0].tolist()
    continuation_ids = output_ids[len(prompt_ids):]
    visible_ids = [idx for idx in continuation_ids if idx >= tokenizer.OFFSET]
    if not visible_ids:
        return ""

    decoded = tokenizer.decode(visible_ids)
    return decoded.replace("\x00", "").strip()


def format_chat_prompt(user_text: str, system_prompt: str | None = None, history=None):
    formatter = ChatPromptFormatter(system_prompt=system_prompt)
    return formatter.format_prompt(user_text, history=history)


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    data_path = os.path.join(data_dir, "data.jsonl")
    data_large_path = os.path.join(data_dir, "data_large.jsonl")
    chml_path = os.path.join(data_dir, "chatml_instruct.jsonl")
    system_prompt = (
        "Sen Türkçe konuşan, kısa, net ve güvenilir bir yardımcı asistanısın. "
        "Cevaplar kısa, anlaşılır, samimi ve işlevsel olsun. "
        "Asla kullanıcı sorusunu tekrar etme; yalnızca asistanın doğal, öz ve yardımcı cevabını üret."
    )

    dialogues = load_jsonl_dialogues(data_path)
    large_dialogues = load_jsonl_dialogues(data_large_path)
    if large_dialogues:
        dialogues = dialogues + large_dialogues
    corpus = build_assistant_only_corpus(dialogues, system_prompt=system_prompt, max_examples=100000)
    if not corpus:
        corpus = [
            {"system": system_prompt, "user": "Merhaba", "assistant": "Merhaba! Size nasıl yardımcı olabilirim?"},
            {"system": system_prompt, "user": "Hava nasıl?", "assistant": "Bugün hava çok güzel. Dışarı çıkıp kısa bir yürüyüş yapabilirsin."},
        ]
    else:
        with open(chml_path, "w", encoding="utf-8") as f:
            for item in corpus:
                msg = [
                    {"role": "system", "content": clean_text(item.get("system", system_prompt))},
                    {"role": "user", "content": clean_text(item["user"])},
                    {"role": "assistant", "content": clean_text(item["assistant"])},
                ]
                f.write(json.dumps({"messages": msg}, ensure_ascii=False) + "\n")

    tokenizer_texts = [item["user"] for item in corpus if isinstance(item, dict)] + [
        item["assistant"] for item in corpus if isinstance(item, dict)
    ]
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        with open(tok_path, "r", encoding="utf-8") as f:
            tokenizer = BPETokenizer.from_dict(json.load(f))
        print(f"tokenizer yuklendi: {tok_path} (vocab={tokenizer.vocab_size})")
    else:
        tokenizer = BPETokenizer.from_corpus(tokenizer_texts, vocab_size=2048)
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer.to_dict(), f, ensure_ascii=False)
        print(f"tokenizer egitildi ve kaydedildi: {tok_path} (vocab={tokenizer.vocab_size})")
    cfg = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=192,
        n_layers=6,
        n_head=6,
        n_kv_head=2,
        ff_dim=768,
        block_size=128,
        dropout=0.1,
        tie_weights=True,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup=200,
        weight_decay=0.1,
        grad_clip=1.0,
        batch_size=32,
        grad_accum=1,
        epochs=6000,
        val_frac=0.05,
        eval_every=250,
        eval_iters=50,
        save_every=500,
        log_every=100,
        amp=True,
        seed=7,
    )
    device = pick_device()
    model, cfg = train_model(
        cfg,
        device=device,
        ckpt_path=os.path.join(os.path.dirname(__file__), "decoder_model_ckpt.pt"),
        tokenizer=tokenizer,
        texts=corpus,
    )

    prompt = format_chat_prompt(
        "Merhaba, bugün nasıl yardımcı olabilirim?",
        system_prompt=system_prompt,
    )
    generation_cfg = GenerationConfig(max_new_tokens=64, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1, min_p=0.02)
    generated = generate_text(model, tokenizer, prompt, generation=generation_cfg, device=device)
    print("prompt:", prompt)
    print("generated_text:", generated)

    restored_model, _ = load_checkpoint(os.path.join(os.path.dirname(__file__), "decoder_model_ckpt.pt"), device=device)
    restored = generate_text(restored_model, tokenizer, prompt, generation=generation_cfg, device=device)
    print("restored_text:", restored)
    print("instruction-tuned + chatml + longer-capacity model ok")
