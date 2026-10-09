"""Saf Python decoder-only transformer (GPT tarzi) + cross-entropy gradyanlari."""

import math
import random

from config import Config
from layers import Block, Embedding, LayerNorm, Linear
from ops import scale_inplace, zeros


def cross_entropy(logits, targets):
    """(ortalama loss, dloss/dlogits) dondurur."""
    n = len(targets)
    probs = []
    total = 0.0
    for row, t in zip(logits, targets):
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        p = [v / s for v in e]
        total -= math.log(max(p[t], 1e-12))
        probs.append(p)
    dlogits = [row[:] for row in probs]
    for i, t in enumerate(targets):
        dlogits[i][t] -= 1.0
    scale_inplace(dlogits, 1.0 / n)
    return total / n, dlogits


class TransformerLM:
    def __init__(self, cfg: Config, rng: random.Random = None):
        self.cfg = cfg
        rng = rng or random.Random(cfg.seed)
        d = cfg.d_model
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        self.tie_weights = bool(getattr(cfg, "tie_weights", False))
        self.use_rope = bool(getattr(cfg, "use_rope", False))
        self.wte = Embedding(cfg.vocab_size, d, rng, std=0.02)
        self.wpe = Embedding(cfg.block_size, d, rng, std=0.01)
        self.blocks = [
            Block(d, cfg.n_heads, cfg.ffn_hidden, rng, resid_std=resid_std, dropout=getattr(cfg, "dropout", 0.0), use_rope=self.use_rope)
            for _ in range(cfg.n_layers)
        ]
        self.ln_f = LayerNorm(d)
        self.head = Linear(d, cfg.vocab_size, rng, bias=False, std=0.02)
        if self.tie_weights:
            self.head.W = [list(col) for col in zip(*self.wte.table)]
            self.head.gW = zeros(d, cfg.vocab_size)
        self._ids = None
        self._dlogits = None

    def forward(self, ids, targets=None, cache=None, position_offset=0):
        self._ids = ids
        self._dlogits = None
        tok = self.wte.forward(ids)
        pos = self.wpe.forward(list(range(position_offset, position_offset + len(ids))))
        x = [[a + b for a, b in zip(trow, prow)] for trow, prow in zip(tok, pos)]
        for blk in self.blocks:
            x = blk.forward(x, cache=cache, position_offset=position_offset if cache is not None else 0)
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)
        loss = None
        if targets is not None:
            loss, self._dlogits = cross_entropy(logits, targets)
        return loss, logits

    def backward(self):
        assert self._dlogits is not None, "once forward(ids, targets) cagirin"
        dx = self.head.backward(self._dlogits)
        if self.tie_weights:
            for tid, row in enumerate(self.wte.gtable):
                for j in range(len(row)):
                    row[j] += self.head.gW[j][tid]
            self.head.gW = zeros(self.cfg.d_model, self.cfg.vocab_size)
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        for i, tid in enumerate(self._ids):
            grow = self.wte.gtable[tid]
            prow = self.wpe.gtable[i]
            row = dx[i]
            for j, v in enumerate(row):
                grow[j] += v
                prow[j] += v
        self._dlogits = None

    def parameters(self):
        yield from self.wte.parameters("wte.")
        yield from self.wpe.parameters("wpe.")
        for i, blk in enumerate(self.blocks):
            yield from blk.parameters(f"blk{i}.")
        yield from self.ln_f.parameters("ln_f.")
        if not self.tie_weights:
            yield from self.head.parameters("head.")

    def zero_grad(self):
        for _, _, g in self.parameters():
            if g and isinstance(g[0], list):
                for row in g:
                    for j in range(len(row)):
                        row[j] = 0.0
            else:
                for j in range(len(g)):
                    g[j] = 0.0

    def state_dict(self):
        out = {}
        for name, p, _ in self.parameters():
            out[name] = p
        return out

    def load_state_dict(self, state):
        for name, p, _ in self.parameters():
            if name not in state:
                continue
            src = state[name]
            if p and isinstance(p[0], list):
                for row, srow in zip(p, src):
                    row[:] = srow
            else:
                p[:] = src

    def num_params(self):
        total = 0
        for _, p, _ in self.parameters():
            if p and isinstance(p[0], list):
                total += len(p) * len(p[0])
            else:
                total += len(p)
        return total
