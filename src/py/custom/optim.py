"""AdamW + gradyan kliplleme + lr zamanlamasi (tamamen el yapimi)."""

import math

from ops import zeros_like


def is_matrix(p):
    return bool(p) and isinstance(p[0], list)


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {name: zeros_like(p) for name, p, _ in self.params}
        self.v = {name: zeros_like(p) for name, p, _ in self.params}

    def step(self, t):
        b1, b2 = self.b1, self.b2
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for name, p, g in self.params:
            m, v = self.m[name], self.v[name]
            wd = self.weight_decay if is_matrix(p) else 0.0
            if is_matrix(p):
                for i in range(len(p)):
                    prow, grow, mrow, vrow = p[i], g[i], m[i], v[i]
                    for j in range(len(prow)):
                        grad = grow[j]
                        mj = mrow[j] = b1 * mrow[j] + (1.0 - b1) * grad
                        vj = vrow[j] = b2 * vrow[j] + (1.0 - b2) * grad * grad
                        mh = mj / bc1
                        vh = vj / bc2
                        prow[j] -= self.lr * (mh / (math.sqrt(vh) + self.eps) + wd * prow[j])
            else:
                for j in range(len(p)):
                    grad = g[j]
                    mj = m[j] = b1 * m[j] + (1.0 - b1) * grad
                    vj = v[j] = b2 * v[j] + (1.0 - b2) * grad * grad
                    mh = mj / bc1
                    vh = vj / bc2
                    p[j] -= self.lr * mh / (math.sqrt(vh) + self.eps)

    def zero_grad(self):
        for _, _, g in self.params:
            if is_matrix(g):
                for row in g:
                    for j in range(len(row)):
                        row[j] = 0.0
            else:
                for j in range(len(g)):
                    g[j] = 0.0


def grad_norm(params):
    total = 0.0
    for _, _, g in params:
        if is_matrix(g):
            for row in g:
                for v in row:
                    total += v * v
        else:
            for v in g:
                total += v * v
    return math.sqrt(total)


def clip_grad_norm(params, max_norm):
    norm = grad_norm(params)
    if max_norm and norm > max_norm:
        scale_inplace_grads(params, max_norm / (norm + 1e-12))
    return norm


def scale_inplace_grads(params, s):
    for _, _, g in params:
        if is_matrix(g):
            for row in g:
                for j in range(len(row)):
                    row[j] *= s
        else:
            for j in range(len(g)):
                g[j] *= s


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    span = max(1, cfg.max_steps - cfg.warmup)
    prog = min(1.0, (step - cfg.warmup) / span)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cos
