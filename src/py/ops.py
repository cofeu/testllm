"""Saf Python ile matris/vektor matematigi (numpy, torch yok)."""

import math
from operator import mul


def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def zeros_like(p):
    if p and isinstance(p[0], list):
        return [[0.0] * len(row) for row in p]
    return [0.0] * len(p)


def randn(n, m, rng, std=1.0):
    return [[rng.gauss(0.0, std) for _ in range(m)] for _ in range(n)]


def matmul(a, b):
    """a: n x k, b: k x m -> n x m"""
    bt = list(zip(*b))
    return [[sum(map(mul, arow, bc)) for bc in bt] for arow in a]


def matmul_t(a, b):
    """a.T @ b   (a: n x k, b: n x m -> k x m)"""
    return matmul([list(col) for col in zip(*a)], b)


def matmul_nt(a, b):
    """a @ b.T   (a: n x k, b: m x k -> n x m)"""
    brows = list(b)
    return [[sum(map(mul, arow, brow)) for brow in brows] for arow in a]


def add_bias(x, b):
    return [[v + bv for v, bv in zip(row, b)] for row in x]


def add_into(dst, src):
    """dst += src (in-place, ayni sekilde)."""
    if dst and isinstance(dst[0], list):
        for drow, srow in zip(dst, src):
            for j, v in enumerate(srow):
                drow[j] += v
    else:
        for j, v in enumerate(src):
            dst[j] += v
    return dst


def scale_inplace(p, s):
    if p and isinstance(p[0], list):
        for row in p:
            for j in range(len(row)):
                row[j] *= s
    else:
        for j in range(len(p)):
            p[j] *= s


def colsum(x):
    acc = [0.0] * len(x[0])
    for row in x:
        for j, v in enumerate(row):
            acc[j] += v
    return acc


def split_cols(rows, a, b):
    return [r[a:b] for r in rows]


def softmax_rows(x):
    out = []
    for row in x:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        out.append([v / s for v in e])
    return out
