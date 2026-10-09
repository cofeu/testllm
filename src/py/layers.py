"""Transformer katmanlari: her biri kendi forward + backward (gradyan) hesabini yapar."""

import math

from ops import (
    add_bias,
    add_into,
    colsum,
    matmul,
    matmul_nt,
    matmul_t,
    randn,
    softmax_rows,
    split_cols,
    zeros,
)

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2 = 1.0 / _SQRT2
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x * _INV_SQRT2))


def gelu_grad(x):
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT2)) + x * _INV_SQRT2PI * math.exp(-0.5 * x * x)


class Linear:
    def __init__(self, d_in, d_out, rng, bias=True, std=None):
        if std is None:
            std = 1.0 / math.sqrt(d_in)
        self.W = randn(d_in, d_out, rng, std)
        self.b = [0.0] * d_out if bias else None
        self.gW = zeros(d_in, d_out)
        self.gb = [0.0] * d_out if bias else None
        self._x = None

    def forward(self, x):
        self._x = x
        y = matmul(x, self.W)
        if self.b is not None:
            y = add_bias(y, self.b)
        return y

    def backward(self, dy):
        add_into(self.gW, matmul_t(self._x, dy))
        if self.gb is not None:
            add_into(self.gb, colsum(dy))
        return matmul_nt(dy, self.W)

    def parameters(self, prefix):
        yield prefix + "W", self.W, self.gW
        if self.b is not None:
            yield prefix + "b", self.b, self.gb


class Embedding:
    def __init__(self, n_vocab, d, rng, std=0.02):
        self.table = randn(n_vocab, d, rng, std)
        self.gtable = zeros(n_vocab, d)
        self._ids = None

    def forward(self, ids):
        self._ids = ids
        return [self.table[i] for i in ids]

    def backward(self, dy):
        for i, row in zip(self._ids, dy):
            grow = self.gtable[i]
            for j, v in enumerate(row):
                grow[j] += v

    def parameters(self, prefix):
        yield prefix + "table", self.table, self.gtable


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.eps = eps
        self.g = [1.0] * d
        self.b = [0.0] * d
        self.gg = [0.0] * d
        self.gb = [0.0] * d
        self._cache = None

    def forward(self, x):
        d = len(x[0])
        out = []
        xhats = []
        invs = []
        for row in x:
            mu = sum(row) / d
            var = sum((v - mu) ** 2 for v in row) / d
            inv = 1.0 / math.sqrt(var + self.eps)
            xhat = [(v - mu) * inv for v in row]
            xhats.append(xhat)
            invs.append(inv)
            out.append([xh * gv + bv for xh, gv, bv in zip(xhat, self.g, self.b)])
        self._cache = (xhats, invs)
        return out

    def backward(self, dy):
        xhats, invs = self._cache
        d = len(dy[0])
        dx = []
        for dyrow, xhat, inv in zip(dy, xhats, invs):
            for j in range(d):
                self.gg[j] += dyrow[j] * xhat[j]
                self.gb[j] += dyrow[j]
            dxhat = [dyrow[j] * self.g[j] for j in range(d)]
            m1 = sum(dxhat) / d
            m2 = sum(dxhat[j] * xhat[j] for j in range(d)) / d
            dx.append([inv * (dxhat[j] - m1 - xhat[j] * m2) for j in range(d)])
        return dx

    def parameters(self, prefix):
        yield prefix + "g", self.g, self.gg
        yield prefix + "b", self.b, self.gb


class CausalSelfAttention:
    def __init__(self, d_model, n_heads, rng, std=0.02):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q = Linear(d_model, d_model, rng, bias=False, std=std)
        self.k = Linear(d_model, d_model, rng, bias=False, std=std)
        self.v = Linear(d_model, d_model, rng, bias=False, std=std)
        self.o = Linear(d_model, d_model, rng, bias=False, std=std)
        self._heads = None

    def forward(self, x):
        n = len(x)
        Q = self.q.forward(x)
        K = self.k.forward(x)
        V = self.v.forward(x)
        hd = self.head_dim
        heads = []
        concat = None
        for h in range(self.n_heads):
            a, b = h * hd, (h + 1) * hd
            Qh = split_cols(Q, a, b)
            Kh = split_cols(K, a, b)
            Vh = split_cols(V, a, b)
            S = matmul_nt(Qh, Kh)
            for i in range(n):
                row = S[i]
                for j in range(i + 1, n):
                    row[j] = -1e30
                for j in range(i + 1):
                    row[j] *= self.scale
            P = softmax_rows(S)
            O = matmul(P, Vh)
            heads.append((Qh, Kh, Vh, P))
            if concat is None:
                concat = [list(r) for r in O]
            else:
                for crow, orow in zip(concat, O):
                    crow.extend(orow)
        self._heads = heads
        return self.o.forward(concat)

    def backward(self, dout):
        n = len(dout)
        hd = self.head_dim
        dconcat = self.o.backward(dout)
        dQ = zeros(n, self.d_model)
        dK = zeros(n, self.d_model)
        dV = zeros(n, self.d_model)
        for h in range(self.n_heads):
            a, b = h * hd, (h + 1) * hd
            Qh, Kh, Vh, P = self._heads[h]
            dOh = split_cols(dconcat, a, b)
            dP = matmul_nt(dOh, Vh)
            dVh = matmul_t(P, dOh)
            dS = zeros(n, n)
            for i in range(n):
                rowsum = sum(dP[i][j] * P[i][j] for j in range(n))
                dSrow = dS[i]
                Prow = P[i]
                dProw = dP[i]
                for j in range(n):
                    dSrow[j] = Prow[j] * (dProw[j] - rowsum)
            dQh = matmul(dS, Kh)
            dKh = matmul_t(dS, Qh)
            for i in range(n):
                dQhrow = dQh[i]
                dKhrow = dKh[i]
                dVhrow = dVh[i]
                dQrow = dQ[i]
                dKrow = dK[i]
                dVrow = dV[i]
                for j in range(hd):
                    dQrow[a + j] += dQhrow[j] * self.scale
                    dKrow[a + j] += dKhrow[j] * self.scale
                    dVrow[a + j] += dVhrow[j]
        dx = self.q.backward(dQ)
        add_into(dx, self.k.backward(dK))
        add_into(dx, self.v.backward(dV))
        return dx

    def parameters(self, prefix):
        yield from self.q.parameters(prefix + "q.")
        yield from self.k.parameters(prefix + "k.")
        yield from self.v.parameters(prefix + "v.")
        yield from self.o.parameters(prefix + "o.")


class MLP:
    def __init__(self, d_model, hidden, rng, std=0.02):
        self.fc = Linear(d_model, hidden, rng, std=std)
        self.proj = Linear(hidden, d_model, rng, std=std)
        self._pre = None

    def forward(self, x):
        pre = self.fc.forward(x)
        self._pre = pre
        h = [[gelu(v) for v in row] for row in pre]
        return self.proj.forward(h)

    def backward(self, dout):
        dh = self.proj.backward(dout)
        pre = self._pre
        dpre = [
            [dh[i][j] * gelu_grad(pre[i][j]) for j in range(len(dh[i]))]
            for i in range(len(dh))
        ]
        return self.fc.backward(dpre)

    def parameters(self, prefix):
        yield from self.fc.parameters(prefix + "fc.")
        yield from self.proj.parameters(prefix + "proj.")


class Block:
    def __init__(self, d_model, n_heads, ffn_hidden, rng, resid_std=0.02):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rng, std=resid_std)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, ffn_hidden, rng, std=resid_std)

    def forward(self, x):
        a = self.attn.forward(self.ln1.forward(x))
        x1 = [[av + xv for av, xv in zip(arow, xrow)] for arow, xrow in zip(a, x)]
        b = self.mlp.forward(self.ln2.forward(x1))
        return [[bv + x1v for bv, x1v in zip(brow, x1row)] for brow, x1row in zip(b, x1)]

    def backward(self, dout):
        # y = x1 + mlp(ln2(x1))
        d_ln2_out = self.mlp.backward(dout)
        d_x1_from_mlp = self.ln2.backward(d_ln2_out)
        dx1 = [[a + b for a, b in zip(drow, mrow)] for drow, mrow in zip(dout, d_x1_from_mlp)]
        # x1 = x + attn(ln1(x))
        d_ln1_out = self.attn.backward(dx1)
        d_x_from_attn = self.ln1.backward(d_ln1_out)
        return [[a + b for a, b in zip(x1row, arow)] for x1row, arow in zip(dx1, d_x_from_attn)]

    def parameters(self, prefix):
        yield from self.ln1.parameters(prefix + "ln1.")
        yield from self.attn.parameters(prefix + "attn.")
        yield from self.ln2.parameters(prefix + "ln2.")
        yield from self.mlp.parameters(prefix + "mlp.")
