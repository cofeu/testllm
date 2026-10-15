"""Sonlu fark (finite difference) ile gradyan dogrulamasi.

Kullanim:  python3 src/py/test_grads.py
"""

import random

from config import Config
from model import TransformerLM
from tokenizer import ByteTokenizer


def finite_diff(p, i, j, get_loss, eps=1e-5):
    is_mat = isinstance(p[0], list)
    if is_mat:
        old = p[i][j]
        p[i][j] = old + eps
        lp = get_loss()
        p[i][j] = old - eps
        lm = get_loss()
        p[i][j] = old
    else:
        old = p[i]
        p[i] = old + eps
        lp = get_loss()
        p[i] = old - eps
        lm = get_loss()
        p[i] = old
    return (lp - lm) / (2.0 * eps)


def main():
    tok = ByteTokenizer()
    cfg = Config(
        vocab_size=tok.vocab_size,
        block_size=8,
        d_model=8,
        n_heads=2,
        n_layers=2,
        ffn_hidden=16,
        seed=7,
    )
    rng = random.Random(7)
    model = TransformerLM(cfg, rng)

    ids = [tok.BOS] + [tok.OFFSET + rng.randrange(256) for _ in range(6)]
    targets = ids[1:] + [tok.EOS]

    loss, _ = model.forward(ids, targets)
    model.backward()

    params = list(model.parameters())
    max_err = 0.0
    worst = None
    checked = 0

    for name, p, g in params:
        is_mat = isinstance(p[0], list)
        if is_mat:
            n_rows, n_cols = len(p), len(p[0])
            picks = [(0, 0), (n_rows - 1, n_cols - 1), (n_rows // 2, n_cols // 3)]
        else:
            picks = [(0,), (len(p) - 1,), (len(p) // 2,)]
        for idx in picks:
            if is_mat:
                i, j = idx
                num = finite_diff(p, i, j, lambda: model.forward(ids, targets)[0])
                ana = g[i][j]
            else:
                (i,) = idx
                num = finite_diff(p, i, None, lambda: model.forward(ids, targets)[0])
                ana = g[i]
            err = abs(num - ana) / max(1e-4, abs(num) + abs(ana))
            checked += 1
            if err > max_err:
                max_err = err
                worst = (name, idx, ana, num, err)

    print(f"loss={loss:.6f} | {checked} gradyan kontrol edildi")
    print(f"maksilil goreli hata: {max_err:.3e}")
    print(f"en kotu: {worst[0]}{worst[1]} analitik={worst[2]:.8f} sayisal={worst[3]:.8f}")
    if max_err > 1e-4:
        print("HATA: gradyanlar uyusmuyor")
        raise SystemExit(1)
    print("OK: tum backward geceleri dogru")


if __name__ == "__main__":
    main()
