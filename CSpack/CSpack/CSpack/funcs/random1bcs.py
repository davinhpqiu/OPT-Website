from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def random1bcs(
    kind: str,
    m: int,
    n: int,
    s: int,
    r: float,
    noise_factor: float = 0.05,
    corr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic one-bit CS data (port of ``random1bcs.m``)."""

    if m is None or n is None or s is None:
        raise ValueError('Inputs are not enough')

    rng = np.random.default_rng()

    if kind == 'Ind':
        X = rng.standard_normal((m, n))
    elif kind == 'Cor':
        idx = np.arange(n)
        cov = corr ** np.abs(idx[:, None] - idx[None, :])
        X = rng.multivariate_normal(np.zeros(n), cov, size=m)
    else:
        raise ValueError("kind must be 'Ind' or 'Cor'")

    xopt, support = _make_sparse_vector(rng, n, s)
    noise = noise_factor * rng.standard_normal(m)
    raw = X[:, support] @ xopt[support] + noise
    y = np.sign(raw)
    y[y == 0] = 1.0
    yf = _flip_signs(rng, y, r)

    _report_generation(m, n, s, r, noise_factor)
    return X, yf, y, xopt


def _make_sparse_vector(rng: np.random.Generator, n: int, s: int):
    support = rng.permutation(n)[:s]
    x = np.zeros(n)
    coeffs = (0.5 + rng.random(s)) * np.sign(rng.standard_normal(s))
    x[support] = coeffs
    norm = np.linalg.norm(x[support])
    if norm > 0:
        x[support] /= norm
    return x, support


def _flip_signs(rng: np.random.Generator, y: np.ndarray, ratio: float) -> np.ndarray:
    yf = y.copy()
    k = int(math.ceil(max(0.0, ratio) * y.size))
    if k == 0:
        return yf
    idx = rng.permutation(y.size)[:k]
    yf[idx] *= -1
    return yf


def _report_generation(m: int, n: int, s: int, r: float, nf: float) -> None:
    print(' Done generation of sample data with:')
    print(f' 1) Sample size: {m} x {n}')
    print(f' 2) Sparsity level: {s}')
    print(f' 3) Sign flipping ratio: {r:4.2f}')
    print(f' 4) Noise ratio: {nf:4.2f}')
