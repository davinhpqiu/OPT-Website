from __future__ import annotations

import numpy as np


def randomData(kind: str, m: int, n: int, r: float):
    """Python translation of ``randomData.m``."""

    rng = np.random.default_rng()
    m2 = int(np.ceil(m / 2))

    if kind == '2D':
        data1 = np.column_stack((0.5 + np.sqrt(0.5) * rng.standard_normal(m2),
                                 -3 + np.sqrt(3) * rng.standard_normal(m2)))
        data2 = np.column_stack((-0.5 + np.sqrt(0.5) * rng.standard_normal(m2),
                                 3 + np.sqrt(3) * rng.standard_normal(m2)))
        A = np.vstack((data1, data2))
        c = np.concatenate((-np.ones(m2), np.ones(m2)))
    elif kind == '3D':
        rho = 0.5 + 0.03 * rng.standard_normal(m2)
        t = 2 * np.pi * rng.random(m2)
        data1 = np.column_stack((rho * np.cos(t), rho * np.sin(t), rho * rho))
        rho = 0.5 + 0.03 * rng.standard_normal(m2)
        t = 2 * np.pi * rng.random(m2)
        data2 = np.column_stack((rho * np.cos(t), rho * np.sin(t), -rho * rho))
        A = np.vstack((data1, data2))
        c = np.concatenate((np.ones(m2), -np.ones(m2)))
    elif kind == 'nD':
        c = np.ones(m)
        idx = rng.permutation(m)[:m2]
        c[idx] = -1
        A = np.repeat((c * rng.random(m))[:, None], n, axis=1) + rng.standard_normal((m, n))
    else:
        raise ValueError("kind must be one of '2D', '3D', 'nD'")

    perm = rng.permutation(m)
    Atr = A[perm[:m2], :]
    ctr = c[perm[:m2]]
    ctr = _flip(ctr, r, rng)

    Ate = A[perm[m2:], :]
    cte = c[perm[m2:]]
    return Atr, ctr, Ate, cte


def _flip(vec: np.ndarray, ratio: float, rng: np.random.Generator):
    if ratio <= 0:
        return vec
    mc = vec.size
    idx = rng.permutation(mc)[: int(np.ceil(ratio * mc))]
    flipped = vec.copy()
    flipped[idx] *= -1
    return flipped
