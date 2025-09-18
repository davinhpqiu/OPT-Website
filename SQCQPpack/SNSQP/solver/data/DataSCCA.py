from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def generate_scca_data(
    nx: int,
    ny: int,
    n_samples: int,
    rng: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
) -> Dict[str, np.ndarray]:
    """Python port of ``DataSCCA.m``."""

    state = _as_random_state(rng)

    print(' Data is generating ...')

    v1 = np.zeros(nx)
    v1[: nx // 8] = 1.0
    v1[nx // 8 : nx // 4] = -1.0

    v2 = np.zeros(ny)
    v2[ny - nx // 4 : ny - nx // 8] = 1.0
    v2[ny - nx // 8 :] = -1.0

    u = state.randn(n_samples)
    base_x = v1[:, None] + 0.1 * state.randn(nx, 1)
    base_y = v2[:, None] + 0.1 * state.randn(ny, 1)

    X = base_x @ u[None, :]
    Y = base_y @ u[None, :]

    Xt = X.T
    Yt = Y.T
    a, b, _ = _leading_canonical_vectors(Xt, Yt)

    Qi = [
        np.block(
            [
                [X @ X.T, np.zeros((nx, ny))],
                [np.zeros((ny, nx)), Y @ Y.T],
            ]
        )
    ]

    Q0 = np.block(
        [
            [np.zeros((nx, nx)), -X @ Y.T],
            [-Y @ X.T, np.zeros((ny, ny))],
        ]
    )

    data = {
        'Q0': Q0,
        'q0': np.zeros(nx + ny),
        'Qi': Qi,
        'qi': np.zeros(nx + ny),
        'ci': np.array([-1.0]),
        'x0': np.concatenate([a, b]),
    }

    print(' Done data generation !!!')

    return data


def _leading_canonical_vectors(X: np.ndarray, Y: np.ndarray):
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0]

    Sxx = (Xc.T @ Xc) / max(n - 1, 1)
    Syy = (Yc.T @ Yc) / max(n - 1, 1)
    Sxy = (Xc.T @ Yc) / max(n - 1, 1)

    Sxx_inv_sqrt = _sym_inv_sqrt(Sxx)
    Syy_inv_sqrt = _sym_inv_sqrt(Syy)

    M = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    a = Sxx_inv_sqrt @ U[:, 0]
    b = Syy_inv_sqrt @ Vt.T[:, 0]
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    return a, b, s[0] if s.size else 0.0


def _sym_inv_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-8, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return inv_sqrt


def _as_random_state(
    seed: Optional[Union[int, np.random.Generator, np.random.RandomState]]
) -> np.random.RandomState:
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return np.random.RandomState(seed.integers(0, 2**32 - 1))
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    return np.random.RandomState()
