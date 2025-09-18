from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def generate_recovery_data(
    n: int,
    k: int,
    m: int,
    xopt: ArrayLike,
    support: Sequence[int],
    rng: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
) -> Dict[str, np.ndarray]:
    """Python port of ``DataRecovery.m`` with matching console output."""

    state = _as_random_state(rng)

    print(' Data is generating ...')

    xopt = np.asarray(xopt, dtype=float).reshape(n)
    T = np.asarray(support, dtype=int)

    ell = int(np.ceil(n / 4))
    Qi = []
    qi = state.randn(n, k) if k else np.zeros((n, 0))
    ci = np.zeros(k)

    B = state.randn(n + 5, n)
    d = B[:, T] @ xopt[T]
    Q0 = B.T @ B
    q0 = -(B.T @ d)

    half_k = int(np.ceil(k / 2))
    for i in range(k):
        Qii = state.randn(ell, n)
        Qii = Qii.T @ Qii + 0.01 * np.eye(n)
        Qi.append(Qii)
        shift = 0.5 * xopt[T] @ Qii[np.ix_(T, T)] @ xopt[T] + qi[T, i] @ xopt[T]
        ci[i] = -shift - (state.rand() if i < half_k else 0.0)

    if m:
        A = state.randn(m, n)
        bn = state.rand(m)
        idx = min(m - 1, int(np.ceil(m / 2)) - 1)
        bn[idx] = 0.0
    else:
        A = np.zeros((0, n))
        bn = np.zeros(0)
    b = A @ xopt + bn

    if Q0.size:
        lambda_max = max(np.linalg.eigvalsh(Q0).max(), 1e-12)
    else:
        lambda_max = 1.0

    print(' Done data generation !!!')

    data = {
        'Q0': Q0 / lambda_max,
        'q0': q0 / lambda_max,
        'Qi': Qi,
        'qi': qi,
        'ci': ci,
        'A': A,
        'b': b,
    }
    return data


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
