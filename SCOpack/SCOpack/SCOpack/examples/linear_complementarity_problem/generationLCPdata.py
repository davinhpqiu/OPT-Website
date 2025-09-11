import numpy as np
from typing import Dict, Tuple


def LCPdata(example: str, n: int, s: int) -> Dict:
    """
    Python port of generationLCPdata.m (function LCPdata)
    example in {'z-mat','sdp','sdp-non'}
    Returns dict with keys: A, At, b, n, (optional) xopt
    """
    print(' Please wait for LCP data generation ...')
    if example == 'z-mat':
        M = np.eye(n) - np.ones((n, n)) / n
        q = np.ones((n, 1)) / n
        q[0, 0] = 1.0 / n - 1.0
        xopt = np.zeros((n, 1))
        xopt[0, 0] = 1.0
        Mt = M.copy()
        data = {"A": M, "At": Mt, "b": q, "n": n, "xopt": xopt}
        return data
    elif example == 'sdp':
        Z = np.random.randn(n, int(np.ceil(n / 2)))
        M = Z @ Z.T
        xopt, T = _get_sparse_x(n, s)
        Mx = M[:, T] @ xopt[T]
        q = np.abs(Mx)
        q[T] = -Mx[T]
        Mt = M / n
        M = M / n
        q = q / n
        data = {"A": M, "At": Mt, "b": q, "n": n, "xopt": xopt}
        return data
    elif example == 'sdp-non':
        Z = np.random.rand(n, int(np.ceil(n / 4)))
        M = Z @ Z.T
        _, T = _get_sparse_x(n, s)
        q = np.random.rand(n, 1)
        q[T] = -np.random.rand(s, 1)
        Mt = M / n
        M = M / n
        q = q / n
        data = {"A": M, "At": Mt, "b": q, "n": n}
        return data
    else:
        raise ValueError("example must be one of 'z-mat', 'sdp', 'sdp-non'")


def generationLCPdata(example: str, n: int, s: int) -> Dict:
    """Alias to match MATLAB call sites."""
    return LCPdata(example, n, s)


def _get_sparse_x(n: int, s: int) -> Tuple[np.ndarray, np.ndarray]:
    I = np.random.permutation(n)
    T = I[:s]
    x = np.zeros((n, 1))
    x[T] = 0.1 + np.abs(np.random.randn(s, 1))
    return x, T

