import numpy as np
from time import time
from typing import Dict, Tuple


def LCPdata(example: str, n: int, s: int) -> Dict:
    """Python port of generationLCPdata.m."""
    start = time()
    print(' Please wait for LCP data generation ...')

    if example == 'z-mat':
        M = np.eye(n) - np.ones((n, n)) / n
        q = np.ones(n) / n
        q[0] = 1 / n - 1
        xopt = np.zeros(n)
        xopt[0] = 1
        Mt = M.copy()
        data = {'xopt': xopt}
    elif example == 'sdp':
        Z = np.random.randn(n, int(np.ceil(n / 2)))
        M = Z @ Z.T
        xopt, T = get_sparse_x(n, s)
        Mx = M[:, T] @ xopt[T]
        q = np.abs(Mx)
        q[T] = -Mx[T]
        Mt = M / n
        M = M / n
        q = q / n
        data = {'xopt': xopt}
    elif example == 'sdp-non':
        Z = np.random.randn(n, int(np.ceil(n / 4)))
        M = Z @ Z.T
        _, T = get_sparse_x(n, s)
        q = np.random.rand(n)
        q[T] = -np.random.rand(s)
        Mt = M / n
        M = M / n
        q = q / n
        data = {}
    else:
        raise ValueError("example must be 'z-mat', 'sdp', or 'sdp-non'")

    result = {
        'A': M,
        'At': Mt,
        'b': q,
        'n': n,
    }
    result.update(data)

    print(f' Data generation used {time() - start:.4f} seconds.\n')
    return result


def get_sparse_x(n: int, s: int) -> Tuple[np.ndarray, np.ndarray]:
    I = np.random.permutation(n)
    T = I[:s]
    x = np.zeros(n)
    x[T] = 0.1 + np.abs(np.random.randn(s))
    return x, T
