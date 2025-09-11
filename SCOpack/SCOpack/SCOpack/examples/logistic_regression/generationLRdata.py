import numpy as np
from typing import Dict, Tuple, Optional


def LogitRegdata(_type: str, m: int, n: int, s: int, rho: Optional[float] = None) -> Tuple[Dict, Dict]:
    """
    Python port of generationLRdata.m (LogitRegdata)
    _type in {'Indipendent','Correlated','Weakly-Indipendent'}
    Returns (data, out) where data has A, b, At, and optionally x
    """
    print(' Please wait for LogitReg data generation ...')
    if _type == 'Indipendent':
        I0 = np.random.permutation(m)
        b = np.ones((m,))
        I = I0[: int(np.ceil(m / 2))]
        b[I] = 0
        A = (b * np.random.rand(m)).reshape(-1, 1) + np.random.randn(m, n)
        x = np.array([])
        out = {}
    elif _type == 'Correlated':
        if rho is None:
            raise ValueError('rho must be provided for Correlated type')
        I0 = np.random.permutation(n)
        x = np.zeros((n,))
        I = I0[:s]
        x[I] = np.random.randn(s)
        v = np.random.randn(m, n)
        A = np.zeros((m, n))
        A[:, 0] = np.random.randn(m)
        for j in range(n - 1):
            A[:, j + 1] = rho * A[:, j] + np.sqrt(1 - rho ** 2) * v[:, j]
        Ax = A[:, I] @ x[I]
        q = 1.0 / (1.0 + np.exp(-Ax))
        b = np.zeros((m,))
        for i in range(m):
            b[i] = np.random.choice([0, 1], p=[1 - q[i], q[i]])
        out = {
            'ser': float(np.sum(np.abs(b - np.maximum(0, np.sign(Ax)))) / m),
            'f': float(np.sum(np.log1p(np.exp(Ax)) - b * Ax) / m),
        }
    elif _type == 'Weakly-Indipendent':
        I0 = np.random.permutation(n)
        x = np.zeros((n,))
        I = I0[:s]
        x[I] = np.random.randn(s)
        A = np.random.randn(m, n)
        Ax = A[:, I] @ x[I]
        q = 1.0 / (1.0 + np.exp(-Ax))
        b = np.zeros((m,))
        for i in range(m):
            b[i] = np.random.choice([0, 1], p=[1 - q[i], q[i]])
        out = {
            'ser': float(np.sum(np.abs(b - np.maximum(0, np.sign(Ax)))) / m),
            'f': float(np.sum(np.log1p(np.exp(Ax)) - b * Ax) / m),
        }
    else:
        raise ValueError("_type must be 'Indipendent','Correlated','Weakly-Indipendent'")

    data = {
        'A': A,
        'b': b,
        'At': A.T,
        'x': x,
    }
    return data, out

