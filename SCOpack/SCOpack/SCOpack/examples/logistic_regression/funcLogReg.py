import numpy as np
from typing import Tuple, Optional


def funcLogReg(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], data) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Python port of funcLogReg.m
    Logistic loss: (1/m) sum log(1+exp(Ax)) - b' Ax + (mu/2)||x||^2, with mu=1e-6/m
    Expects data.A (m x n), data.b (m,)
    Returns (obj, grad) for key=='fg', and (H_T1T1, H_T1T2) for key=='h'.
    """
    A = data.A
    b = data.b.reshape(-1)
    x = np.asarray(x).reshape(-1)
    m = b.shape[0]

    # Efficient Ax using sparsity pattern of x
    if np.count_nonzero(x) >= 0.8 * x.size:
        Ax = A @ x
    else:
        Tx = np.flatnonzero(x)
        Ax = A[:, Tx] @ x[Tx]

    eAx = np.exp(Ax)
    mu = 1e-6 / m
    if key == 'fg':
        if not np.isfinite(eAx).all():
            Tpos = np.flatnonzero(Ax > 300)
            Tneg = np.setdiff1d(np.arange(m), Tpos)
            obj = np.sum(np.log1p(eAx[Tneg])) + np.sum(Ax[Tpos]) - np.dot(b, Ax)
        else:
            obj = np.sum(np.log1p(eAx) - b * Ax)
        obj = obj / m
        grad = (A.T @ (1 - b - 1.0 / (1.0 + eAx))) / m + mu * x
        return float(obj), grad
    elif key == 'h':
        eXx = 1.0 / (1.0 + eAx)
        d = eXx * (1.0 - eXx) / m  # length m
        T1 = np.asarray(T1, dtype=int)
        XT = A[:, T1]
        s = T1.size
        if s < 1000:
            H11 = (XT * d[:, None]).T @ XT + mu * np.eye(s)
        else:
            def H11(v):
                return mu * v + (XT.T @ (d * (XT @ v)))
        if T2 is None:
            return H11
        T2 = np.asarray(T2, dtype=int)
        def H12(v):
            return (XT.T @ (d * (A[:, T2] @ v)))
        if isinstance(H11, np.ndarray):
            return H11, H12
        else:
            return H11, H12
    else:
        raise ValueError("key must be 'fg' or 'h'")
