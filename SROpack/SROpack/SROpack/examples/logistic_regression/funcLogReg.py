import numpy as np
from typing import Optional, Tuple


def funcLogReg(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], data) -> Tuple[float, Optional[np.ndarray]]:
    """Python port of funcLogReg.m for logistic regression loss."""
    A = data.A if hasattr(data, 'A') else data['A']
    b = data.b if hasattr(data, 'b') else data['b']
    b = np.asarray(b).reshape(-1)
    x = np.asarray(x).reshape(-1)
    m = b.size

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
        obj /= m
        grad = (A.T @ (1 - b - 1.0 / (1.0 + eAx))) / m + mu * x
        return float(obj), grad

    if key == 'h':
        eXx = 1.0 / (1.0 + eAx)
        d = eXx * (1.0 - eXx) / m
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
            return XT.T @ (d * (A[:, T2] @ v))
        if isinstance(H11, np.ndarray):
            return H11, H12
        return H11, H12

    raise ValueError("key must be 'fg' or 'h'")
