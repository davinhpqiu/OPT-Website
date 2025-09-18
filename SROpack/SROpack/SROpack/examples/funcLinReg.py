import numpy as np
from typing import Optional, Tuple


def funcLinReg(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], A: np.ndarray, b: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """Least squares objective 0.5*||Ax - b||^2 with Hessian sub-blocks."""
    x = np.asarray(x).reshape(-1)
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1)

    if key == 'fg':
        Tx = np.flatnonzero(x)
        if Tx.size == x.size:
            Axb = A @ x - b
        else:
            Axb = A[:, Tx] @ x[Tx] - b
        obj = 0.5 * float(Axb.T @ Axb)
        grad = A.T @ Axb
        return obj, grad

    if key == 'h':
        T1 = np.asarray(T1, dtype=int)
        AT = A[:, T1]
        H11 = AT.T @ AT
        if T2 is None:
            return H11
        T2 = np.asarray(T2, dtype=int)
        H12 = AT.T @ A[:, T2]
        return H11, H12

    raise ValueError("key must be 'fg' or 'h'")
