import numpy as np
from typing import Tuple, Optional


def funcLinReg(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Python port of funcLinReg.m
      Problem: min 0.5*||A x - b||^2 subject to ||x||_0 <= s
    Returns (out1,) for key=='fg' or 'h'; and (out1, out2) if gradient/sub-Hessian also requested.
    """
    x = np.asarray(x).reshape(-1)
    if key == 'fg':
        Tx = np.flatnonzero(x)
        Axb = (A[:, Tx] @ x[Tx]) - b
        out1 = 0.5 * float(Axb.T @ Axb)
        # Gradient if needed
        grad = A.T @ Axb
        return out1, grad
    elif key == 'h':
        T1 = np.asarray(T1, dtype=int)
        AT = A[:, T1]
        H11 = AT.T @ AT
        if T2 is None:
            return H11
        T2 = np.asarray(T2, dtype=int)
        H12 = AT.T @ A[:, T2]
        return H11, H12
    else:
        raise ValueError("key must be 'fg' or 'h'")
