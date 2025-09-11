import numpy as np
from typing import Tuple, Optional


def funcSimpleEx(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray]):
    """
    Python port of funcSimpleEx.m
      Problem: min x'Qx + c'x - sqrt(x'x + 1), s.t. ||x||_0 <= s (here s=1)
      with Q=[[6,5],[5,8]], c=[1,9].
    """
    Q = np.array([[6.0, 5.0], [5.0, 8.0]])
    c = np.array([1.0, 9.0])
    x = np.asarray(x).reshape(-1)
    a = np.sqrt(np.dot(x, x) + 1.0)
    if key == 'fg':
        out1 = float(x @ (Q @ x) + c @ x - a)
        grad = 2 * (Q @ x) + c - x / a
        return out1, grad
    elif key == 'h':
        H = 2 * Q + (np.outer(x, x) - a * np.eye(2)) / (a**3)
        T1 = np.asarray(T1, dtype=int)
        H11 = H[np.ix_(T1, T1)]
        if T2 is None:
            return H11
        T2 = np.asarray(T2, dtype=int)
        H12 = H[np.ix_(T1, T2)]
        return H11, H12
    else:
        raise ValueError("key must be 'fg' or 'h'")
