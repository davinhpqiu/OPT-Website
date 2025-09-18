import numpy as np
from typing import Optional, Tuple


def funcSimpleEx(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray]) -> Tuple[float, Optional[np.ndarray]]:
    """Python port of funcSimpleEx.m."""
    x = np.asarray(x).reshape(-1, 1)
    a = np.sqrt(float(np.sum(x * x)) + 1.0)

    if key == 'fg':
        obj = float(x.T @ np.array([[6, 5], [5, 8]]) @ x + np.array([[1, 9]]) @ x - a)
        grad = 2 * np.array([[6, 5], [5, 8]]) @ x + np.array([[1], [9]]) - x / a
        return obj, grad.ravel()
    elif key == 'h':
        H = 2 * np.array([[6, 5], [5, 8]]) + (x @ x.T - a * np.eye(2)) / (a ** 3)
        T1 = np.asarray(T1, dtype=int)
        H11 = H[np.ix_(T1, T1)]
        if T2 is None:
            return H11
        T2 = np.asarray(T2, dtype=int)
        H12 = H[np.ix_(T1, T2)]
        return H11, H12
    else:
        raise ValueError("key must be 'fg' or 'h'")
