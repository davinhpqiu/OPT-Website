import numpy as np
from typing import Callable, Optional, Tuple, Union

ArrayLike = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


def funcCS(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], data) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]], Callable[[np.ndarray], np.ndarray]]:
    """Least-squares objective f(x)=0.5*||Ax-b||^2 with optional function-handle A."""
    A = data['A']
    b = data['b']
    x = np.asarray(x).reshape(-1)

    if not callable(A):
        if key == 'fg':
            if np.count_nonzero(x) >= 0.8 * x.size:
                Axb = A @ x - b
            else:
                Tx = np.flatnonzero(x)
                Axb = A[:, Tx] @ x[Tx] - b
            obj = 0.5 * float(Axb.T @ Axb)
            grad = A.T @ Axb
            return obj, grad
        if key == 'h':
            T1 = np.asarray(T1, dtype=int)
            AT = A[:, T1]
            if T1.size <= 1e3 and AT.shape[0] <= 5e3:
                H11 = AT.T @ AT
            else:
                def H11(var):
                    return AT.T @ (AT @ var)
            if T2 is None:
                return H11
            T2 = np.asarray(T2, dtype=int)
            def H12(var):
                return AT.T @ (A[:, T2] @ var)
            if isinstance(H11, np.ndarray):
                return H11, H12
            return H11, H12
        raise ValueError("key must be 'fg' or 'h'")

    if 'At' not in data:
        raise ValueError('Missing transpose operator data["At"] for callable A')
    if 'n' not in data:
        raise ValueError('Missing dimension data["n"] for callable A')

    if key == 'fg':
        Axb = A(x) - b
        obj = 0.5 * float(Axb.T @ Axb)
        grad = data['At'](Axb)
        return obj, grad
    if key == 'h':
        func = _fg_h(data)
        def H11(var):
            return func(var, T1, T1)
        if T2 is None:
            return H11
        def H12(var):
            return func(var, T1, T2)
        return H11, H12
    raise ValueError("key must be 'fg' or 'h'")


def _fg_h(data):
    n = data['n']
    def supp(x, T):
        z = np.zeros(n)
        z[T] = x
        return z

    def Hess(z, t1, t2):
        return data['At'](data['A'](supp(z, t2)))[t1]

    return Hess
