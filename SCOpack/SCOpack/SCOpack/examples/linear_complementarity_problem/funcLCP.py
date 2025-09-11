import numpy as np
from typing import Optional, Tuple


def _as_array(x):
    return np.asarray(x).reshape(-1)


def _get(field, data):
    # supports dict or simple object with attributes
    if isinstance(data, dict):
        return data[field]
    return getattr(data, field)


def funcLCP(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], data) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Python port of funcLCP.m
    Implements the smoothed merit function for LCP as in the MATLAB code.

    Inputs:
      - x: current iterate (n,)
      - key: 'fg' or 'h'
      - T1, T2: index arrays for sub-Hessian blocks (when key=='h')
      - data: dict or object with fields A (M), At (M^T), b (q), optional r

    Returns:
      - For key=='fg': (obj, grad)
      - For key=='h': (H_T1T1, H_T1T2)
    """
    r = _get('r', data) if hasattr(data, 'r') or (isinstance(data, dict) and 'r' in data) else 2
    M = _get('A', data)
    Mt = _get('At', data)
    q = _as_array(_get('b', data))
    n = q.shape[0]

    x = _as_array(x)

    eps = 0.0
    ip = np.flatnonzero(x > eps)
    ineg = np.flatnonzero(x < -eps)
    ix = np.union1d(ip, ineg).astype(int)
    
    if ix.size > 0:
        Mx = M[:, ix] @ x[ix] + q
    else:
        Mx = q.copy()
    tp = np.flatnonzero(Mx > eps)
    tn = np.flatnonzero(Mx < -eps)
    tt = np.intersect1d(ip, tp)
    com = tt.size > 0
    Mxn = np.abs(Mx[tn])
    xn = np.abs(x[ineg])

    if key == 'fg':
        obj = (np.sum(xn ** r) + np.sum(Mxn ** r)) / r
        if com:
            obj += np.sum((x[tt] * Mx[tt]) ** r) / r
        # Gradient
        grad = np.zeros((n,))
        if tn.size > 0:
            grad += -(Mt[:, tn] @ (Mxn ** (r - 1)))
        if ineg.size > 0:
            grad[ineg] -= xn ** (r - 1)
        if com and tt.size > 0:
            grad += Mt[:, tt] @ ( (x[tt] ** r) * (Mx[tt] ** (r - 1)) )
            grad[tt] += (x[tt] ** (r - 1)) * (Mx[tt] ** r)
        return float(obj), grad

    elif key == 'h':
        T1 = np.asarray(T1, dtype=int)
        s1 = T1.size
        mx = np.maximum(x, 0.0)
        mMx = np.maximum(Mx, 0.0)

        if r != 2:
            r1 = r - 1
            r2 = r - 2
            z2 = r1 * (Mxn ** r2)  # len tn
            neg_part_T1 = np.maximum(-x[T1], 0.0)
            xy = r1 * ((mx[T1] ** r2) * (mMx[T1] ** r) + (neg_part_T1 ** r2))
            if tn.size > 0:
                MM = Mt[np.ix_(T1, tn)] @ ( (z2[:, None]) * M[np.ix_(tn, T1)] )
            else:
                MM = np.zeros((s1, s1))
            if com and tt.size > 0:
                z1 = r1 * ( (mx[tt] ** r) * (mMx[tt] ** r2) )
                MM = MM + Mt[np.ix_(T1, tt)] @ ( (z1[:, None]) * M[np.ix_(tt, T1)] )
        else:
            z = np.ones((n,))
            if ip.size > 0:
                z[ip] = (mMx[ip] ** 2)
            xy = z[T1]
            tn0 = np.setdiff1d(np.arange(n), tp)
            if tn0.size > 0:
                MM = Mt[np.ix_(T1, tn0)] @ M[np.ix_(tn0, T1)]
            else:
                MM = np.zeros((s1, s1))
            if com and tt.size > 0:
                z1 = (mx[tt] ** r)
                MM = MM + Mt[np.ix_(T1, tt)] @ ( (z1[:, None]) * M[np.ix_(tt, T1)] )

        tem1 = r * ( (mx[T1] * mMx[T1]) ** (r - 1) ) if s1 > 0 else np.array([])
        H11 = (tem1[:, None] * M[np.ix_(T1, T1)]) if s1 > 0 else np.zeros((0, 0))
        H11 = H11 + H11.T + MM
        if s1 > 0:
            H11[np.arange(s1), np.arange(s1)] += xy

        if T2 is None:
            return H11

        T2 = np.asarray(T2, dtype=int)
        s2 = T2.size
        if r != 2:
            if tn.size > 0 and s2 > 0:
                MM12 = Mt[np.ix_(T1, tn)] @ ( (z2[:, None]) * M[np.ix_(tn, T2)] )
            else:
                MM12 = np.zeros((s1, s2))
        else:
            tn0 = np.setdiff1d(np.arange(n), tp)
            if tn0.size > 0 and s2 > 0:
                MM12 = Mt[np.ix_(T1, tn0)] @ M[np.ix_(tn0, T2)]
            else:
                MM12 = np.zeros((s1, s2))
        if com and tt.size > 0 and s2 > 0:
            if r != 2:
                z1 = r1 * ( (mx[tt] ** r) * (mMx[tt] ** r2) )
            else:
                z1 = (mx[tt] ** r)
            MM12 = MM12 + Mt[np.ix_(T1, tt)] @ ( (z1[:, None]) * M[np.ix_(tt, T2)] )

        tem2 = r * ( (mx[T2] * mMx[T2]) ** (r - 1) ) if s2 > 0 else np.array([])
        H12 = (tem1[:, None] * M[np.ix_(T1, T2)]) if s1 * s2 > 0 else np.zeros((s1, s2))
        if s1 * s2 > 0:
            H12 = H12 + (Mt[np.ix_(T1, T2)] * tem2[None, :]) + MM12
        return H11, H12

    else:
        raise ValueError("key must be 'fg' or 'h'")
