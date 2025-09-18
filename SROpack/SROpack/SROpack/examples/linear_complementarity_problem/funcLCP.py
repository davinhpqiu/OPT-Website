import numpy as np
from typing import Optional, Tuple


def funcLCP(x: np.ndarray, key: str, T1: Optional[np.ndarray], T2: Optional[np.ndarray], data) -> Tuple[float, Optional[np.ndarray]]:
    """Python port of funcLCP.m."""
    if isinstance(data, dict):
        r = data.get('r', 2)
        M = data['A']
        Mt = data['At']
        q = data['b']
    else:
        r = getattr(data, 'r', 2)
        M = data.A
        Mt = data.At
        q = data.b

    x = np.asarray(x).reshape(-1)
    M = np.asarray(M)
    Mt = np.asarray(Mt)
    q = np.asarray(q).reshape(-1)
    n = q.size

    eps = 0.0
    ip = np.flatnonzero(x > eps)
    in_ = np.flatnonzero(x < -eps)
    ix = np.sort(np.unique(np.concatenate((ip, in_))))
    if ix.size:
        Mx = M[:, ix] @ x[ix] + q
    else:
        Mx = q.copy()
    tp = np.flatnonzero(Mx > eps)
    tn = np.flatnonzero(Mx < -eps)
    tt = np.intersect1d(ip, tp)

    com = tt.size > 0
    Mxn = np.abs(Mx[tn])
    xn = np.abs(x[in_])

    if key == 'fg':
        obj = (np.sum(xn ** r) + np.sum(Mxn ** r)) / r
        if com:
            obj += np.sum((x[tt] * Mx[tt]) ** r) / r
        grad = np.zeros(n)
        if tn.size:
            grad -= Mt[:, tn] @ (Mxn ** (r - 1))
        if in_.size:
            grad[in_] -= xn ** (r - 1)
        if com:
            grad += Mt[:, tt] @ ((x[tt] ** r) * (Mx[tt] ** (r - 1)))
            grad[tt] += (x[tt] ** (r - 1)) * (Mx[tt] ** r)
        return float(obj), grad

    if key == 'h':
        T1 = np.asarray(T1, dtype=int)
        s1 = T1.size
        mx = np.maximum(x, 0)
        mMx = np.maximum(Mx, 0)
        M_T1T1 = M[np.ix_(T1, T1)]
        MM = np.zeros((s1, s1))
        if r != 2:
            r1 = r - 1
            r2 = r - 2
            if tn.size:
                z2 = r1 * (Mxn ** r2)
                MM += Mt[np.ix_(T1, tn)] @ (z2[:, None] * M[np.ix_(tn, T1)])
            if com and tt.size:
                z1 = r1 * (mx[tt] ** r) * (mMx[tt] ** r2)
                MM += Mt[np.ix_(T1, tt)] @ (z1[:, None] * M[np.ix_(tt, T1)])
            xy = r1 * ((mx[T1] ** r2) * (mMx[T1] ** r) + (np.maximum(-x[T1], 0) ** r2))
        else:
            z = np.ones(n)
            z[ip] = mMx[ip] ** 2
            xy = z[T1]
            tn0 = np.setdiff1d(np.arange(n), tp)
            if tn0.size:
                MM += Mt[np.ix_(T1, tn0)] @ M[np.ix_(tn0, T1)]
            if com and tt.size:
                z1 = (mx[tt] ** r)
                MM += Mt[np.ix_(T1, tt)] @ (z1[:, None] * M[np.ix_(tt, T1)])

        tem1 = r * np.power(mx[T1] * mMx[T1], r - 1)
        H11 = tem1[:, None] * M_T1T1 + (tem1[None, :] * M_T1T1.T) + MM
        H11[np.arange(s1), np.arange(s1)] += xy

        if T2 is None:
            return H11

        T2 = np.asarray(T2, dtype=int)
        s2 = T2.size
        if r != 2:
            MM12 = np.zeros((s1, s2))
            if tn.size:
                z2 = (r - 1) * (Mxn ** (r - 2))
                MM12 += Mt[np.ix_(T1, tn)] @ (z2[:, None] * M[np.ix_(tn, T2)])
            if com and tt.size:
                z1 = (r - 1) * (mx[tt] ** r) * (mMx[tt] ** (r - 2))
                MM12 += Mt[np.ix_(T1, tt)] @ (z1[:, None] * M[np.ix_(tt, T2)])
        else:
            tn0 = np.setdiff1d(np.arange(n), tp)
            MM12 = np.zeros((s1, s2))
            if tn0.size:
                MM12 += Mt[np.ix_(T1, tn0)] @ M[np.ix_(tn0, T2)]
            if com and tt.size:
                z1 = (mx[tt] ** r)
                MM12 += Mt[np.ix_(T1, tt)] @ (z1[:, None] * M[np.ix_(tt, T2)])

        tem2 = r * np.power(mx[T2] * mMx[T2], r - 1)
        H12 = tem1[:, None] * M[np.ix_(T1, T2)] + Mt[np.ix_(T1, T2)] * tem2[None, :] + MM12
        return H11, H12

    raise ValueError("key must be 'fg' or 'h'")
