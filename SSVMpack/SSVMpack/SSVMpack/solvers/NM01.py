from __future__ import annotations

import math
import time
from typing import Any, Dict, Tuple

import numpy as np


class NM01Error(ValueError):
    """Custom error class for NM01 solver."""


def NM01(A, y, pars: Dict[str, Any] | None = None):
    if pars is None:
        pars = {}

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size != A.shape[0]:
        print('No enough inputs!!!')
        return {}

    m, n_minus = A.shape
    ny = -y

    ones_col = np.ones((m, 1))
    Any = np.column_stack((ny[:, None] * A, ny))

    Fnorm = lambda v: float(np.linalg.norm(v) ** 2)
    n = n_minus + 1
    maxit, tau, w_pen, mu, disp, acc0, x, cgtol = _get_parameters(m, n, pars)

    z = np.ones(m)
    ACC = np.zeros(maxit)
    if Fnorm(x) == 0:
        Ax = z.copy()
    else:
        Ax = Any @ x + np.ones(m)
    Axz = Ax + tau * z

    H = np.concatenate([np.ones(n - 1), [w_pen]])
    H1 = np.concatenate([np.ones(n - 1), [1.0 / max(w_pen, 1e-12)]])
    maxAcc = 0.0
    maxx = x.copy()
    maxiter = 0
    best_T: np.ndarray | None = None

    lam = float(pars.get('lam', _initial_lambda(m, n, Axz, tau)))
    lam = max(1.0 / (2 * tau), lam)

    if disp:
        print(' \n Start to run the solver -- NM01')
        print(' ------------------------------------------')
        print('  Iter          Accuracy          CPUTime ')
        print(' ------------------------------------------')

    start = time.perf_counter()
    for iter_idx in range(1, maxit + 1):
        T, empT, lam = _Ttau(Axz, Ax, tau, lam)
        g = x.copy()
        g[-1] = w_pen * x[-1]

        if empT:
            raw = 0.0
        else:
            P = Any[T, :]
            zJ = z[T]
            tmp1 = g + P.T @ zJ
            tmp2 = Ax[T]
            raw = Fnorm(tmp2) / m

        sb = np.sign(ny * (Ax - 1))
        sb[sb == 0] = -1
        acc = 1 - np.count_nonzero(sb + ny) / m
        x0 = x.copy()
        if iter_idx > 1 and acc < min(0.5, ACC[iter_idx - 2]):
            acc = 1 - acc
            x0 = -x

        ACC[iter_idx - 1] = acc
        if acc >= maxAcc:
            maxAcc = acc
            maxx = x0.copy()
            maxiter = iter_idx
            best_T = T.copy() if T.size else np.array([], dtype=int)

        if disp:
            elapsed = time.perf_counter() - start
            print(f"  {iter_idx:3d}          {acc:8.5f}          {elapsed:.3f}sec")

        stop1 = iter_idx > 4 and abs(acc - ACC[iter_idx - 2]) <= 5e-5
        stop2 = raw < 1e-1 and maxiter < iter_idx - 4
        stop3 = raw < 1e-8 and acc > 0.5 and iter_idx > 4
        if stop1 or acc >= acc0 - 1e-5 or stop2 or stop3:
            break

        if empT:
            u = -g
            v = -z
        else:
            P = Any[T, :]
            tmp1 = g + P.T @ z[T]
            tmp2 = Ax[T]
            rhs = -mu * tmp1 - P.T @ tmp2
            nT = T.size

            if n > 5_000 or (m <= n and n > 2_000):
                if m > 1_000:
                    fx = lambda var: P.T @ (P @ var) + mu * H * var
                    u = _my_cg(fx, rhs, cgtol, 25, np.zeros(n))
                    v = -z.copy()
                    v[T] = (tmp2 + P @ u) / mu
                else:
                    if nT == 0:
                        vT = np.zeros(0)
                    elif nT > 500:
                        fx = lambda var: P @ (H1[:, None] * (P.T @ var)) + mu * var
                        vT = _my_cg(fx, tmp2 - P @ (H1 * tmp1), cgtol, 25, np.zeros(nT))
                    else:
                        D = P @ (H1[:, None] * P.T)
                        diag = np.arange(nT)
                        D[diag, diag] += mu
                        vT = np.linalg.solve(D, tmp2 - P @ (H1 * tmp1))
                    v = -z.copy()
                    v[T] = vT
                    u = -H1 * (tmp1 + P.T @ vT)
            else:
                if n > 2_000:
                    fx = lambda var: P.T @ (P @ var) + mu * H * var
                    u = _my_cg(fx, rhs, cgtol, 25, np.zeros(n))
                else:
                    D = P.T @ P
                    diag = np.arange(n)
                    D[diag, diag] += mu * H
                    u = np.linalg.solve(D, rhs)
                v = -z.copy()
                v[T] = (tmp2 + P @ u) / mu

        x = x + u
        z = z + v
        Ax = Any @ x + np.ones(m)

        if iter_idx % 5 == 0:
            mu = max(1e-10, mu / 2)
            tau = max(1e-4, tau / 1.5)
            lam *= 1.1
        Axz = Ax + tau * z

    if disp:
        print(' ------------------------------------------')

    x = maxx
    elapsed_total = time.perf_counter() - start
    support = best_T if best_T is not None else np.nonzero(x)[0]
    out = {
        'w': x,
        'x': x,
        'z': z,
        'obj': Fnorm(H * x),
        'time': elapsed_total,
        'Acc': maxAcc,
        'sv': int(support.size),
        'iter': iter_idx,
    }
    return out


def _get_parameters(m: int, n: int, pars: Dict[str, Any]):
    maxit = int(pars.get('maxit', 1_000))
    w = float(pars.get('w', 1e-8))
    disp = int(pars.get('disp', 1))
    acc0 = float(pars.get('acc', 1.0))
    tau = float(pars.get('tau', 5.0))
    mn = max(m, n)
    mu = float(pars.get('mu', 0.001 if mn < 500 else (0.01 if (mn >= 500 and m < n) else 10.0)))
    x0 = np.asarray(pars.get('x0', np.zeros(n)), dtype=float)
    cgtol = float(pars.get('cgtol', 1e-10 * math.sqrt(mn)))
    return maxit, tau, w, mu, disp, acc0, x0, cgtol


def _Ttau(Axz: np.ndarray, Ax: np.ndarray, tau: float, lam: float):
    tl = math.sqrt(tau * lam / 2.0)
    T = np.flatnonzero(np.abs(Axz - tl) < tl)
    empT = T.size == 0
    if empT:
        zp = Ax[Ax >= 0]
        if zp.size:
            s = int(math.ceil(0.01 * zp.size))
            tau = (zp[s - 1]) ** 2 / (2 * lam)
            tl = math.sqrt(tau * lam / 2.0)
            T = np.flatnonzero(np.abs(Ax - tl) < tl)
            empT = T.size == 0
    return T, empT, lam


def _initial_lambda(m: int, n: int, z: np.ndarray, tau: float) -> float:
    zp = z[z > 0]
    if zp.size == 0:
        return 1.0
    s = min(m, 20 * n, zp.size)
    return max(5.0, (zp[s - 1]) ** 2 / (2 * tau))


def _my_cg(fx, b: np.ndarray, cgtol: float, cgit: int, x0: np.ndarray):
    x = x0.copy()
    r = b - (fx @ x if isinstance(fx, np.ndarray) else fx(x))
    e = float(np.dot(r, r))
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        w = fx @ p if isinstance(fx, np.ndarray) else fx(p)
        denom = float(np.dot(p, w))
        a = e / denom if denom != 0 else 0.0
        x = x + a * p
        r = r - a * w
        e0 = e
        e = float(np.dot(r, r))
        if e0 != 0:
            p = r + (e / e0) * p
        else:
            p = r
    return x
