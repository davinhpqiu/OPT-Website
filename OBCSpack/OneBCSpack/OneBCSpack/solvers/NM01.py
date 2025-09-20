from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np


def NM01(A, b, sp, pars: Dict | None = None):
    """Python translation of ``NM01.m``."""

    if pars is None:
        pars = {}

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if sp is None:
        raise ValueError('Inputs is not enough !!!')

    m, n = A.shape
    if b.size != m:
        raise ValueError('Dimension mismatch between A and b')

    negb = -b
    Ab = (negb[:, None]) * A if n < 1e4 else (negb[:, None]) * A

    (maxit, lam, tau, mu, epsilon, vareps, cgtol, x, disp) = _get_parameters(m, n, pars)
    lam = max(lam, epsilon * epsilon / (2 * tau))
    grad, hess = _def_func(pars)
    z = np.ones(m)
    if np.linalg.norm(x) == 0:
        Ax = epsilon * np.ones(m)
    else:
        Ax = Ab @ x + epsilon
    Axz = Ax.copy()
    ACC = np.zeros(maxit)
    maxAcc = 0
    maxx = x.copy()

    t0 = time.perf_counter()
    if disp:
        print('\n Start to run the solver: NM01')
        print('------------------------------------------')
        print('  Iter           HamDist          CPUTime ')
        print('------------------------------------------')

    for it in range(1, maxit + 1):
        T, empT = _Ttau(Axz, Ax, tau, lam)
        g = grad(x, vareps)

        if empT:
            tmp1 = g
            raw = 0.0
            err = _fnorm2(g) + raw + _fnorm2(z)
        else:
            P = Ab[T, :]
            zT = z[T]
            tmp1 = g + P.T @ zT
            tmp2 = Ax[T]
            raw = _fnorm2(tmp2)
            err = _fnorm2(tmp1) + raw + (_fnorm2(z) - _fnorm2(zT))

        sb = np.sign(negb * (Ax - epsilon))
        sb[sb == 0] = -1
        acc = 1 - np.count_nonzero(sb + negb) / m
        x0 = x.copy()
        if it > 1 and acc < min(0.5, ACC[it - 2]):
            acc = 1 - acc
            x0 = -x

        ACC[it - 1] = acc
        if acc > maxAcc:
            maxAcc = acc
            maxx = x0

        stop1 = it > 5 and acc < ACC[it - 2] and np.min(ACC[max(0, it - 3):it - 1]) == 1
        stop2 = it > 5 and acc > 0.99995 and raw < 1e-5 * math.sqrt(n) and n > 500

        if not stop1 and disp:
            print(f"  {it:3d}          {acc * 100:8.2f}           {time.perf_counter()-t0:.3f}sec")
        if (err < 1e-4 and stop2) or stop1:
            break

        if empT:
            u = -g
            v = -z
        else:
            H = hess(x, vareps)
            if T.size < n:
                H1 = 1.0 / H
                rhs = tmp2 - P @ (H1 * tmp1)
                if T.size < 2000:
                    D = P @ (H1[:, None] * P.T)
                    D[np.diag_indices(T.size)] += mu
                    vT = np.linalg.solve(D, rhs)
                else:
                    fx = lambda var: mu * var + P @ (H1 * (P.T @ var))
                    vT = _my_cg(fx, rhs, cgtol, 30, np.zeros(T.size))
                v = -z.copy()
                v[T] = vT
                u = -H1 * (tmp1 + P.T @ vT)
            else:
                rhs = -mu * tmp1 - P.T @ tmp2
                if n < 2000:
                    D = P.T @ P
                    D[np.diag_indices(n)] += mu * H
                    u = np.linalg.solve(D, rhs)
                else:
                    fx = lambda var: mu * (H * var) + P.T @ (P @ var)
                    u = _my_cg(fx, rhs, cgtol, 50, np.zeros(n))
                v = -z.copy()
                v[T] = (tmp2 + P @ u) / mu
        x = x + u
        z = z + v
        Ax = Ab @ x + epsilon
        Axz = Ax + tau * z
        vareps = max(1e-5, vareps / 2)
        if it % 5 == 0:
            mu = max(1e-10, mu / 2)

    if disp:
        print('------------------------------------------')
    if acc < ACC[0]:
        x = maxx

    if sp:
        out_sol = _refine_with_sparsity(A, b, x, sp)
    else:
        out_sol = _sparse_approx(x)
        norm = np.linalg.norm(out_sol)
        if norm > 0:
            out_sol /= norm

    out = {
        'sol': out_sol,
        'lam': z,
        'time': time.perf_counter() - t0,
        'iter': it,
    }
    return out


def _get_parameters(m: int, n: int, pars: Dict) -> Tuple[int, float, float, float, float, float, float, np.ndarray, int]:
    maxit = int(pars.get('maxit', 1_000))
    lam = float(pars.get('lam', 1.0))
    tau = float(pars.get('tau', 1.0))
    mu = float(pars.get('mu', 0.05))
    epsilon = float(pars.get('epsilon', 0.15))
    vareps = float(pars.get('vareps', 0.5))
    tolcg = float(pars.get('tolcg', 1e-10 * math.sqrt(max(m, n))))
    x0 = np.asarray(pars.get('x0', np.zeros(n)), dtype=float)
    disp = int(pars.get('disp', 1))
    return maxit, lam, tau, mu, epsilon, vareps, tolcg, x0, disp


def _def_func(pars: Dict):
    q = float(pars.get('q', 0.5))
    q1 = q / 2 - 1
    q2 = q / 2 - 2
    q3 = q - 1

    def grad(t, e):
        return t * (t * t + e) ** q1

    def hess(t, e):
        return (t * t + e) ** q2 * (q3 * t * t + e)

    return grad, hess


def _Ttau(Axz: np.ndarray, Ax: np.ndarray, tau: float, lam: float):
    tl = math.sqrt(tau * lam / 2)
    mask = np.abs(Axz - tl) < tl
    T = np.flatnonzero(mask)
    empT = T.size == 0
    if empT:
        zp = Ax[Ax >= 0]
        if zp.size:
            s = int(math.ceil(0.01 * zp.size))
            tau = (zp[s - 1]) ** 2 / (2 * lam)
            tl = math.sqrt(tau * lam / 2)
            mask = np.abs(Ax - tl) < tl
            T = np.flatnonzero(mask)
            empT = T.size == 0
    return T, empT


def _refine_with_sparsity(A: np.ndarray, b: np.ndarray, x: np.ndarray, sp: int):
    K = 6
    idx = np.argpartition(-np.abs(x), sp + K - 1)[: sp + K - 1]
    sx = np.abs(x[idx])
    order = np.argsort(-sx)
    Ts = idx[order]
    HD = np.ones(K)
    X = np.zeros((x.size, K))
    if sx[order][sp - 1] - sx[order][sp] <= 2e-4:
        subset = Ts[sp - 1: sp - 1 + K]
        for i in range(min(K, subset.size)):
            Xi = np.zeros_like(x)
            Xi[Ts[: sp - 1]] = x[Ts[: sp - 1]]
            Xi[subset[i]] = x[subset[i]]
            norm = np.linalg.norm(Xi)
            if norm > 0:
                Xi /= norm
            X[:, i] = Xi
            HD[i] = np.count_nonzero(np.sign(A @ Xi) - b) / A.shape[0]
        j = np.argmin(HD)
        return X[:, j]
    sol = np.zeros_like(x)
    sol[Ts[:sp]] = x[Ts[:sp]]
    norm = np.linalg.norm(sol)
    if norm > 0:
        sol /= norm
    return sol


def _sparse_approx(x: np.ndarray):
    n = x.size
    xo = x.copy()
    sq = x * x
    T = np.flatnonzero(sq > 1e-2 / n)
    if T.size == 0:
        return xo
    sx = sq[T]
    order = np.argsort(-sx)
    sx_sorted = sx[order]
    y = 0.0
    total = float(np.sum(sx_sorted))
    limit = 0
    for i, val in enumerate(sx_sorted, start=1):
        y += val
        if y > 0.5 * total:
            limit = i
            break
    if limit == 0:
        limit = sx_sorted.size
    ratios = np.zeros(max(0, limit - 1))
    for i in range(limit - 1):
        if sx_sorted[i + 1] != 0:
            ratios[i] = sx_sorted[i] / sx_sorted[i + 1]
    if ratios.size:
        j = np.argmax(ratios) + 1
    else:
        j = limit
    if j > 1:
        j = min(T.size, 10 * j)
    indices = T[order[:j]]
    sparse = np.zeros_like(x)
    sparse[indices] = xo[indices]
    return sparse


def _fnorm2(var: np.ndarray) -> float:
    return float(np.linalg.norm(var) ** 2)


def _my_cg(fx, b: np.ndarray, cgtol: float, cgit: int, x0: np.ndarray):
    x = x0.copy()
    r = b - fx(x)
    e = float(r @ r)
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        w = fx(p)
        a = e / float(p @ w)
        x = x + a * p
        r = r - a * w
        e0 = e
        e = float(r @ r)
        p = r + (e / e0) * p
    return x
