from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

LinearOp = Any


def yall1(A, b: np.ndarray, opts: Dict[str, Any]):
    Aop, Atop, b_scaled, opts = _linear_operators(A, b, opts)

    m = len(b_scaled)
    L1L1 = opts.get('nu', 0) > 0
    if L1L1 and 'weights' in opts:
        opts['weights'] = np.concatenate([np.asarray(opts['weights']).ravel(), np.ones(m)])

    Atb = Atop(b_scaled)
    bmax = np.linalg.norm(b_scaled, ord=np.inf)

    rho = opts.get('rho', 0)
    delta = opts.get('delta', 0)
    nu = opts.get('nu', 0)
    nonneg = opts.get('nonneg', 0)
    tol = opts['tol']

    L2Unc_zsol = rho > 0 and np.linalg.norm(Atb, ord=np.inf) <= rho
    L2Con_zsol = delta > 0 and np.linalg.norm(b_scaled) <= delta
    L1L1_zsol = nu > 0 and bmax < tol
    BP_zsol = rho == 0 and delta == 0 and nu == 0 and bmax < tol

    if L2Unc_zsol or L2Con_zsol or BP_zsol or L1L1_zsol:
        x = np.zeros_like(Atb)
        return x

    x1, _ = _yall1_solve(Aop, Atop, b_scaled, opts)
    x = x1 * bmax
    if L1L1:
        x = x[:-m]
    if 'basis' in opts:
        x = opts['basis']['trans'](x)
    if nonneg:
        x = np.maximum(0, x)
    return x


def _linear_operators(A0, b0, opts):
    b = np.asarray(b0, dtype=float)
    if callable(A0):
        A = lambda x: A0(x, 1)
        At = lambda x: A0(x, 2)
    elif isinstance(A0, dict) and 'times' in A0 and 'trans' in A0:
        A = A0['times']
        At = A0['trans']
    else:
        A0 = np.asarray(A0, dtype=float)
        A = lambda x: A0 @ x
        At = lambda y: A0.T @ y

    if 'basis' in opts:
        C = A
        Ct = At
        B = opts['basis']['times']
        Bt = opts['basis']['trans']
        A = lambda x: C(Bt(x))
        At = lambda y: B(Ct(y))

    if opts.get('nu', 0) > 0:
        C = A
        Ct = At
        m = len(b0)
        nu = opts['nu']
        t = 1.0 / math.sqrt(1 + nu ** 2)
        A = lambda x: (C(x[:-m]) + nu * x[-m:]) * t
        At = lambda y: np.concatenate([Ct(y), nu * y]) * t
        b = b0 * t

    if 'nonorth' not in opts:
        opts['nonorth'] = _check_orth(A, At, b)

    return A, At, b, opts


def _check_orth(A, At, b):
    s1 = np.random.randn(len(b))
    s2 = A(At(s1))
    err = np.linalg.norm(s1 - s2) / np.linalg.norm(s1)
    return int(err > 1e-12)


def _yall1_solve(A, At, b, opts):
    m = len(b)
    bnrm = np.linalg.norm(b)

    (tol, mu, maxit, print_it, nu, rho, delta, weights,
     nonneg, nonorth, gamma) = _get_opts(b, opts)

    x = opts.get('x0')
    if x is None:
        x = At(b)
    x = x.astype(float)
    n = len(x)
    z = opts.get('z0')
    if z is None:
        z = np.zeros(n)

    if nonorth:
        y = np.zeros(m)
        Aty = np.zeros(n)
    else:
        y = np.zeros(m)
        Aty = np.zeros(n)

    mu_orig = mu
    rdmu = rho / mu
    rdmu1 = rdmu + 1
    bdmu = b / mu
    ddmu = delta / mu

    Out = {'cntA': 0, 'cntAt': 0}
    rel_gap = 0.0
    rel_rd = 0.0
    rel_rp = 0.0

    for iter_idx in range(1, maxit + 1):
        xdmu = x / mu
        if not nonorth:
            y = A(z - xdmu) + bdmu
            if rho > 0:
                y = y / rdmu1
            elif delta > 0:
                norm_y = np.linalg.norm(y)
                if norm_y > ddmu:
                    y *= ddmu / norm_y
            Aty = At(y)
        else:
            ry = A(Aty - z + xdmu) - bdmu
            if rho > 0:
                ry += rdmu * y
            Atry = At(ry)
            denom = float(Atry @ Atry)
            if rho > 0:
                denom += rdmu * float(ry @ ry)
            stp = float(ry @ ry) / (denom + np.finfo(float).eps)
            Out['cntAt'] += 1
            y = y - stp * ry
            Aty = Aty - stp * Atry

        z = _proj2box(Aty + xdmu, weights, nonneg, nu, m)
        Out['cntA'] += 1
        Out['cntAt'] += 1

        rd = Aty - z
        xp = x.copy()
        x = x + gamma * mu * rd

        if iter_idx % 2 == 0:
            stop, rel_gap, rel_rd, rel_rp = _check_stopping(
                A, At, b, bnrm, x, xp, rd, z, weights, rho, delta, mu, tol, nu)
            mu, rdmu, rdmu1, bdmu, ddmu = _update_mu(mu, mu_orig, rho, rel_gap, rel_rd, iter_idx, tol, delta, b, rdmu)
            if stop:
                break

    Out['iter'] = iter_idx
    Out['mu'] = (mu_orig, mu)
    Out['y'] = y
    Out['z'] = z
    return x, Out


def _get_opts(b, opts):
    tol = opts['tol']
    mu = opts.get('mu', np.mean(np.abs(b)))
    maxit = opts.get('maxit', 5000)
    print_it = opts.get('print', 0)
    nu = opts.get('nu', 0)
    rho = opts.get('rho', np.finfo(float).eps)
    delta = opts.get('delta', 0)
    weights = np.asarray(opts.get('weights', 1.0))
    nonneg = opts.get('nonneg', 0)
    nonorth = opts.get('nonorth', 0)
    gamma = opts.get('gamma', 1.0)
    return tol, mu, maxit, print_it, nu, rho, delta, weights, nonneg, nonorth, gamma


def _proj2box(z, w, nonneg, nu, m):
    z = z.copy()
    if nonneg:
        wv = w if np.isscalar(w) else np.asarray(w).ravel()
        z = np.minimum(wv, np.real(z))
        if nu > 0:
            z[-m:] = np.maximum(-1, z[-m:])
    else:
        wv = w if np.isscalar(w) else np.asarray(w).ravel()
        z = z * wv / np.maximum(wv, np.abs(z))
    return z


def _check_stopping(A, At, b, bnrm, x, xp, rd, z, w, rho, delta, mu, tol, nu):
    q = 0.1
    if delta > 0:
        q = 0
    rdnrm = np.linalg.norm(rd)
    rel_rd = rdnrm / (np.linalg.norm(z) + 1e-12)

    objp = np.sum(np.abs(w * x))
    objd = float(b @ np.zeros_like(b))

    if rho > 0:
        rp = A(x) - b
        rpnrm = np.linalg.norm(rp)
        objp += (0.5 / rho) * rpnrm ** 2
        objd -= 0.5 * rho * np.linalg.norm(np.zeros_like(b)) ** 2
    else:
        rp = A(x) - b
        rpnrm = np.linalg.norm(rp)

    rel_gap = abs(objd - objp) / (abs(objp) + 1e-12)
    xrel_chg = np.linalg.norm(x - xp) / (np.linalg.norm(x) + 1e-12)
    if xrel_chg < tol * (1 - q):
        return True, rel_gap, rel_rd, rpnrm / (bnrm + 1e-12)
    if xrel_chg >= tol * (1 + q):
        return False, rel_gap, rel_rd, rpnrm / (bnrm + 1e-12)
    if rel_gap >= tol:
        return False, rel_gap, rel_rd, rpnrm / (bnrm + 1e-12)
    if rel_rd >= tol:
        return False, rel_gap, rel_rd, rpnrm / (bnrm + 1e-12)
    if rho > 0:
        p_feasible = True
    elif delta > 0:
        p_feasible = rpnrm <= delta * (1 + tol)
    else:
        p_feasible = rpnrm < tol * (bnrm + 1e-12)
    return p_feasible, rel_gap, rel_rd, rpnrm / (bnrm + 1e-12)


def _update_mu(mu, mu_orig, rho, rel_gap, rel_rd, iter_idx, tol, delta, b, rdmu):
    mfrac = 0.1
    big = 50
    nup = 8
    mu_min = (mfrac ** nup) * mu_orig
    if not (rel_gap > big * rel_rd and mu > 1.1 * mu_min and iter_idx > 10):
        return mu, rdmu, rdmu + 1, b / mu, delta / mu
    mu = max(mfrac * mu, mu_min)
    rdmu = rho / mu
    return mu, rdmu, rdmu + 1, b / mu, delta / mu

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
        denom = float(p @ w)
        a = e / denom if denom != 0 else 0.0
        x = x + a * p
        r = r - a * w
        e0 = e
        e = float(r @ r)
        p = r + (e / e0 if e0 != 0 else 0.0) * p
    return x
