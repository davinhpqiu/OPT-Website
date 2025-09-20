from __future__ import annotations

import math
import time
from typing import Any, Dict, Tuple

import numpy as np


def PSNP(data: Dict[str, Any], n: int, lam: float, pars: Dict[str, Any] | None = None):
    if pars is None:
        pars = {}
    if data is None or n is None or lam is None:
        print(' No enough inputs. No problems will be solverd!')
        return {}

    (sig1, sig2, q, q1, q2, lamq, lamq1, alpha0, cg_tol, cg_it, x0, tol, maxit,
     newton, show, cond, rate, i0) = _set_parameters(n, lam, pars)

    fnorm = lambda var: float(np.linalg.norm(var, ord='fro') ** 2)
    funcs = lambda xT, T, key: _funCS(xT, T, n, key, data)
    prox_lq = lambda x, t: _prox_lq(x, t, q)
    qnorm = lambda absvar: np.sum(absvar ** q)
    g_qnorm = lambda var, absvar: np.sign(var) * absvar ** q1
    h_qnorm = lambda absvar: absvar ** q2

    Error = np.zeros(maxit)
    x = x0.copy()
    T = np.flatnonzero(x)
    sT = T.size
    w = x[T]

    if sT > 0:
        absw = np.abs(w)
        obj = funcs(w, T, 'f') + lam * qnorm(absw)
        grad = funcs(w, T, 'g')
        gradT = grad[T] + lamq * g_qnorm(w, absw)
        grad_full = np.zeros_like(x)
        grad_full[T] = gradT
    else:
        obj = funcs(x, np.array([], dtype=int), 'f')
        grad_full = funcs(x, np.array([], dtype=int), 'g')
    
    t0 = time.perf_counter()
    if show:
        print(' \nStart to run the solver -- PCSNP' if cond else '\n Start to run the solver -- PSNP')
        print(' -------------------------------------')
        print(' Iter     ObjVal    Sparsity     Time ')
        print(' -------------------------------------')
        print(f"{0:4d}     {obj:5.2e}    {sT:4d}    {time.perf_counter()-t0:6.2f}sec")

    for it in range(1, maxit + 1):
        alpha = alpha0
        Told = T.copy()
        grad = grad_full

        for _ in range(i0):
            w_new, T = prox_lq(x - alpha * grad, alpha * lam)
            absw = np.abs(w_new)
            fx = funcs(w_new, T, 'f')
            obj_w = fx + lam * qnorm(absw)
            if T.size == 0 or obj_w < obj - sig1 * fnorm(w_new - x[T]):
                break
            alpha *= rate

        if T.size == 0:
            alpha /= rate
            w_new, T = prox_lq(x - alpha * grad, alpha * lam)
            absw = np.abs(w_new)
            fx = funcs(w_new, T, 'f')
            obj_w = fx + lam * qnorm(absw)

        x = np.zeros(n)
        x[T] = w_new
        obj_old = obj
        obj = obj_w
        sT_old = sT
        sT = T.size
        ident = sT == sT_old and np.array_equal(T, Told)

        if cond:
            switch_on = ident
        else:
            switch_on = (sT < 0.25 * n or ident) and (_ < i0 - 1 or it >= 5)

        if newton and sT > 0 and switch_on:
            grad_tmp, Hess = funcs(w_new, T, 'gh')
            gradT = grad_tmp[T]
            if q > 0:
                gradT = gradT + lamq * g_qnorm(w_new, absw)
                dw = lamq1 * h_qnorm(absw)
                if callable(Hess):
                    orig = Hess

                    def Hess(v):
                        return orig(v) + dw * v
                else:
                    Hess = Hess.astype(float)
                    diag = np.arange(sT)
                    Hess[diag, diag] += dw
            if callable(Hess):
                if it > 10:
                    cg_tol = max(cg_tol / 10.0, 1e-15 * sT)
                    cg_it = min(cg_it + 5, 25)
                d = _my_cg(Hess, gradT, cg_tol, cg_it, np.zeros(sT))
            else:
                d = np.linalg.solve(Hess, gradT)

            beta = 1.0
            Fd = fnorm(d)
            for _ in range(5):
                v = w_new - beta * d
                abs_v = np.abs(v)
                fx = funcs(v, T, 'f')
                obj_v = fx + lam * qnorm(abs_v)
                if obj_v <= obj - sig2 * beta * beta * Fd:
                    x[T] = v
                    w_new = v
                    absw = abs_v
                    obj = obj_v
                    break
                beta *= 0.25

        grad_tmp = funcs(w_new, T, 'g')
        gradT = grad_tmp[T]
        if q > 0:
            gradT = gradT + lamq * g_qnorm(w_new, absw)
        grad_full = np.zeros(n)
        grad_full[T] = gradT

        if it > 1 and T.size == 0:
            ErrGradT = 1e10
            lam /= 1.5
            lamq = lam * q
            lamq1 = lamq * q1
        else:
            ErrGradT = np.linalg.norm(gradT, ord=np.inf)

        ErrObj = abs(obj - obj_old) / (1 + abs(obj))
        Error[it - 1] = ErrGradT / math.sqrt(n)

        if show:
            print(f"{it:4d}     {fx:5.2e}    {sT:4d}    {time.perf_counter()-t0:6.2f}sec")

        if ((n > 5e4 and callable(data['A'])) or ident) and max(Error[it - 1], ErrObj) < tol:
            break

    return {
        'time': time.perf_counter() - t0,
        'iter': it,
        'sol': x,
        'obj': obj,
        'error': fnorm(grad_full),
    }


def _set_parameters(n: int, lam: float, pars: Dict[str, Any]):
    x0 = np.asarray(pars.get('x0', np.zeros(n)), dtype=float)
    q = float(pars.get('q', 0.5))
    tol = float(pars.get('tol', 1e-6))
    maxit = int(pars.get('maxit', 10_000))
    newton = int(pars.get('newton', 1))
    show = int(pars.get('show', 1))
    cond = int(pars.get('cond', 1))

    sig1 = 1e-6
    sig2 = 1e-10
    q1 = q - 1
    q2 = q - 2
    lamq = lam * q
    lamq1 = lamq * q1
    alpha0 = 1 - q / 2
    cg_tol = 1e-8
    cg_it = 10
    rate = 0.5
    i0 = 6
    return sig1, sig2, q, q1, q2, lamq, lamq1, alpha0, cg_tol, cg_it, x0, tol, maxit, newton, show, cond, rate, i0


def _funCS(xT, T, n: int, key: str, data: Dict[str, Any]):
    A = data['A']
    b = data['b']
    is_func = callable(A)
    if is_func:
        At = data['At']
    else:
        A = np.asarray(A, dtype=float)
        At = lambda v: A.T @ v

    if T is None or len(T) == 0:
        Axb = -b
        if key == 'f':
            return 0.5 * float(np.linalg.norm(Axb) ** 2)
        grad = At(Axb)
        if key == 'g':
            out = np.zeros(n)
            out[: grad.size] = grad
            return out
        if key == 'gh':
            return np.zeros(n), []
        return grad

    T = np.asarray(T, dtype=int)
    if is_func:
        vec = np.zeros(n)
        vec[T] = xT
        Axb = A(vec) - b
    else:
        AT = A[:, T]
        Axb = AT @ xT - b

    if key == 'f':
        return 0.5 * float(np.linalg.norm(Axb) ** 2)

    grad = At(Axb)
    if key == 'g':
        out = np.zeros(n)
        out[T] = grad[T]
        return out

    if key == 'gh':
        grad_full = np.zeros(n)
        grad_full[T] = grad[T]
        if is_func:
            def Hess(v):
                tmp = np.zeros(n)
                tmp[T] = v
                return At(A(tmp))[T]
        else:
            if T.size < 1000 and A.shape[0] < 1000:
                Hess = AT.T @ AT
            else:
                Hess = lambda v: AT.T @ (AT @ v)
        return grad_full, Hess

    raise ValueError('Invalid key')


def _prox_lq(a: np.ndarray, lam: float, q: float):
    a = np.asarray(a, dtype=float)
    if q == 0:
        thresh = math.sqrt(2 * lam)
        mask = np.abs(a) > thresh
        return a[mask], np.flatnonzero(mask)
    if q == 0.5:
        thresh = 1.5 * lam ** (2 / 3)
        mask = np.abs(a) > thresh
        aT = a[mask]
        phi = np.arccos((lam / 4.0) * (3.0 / np.abs(aT)) ** 1.5)
        px = (4.0 / 3.0) * aT * (np.cos((np.pi - phi) / 3.0) ** 2)
        return px, np.flatnonzero(mask)
    if q == 2 / 3:
        thresh = 2 * (2 * lam / 3) ** (3 / 4)
        mask = np.abs(a) > thresh
        aT = a[mask]
        tmp1 = aT ** 2 / 2.0
        tmp2 = np.sqrt(tmp1 ** 2 - (8 * lam / 9) ** 3)
        phi = (tmp1 + tmp2) ** (1 / 3) + (tmp1 - tmp2) ** (1 / 3)
        px = np.sign(aT) / 8.0 * (np.sqrt(phi) + np.sqrt(2 * np.abs(aT) / np.sqrt(phi) - phi)) ** 3
        return px, np.flatnonzero(mask)
    return _newton_lq(a, lam, q)


def _newton_lq(a: np.ndarray, lam: float, q: float):
    thresh = (2 - q) * lam ** (1 / (2 - q)) * (2 * (1 - q)) ** ((1 - q) / (q - 2))
    mask = np.abs(a) > thresh
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([]), idx
    zT = a[idx]
    w = zT.copy()
    maxit = 100
    q1 = q - 1
    q2 = q - 2
    lamq = lam * q
    lamq1 = lamq * q1

    def grad(u, absu):
        return u - zT + lamq * np.sign(u) * absu ** q1

    def hess(absu):
        return 1 + lamq1 * absu ** q2

    def func(u, absu):
        return np.linalg.norm(u - zT) ** 2 / 2.0 + lam * np.sum(absu ** q)

    absw = np.abs(w)
    fx0 = func(w, absw)
    for _ in range(maxit):
        g = grad(w, absw)
        d = -g / hess(absw)
        alpha = 1.0
        w0 = w.copy()
        for _ in range(10):
            w = w0 + alpha * d
            absw = np.abs(w)
            fx = func(w, absw)
            if fx < fx0 - 1e-4 * np.linalg.norm(w - w0) ** 2:
                break
            alpha *= 0.5
        if np.linalg.norm(g) < 1e-8:
            break
        fx0 = fx
    return w, idx


def _my_cg(fx, b: np.ndarray, cgtol: float, cgit: int, x0: np.ndarray):
    x = x0.copy()
    r = b - fx(x)
    e = float(np.linalg.norm(r, ord='fro') ** 2)
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        w = fx(p)
        denom = float(np.sum(p * w))
        a = e / denom if denom != 0 else 0.0
        x = x + a * p
        r = r - a * w
        e0 = e
        e = float(np.linalg.norm(r, ord='fro') ** 2)
        p = r + (e / e0 if e0 != 0 else 0.0) * p
    return x
