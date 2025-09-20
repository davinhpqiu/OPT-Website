from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from .yall1 import yall1


def MIRL1(data: Dict[str, Any], n: int, mu: float, pars: Dict[str, Any] | None = None):
    if pars is None:
        pars = {}
    if data is None or n is None or mu is None:
        raise ValueError('Inputs are not enough...')

    if 'A' not in data or 'b' not in data:
        raise ValueError('data.A or data.b is missing')

    A = data['A']
    b = np.asarray(data['b'], dtype=float)
    m = b.size

    if callable(A):
        Aya = {'times': A, 'trans': data.get('At')}
        if Aya['trans'] is None:
            raise ValueError('<data.At> is missing, unable to run the solver ...')
    else:
        Aya = np.asarray(A, dtype=float)

    itmax = 1000 if n < 1000 else 100
    rate = pars.get('rate', 1.0 / max(np.log(n / m), 1.0))
    tol = pars.get('tol', 1e-4)
    disp = pars.get('disp', 1)
    i0 = int(np.ceil(m / (4 * max(np.log(n / m), 1.0))))
    theta = mu * m / n / 10

    x = np.zeros(n)
    w = np.ones(n)
    opts = {'tol': tol}
    t_start = time.perf_counter()

    if disp:
        print(' \n Start to run the solver -- MIRL1 ')
        print(' -------------------------------------')
        print(' Iter          ObjVal         CPUTime ')
        print(' -------------------------------------')

    for iter_idx in range(1, itmax + 1):
        opts['pho'] = mu
        opts['weights'] = w
        opts['x0'] = x

        x_prev = x.copy()
        w_prev = w.copy()

        x = yall1(Aya, b, opts)
        dx = x - x_prev
        error = np.linalg.norm(dx) / max(np.linalg.norm(x_prev), 1.0)

        if disp:
            Ax = _apply_A(A, x)
            fx = 0.5 * np.linalg.norm(Ax - b) ** 2
            print(f"{iter_idx:4d}          {fx:5.2e}      {time.perf_counter()-t_start:6.3f}sec")

        if np.count_nonzero(x) > 0 and error < 1e-4 * np.sqrt(n) or iter_idx == itmax:
            Ax = _apply_A(A, x)
            out = {
                'sol': x,
                'sp': int(np.count_nonzero(x)),
                'iter': iter_idx,
                'time': time.perf_counter() - t_start,
                'obj': 0.5 * np.linalg.norm(Ax - b) ** 2,
                'error': np.linalg.norm(_apply_At(data, Ax - b)) ** 2,
            }
            return out

        sx = np.sort(np.abs(x))[::-1]
        eps2 = max(1e-3, sx[i0 - 1]) if i0 - 1 < sx.size else 1e-3
        s_val = int(pars.get('s', sparsity(sx, rate)))

        theta *= 1.005
        w = _mod_weight(x, np.abs(dx), theta, s_val, eps2)
        beta = np.sum(w_prev * np.abs(x)) / np.sum(w * np.abs(x) + 1e-12)
        if beta > 1 or np.count_nonzero(x) == 0:
            mu *= 0.2
        else:
            mu *= beta

    out = {
        'sol': x,
        'sp': int(np.count_nonzero(x)),
        'iter': itmax,
        'time': time.perf_counter() - t_start,
        'obj': 0.5 * np.linalg.norm(_apply_A(A, x) - b) ** 2,
        'error': np.nan,
    }
    return out


def _apply_A(A, x):
    if callable(A):
        return A(x)
    return A @ x


def _apply_At(data, v):
    At = data.get('At')
    if At is None:
        A = data['A']
        return A.T @ v
    return At(v)


def _mod_weight(x, h, theta, k, eps2):
    n = len(x)
    w = np.ones(n)
    eps1 = 1e-10
    idx = np.argsort(-h)
    if k == 0:
        w = 1.0 / (np.abs(x) + eps2)
    else:
        top = idx[:k]
        rest = idx[k:]
        if top.size:
            w[top] = eps1 + theta * np.sum(h[top[1:]]) / (np.sum(h[top]) + np.finfo(float).eps)
        if rest.size:
            w[rest] = eps1 + theta + 1.0 / (np.abs(x[rest]) + eps2)
    return w


def sparsity(x_sorted, rate):
    rs = rate * np.sum(x_sorted)
    y = 0.0
    sp = 0
    for val in x_sorted:
        if y >= rs:
            break
        sp += 1
        y += val
    return sp
