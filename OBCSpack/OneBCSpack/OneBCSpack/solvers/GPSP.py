from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np


class GPSPError(ValueError):
    """Custom exception for the GPSP solver."""


def GPSP(A, b, s: int, k: int, pars: Dict | None = None):
    """Python port of ``GPSP.m``."""

    if pars is None:
        pars = {}

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if s is None or k is None:
        raise GPSPError('Inputs is not enough !!!')

    m, n = A.shape
    if b.size != m:
        raise GPSPError('Dimension mismatch between A and b')

    Ab = (b[:, None]) * A

    maxit, tol, eta, eps, acc, big, disp = _get_parameters(m, n, pars)

    s0 = s
    if big:
        sn = s / n
        if 0.01 < sn < 0.1:
            s = math.ceil((1 + min(1, m / n)) * s)
        elif 0.005 <= sn <= 0.01:
            s = math.ceil(1.2 * s)

    x = np.zeros(n)
    y = np.zeros(m)
    a = 1.0
    barx = x.copy()
    bary = y.copy()
    T = np.arange(min(s, n))
    I = np.array([], dtype=int)
    Axy = y - eps
    obj = _fnorm2(Axy)

    HAM = np.zeros(maxit)
    OBJ = np.zeros(maxit)
    stop0 = np.zeros(maxit)

    t0 = time.perf_counter()
    if disp:
        print('\n Start to run the solver: GPSP')
        print('------------------------------------------')
        print('  Iter           HamDist          CPUTime ')
        print('------------------------------------------')

    AT = Ab[:, T] if T.size else np.zeros((m, 0))
    for it in range(1, maxit + 1):
        alpha = max(1e-3, 10 / n)
        T0 = T.copy()
        I0 = I.copy()
        x0 = x.copy()
        y0 = y.copy()
        obj0 = obj

        v = Axy
        u = Ab.T @ v + eta * barx
        prev_inner_T = T0.copy()

        for _ in range(20):
            x, xT, T = _pros(barx - alpha * u, s)
            y = _proK(bary - alpha * v, k)
            if T.size:
                if AT.shape[1] != T.size or not np.array_equal(T, prev_inner_T):
                    AT = Ab[:, T]
                Ax = AT @ xT
            else:
                AT = np.zeros((m, 0))
                Ax = np.zeros(m)
            Axy = Ax - eps + y
            obj = _fnorm2(Axy) + eta * _fnorm2(xT)
            gap = _fnorm2(x - barx) + _fnorm2(y - bary)
            if obj < obj0 - 1e-6 * gap:
                break
            alpha *= 0.5
            prev_inner_T = T.copy()

        flag = bool(T.size == T0.size and np.array_equal(T, T0)) or _fnorm2(u) < tol
        if flag:
            I = np.flatnonzero(y)
            if I.size == I0.size:
                flag = np.array_equal(I, I0)

        if flag and I.size < m:
            if it > 5 and np.min(stop0[max(0, it - 5): it - 1]) == 1:
                break
            AT0 = Ab[:, T][y == 0, :]
            if AT0.size:
                tmp1 = np.linalg.solve(AT0.T @ AT0 + eta * np.eye(T.size), eps * np.sum(AT0, axis=0))
            else:
                tmp1 = np.zeros(T.size)
            tmp2 = eps - Ab[:, T][I, :] @ tmp1
            Ax1 = Ab[:, T] @ tmp1
            Axy1 = Ax1 - eps
            Axy1[I] = Axy1[I] + tmp2
            obj1 = _fnorm2(Axy1) + eta * _fnorm2(tmp1)
            gap = _fnorm2(xT - tmp1) + _fnorm2(y[I] - tmp2)
            stop0[it - 1] = 1
            if obj1 <= obj - 1e-6 * gap and np.count_nonzero(tmp2 > 0) <= k:
                x = np.zeros(n)
                y = np.zeros(m)
                if T.size:
                    x[T] = tmp1
                y[I] = tmp2
                Ax = Ax1
                Axy = Axy1
                obj = obj1

        sb = np.sign(-b * Ax)
        sb[sb == 0] = -1
        ham = 1 - np.count_nonzero(sb + b) / m
        HAM[it - 1] = ham
        OBJ[it - 1] = obj
        if disp:
            print(f"  {it:3d}          {ham * 100:8.2f}           {time.perf_counter()-t0:.3f}sec")

        stop1 = it > 5 and gap < tol
        stop2 = it > 5 and np.std(HAM[max(0, it - 6):it]) < 1e-6 * math.log(n)
        stop3 = it > 5 and np.std(OBJ[max(0, it - 6):it]) < 1e-6 * math.log(n)
        stop4 = stop0[it - 1] * (n < 1e4) + (n >= 1e4)
        stop5 = ham == 1 and gap < 1e-4
        if (stop1 and (stop2 or stop3) and stop4) or stop5:
            break

        if it % 50 == 0:
            k = max(0, math.ceil(k / 2))

        if acc:
            a0 = a
            a = (1 + math.sqrt(4 * a0 * a0 + 1)) / 2
            barx = x + ((a0 - 1) / a) * (x - x0)
            bary = y + ((a0 - 1) / a) * (y - y0)
            if stop0[it - 1]:
                Ax = AT @ barx[T]
            else:
                T_bar = np.flatnonzero(barx)
                Ax = (b[:, None] * A)[:, T_bar] @ barx[T_bar] if T_bar.size else np.zeros(m)
            barAxy = Ax - eps + bary
            barobj = _fnorm2(barAxy) + eta * _fnorm2(barx[np.flatnonzero(barx)])
            if barobj > obj:
                barx = x.copy()
                bary = y.copy()
                a = a0
            else:
                Axy = barAxy
        else:
            barx = x.copy()
            bary = y.copy()

    if disp:
        print('------------------------------------------')

    if np.count_nonzero(x) > s0:
        _, T = _maxk_abs(np.abs(x), s0)
        xn = np.zeros_like(x)
        xn[T] = x[T]
        x = xn

    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm

    out = {
        'sol': x,
        'soly': y,
        'obj': obj,
        'OBJ': OBJ[:it],
        'time': time.perf_counter() - t0,
        'iter': it,
    }
    return out


def _fnorm2(var: np.ndarray) -> float:
    return float(np.linalg.norm(var) ** 2)


def _get_parameters(m: int, n: int, pars: Dict) -> Tuple[int, float, float, float, int, int, int]:
    maxit = int(pars.get('maxit', 1_000))
    tol = float(pars.get('tol', 1e-9 * math.sqrt(min(m, n))))
    eta = float(pars.get('eta', 1e-4))
    eps = float(pars.get('eps', 0.01 * ((n < 1e4) + (n >= 1e4) / math.log(max(n, 2)))))
    acc = int(pars.get('acc', 0))
    big = int(pars.get('big', int(m < 2.1 * n)))
    disp = int(pars.get('disp', 1))
    return maxit, tol, eta, eps, acc, big, disp


def _pros(x: np.ndarray, s: int):
    if s <= 0:
        return np.zeros_like(x), np.array([], dtype=int), np.array([], dtype=int)
    _, idx = _maxk_abs(np.abs(x), s)
    idx = np.sort(idx)
    xT = x[idx]
    xs = np.zeros_like(x)
    xs[idx] = xT
    return xs, xT, idx


def _proK(y: np.ndarray, k: int):
    y_new = y.copy()
    if k <= 0:
        y_new[y_new > 0] = 0
        return y_new
    vals, idx = _maxk(y_new, k)
    if vals.size == 0:
        y_new[y_new > 0] = 0
        return y_new
    thresh = vals[-1]
    if thresh > 0:
        mask = (y_new < 0) | (y_new >= thresh)
        y_new = y_new * mask
    else:
        y_new[y_new > 0] = 0
    return y_new


def _maxk_abs(v: np.ndarray, k: int):
    if k <= 0:
        return np.array([]), np.array([], dtype=int)
    k = min(k, v.size)
    idx = np.argpartition(-np.abs(v), k - 1)[:k]
    order = np.argsort(-np.abs(v[idx]))
    idx = idx[order]
    return v[idx], idx


def _maxk(v: np.ndarray, k: int):
    if k <= 0:
        return np.array([]), np.array([], dtype=int)
    k = min(k, v.size)
    idx = np.argpartition(-v, k - 1)[:k]
    order = np.argsort(-v[idx])
    idx = idx[order]
    return v[idx], idx
