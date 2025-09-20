from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np


class NSSVMError(ValueError):
    """Custom exception for the NSSVM solver."""


def NSSVM(A, y, pars: Optional[Dict] = None):
    """Python port of ``NSSVM.m``."""

    if pars is None:
        pars = {}

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size != A.shape[0]:
        raise NSSVMError('Inputs are not enough')

    m, n = A.shape
    if np.issparse(A) and A.nnz / (m * n) > 0.1:
        A = np.asarray(A.todense())

    if n < 3e4:
        Qt = (y[:, None]) * A
    else:
        Qt = (y[:, None]) * A
    Q = Qt.T
    maxit, alpha, tune, disp, tol, eta, s0, C, c = _get_parameters(m, n, pars)

    T1 = np.flatnonzero(y == 1)
    T2 = np.flatnonzero(y == -1)
    nT1 = T1.size
    nT2 = T2.size

    if nT1 < s0:
        T = np.concatenate([T1, T2[: s0 - nT1]])
    elif nT2 < s0:
        T = np.concatenate([T1[: s0 - nT2], T2])
    else:
        half = int(math.ceil(s0 / 2))
        T = np.concatenate([T1[:half], T2[: s0 - half]])
    T = np.sort(T[:s0])
    s = s0

    nT1 = int(nT1)
    nT2 = int(nT2)
    b = 1 if nT1 >= nT2 else -1
    bb = b
    w = np.zeros(n)
    gz = -np.ones(m)

    ERR = np.zeros(maxit)
    ACC = np.zeros(maxit + 1)
    ACC[0] = 1 - np.count_nonzero(np.sign(b) - y) / m
    ET = np.ones(s) / C

    maxACC = 0.0
    flag = True
    r = 1.1
    count = 1
    count0 = 2
    iter0 = -1
    alpha0 = alpha.copy()
    tmp0 = np.zeros(m)
    maxwb = np.zeros(n + 1)

    t0 = time.perf_counter()
    if disp:
        print(' \n Start to run the solver -- NSSVM')
        print(' ------------------------------------------')
        print('  Iter          Accuracy          CPUTime ')
        print(' ------------------------------------------')

    for iter_idx in range(1, maxit + 1):
        if iter_idx == 1 or flag:
            QT = Q[:, T]
            QtT = Qt[T, :]
            yT = y[T]
            ytT = yT.reshape(1, -1)
        alphaT = alpha[T]
        gzT = -gz[T]
        alyT = -float(ytT @ alphaT)

        err = (abs(np.linalg.norm(alpha) ** 2 - np.linalg.norm(alphaT) ** 2)
               + np.linalg.norm(gzT) ** 2 + alyT ** 2) / (m * n)
        ERR[iter_idx - 1] = math.sqrt(err)

        if tune and iter_idx < 30 and m <= 1e8:
            stop1 = iter_idx > 5 and err < tol * s * math.log2(m) / 100
            stop2 = s != s0 and abs(ACC[iter_idx - 1] - np.max(ACC[: iter_idx - 1])) <= 1e-4
            stop3 = s != s0 and iter_idx > 10 and np.max(ACC[iter_idx - 5: iter_idx + 1]) < maxACC
            stop4 = (count != count0 + 1) and ACC[iter_idx - 1] >= ACC[0]
            stop = stop1 and (stop2 or stop3) and stop4
        else:
            stop1 = err < tol * math.sqrt(s) * math.log10(m)
            stop2 = iter_idx > 4 and np.std(ACC[iter_idx - 1: iter_idx + 1]) < 1e-4
            stop3 = iter_idx > 20 and abs(np.max(ACC[max(0, iter_idx - 8): iter_idx + 1]) - maxACC) <= 1e-4
            stop = (stop1 and stop2) or stop3

        if disp:
            print(f"  {iter_idx:3d}          {ACC[iter_idx - 1]:8.5f}          {time.perf_counter()-t0:.3f}sec")

        if ACC[iter_idx - 1] > 0 and (ACC[iter_idx - 1] >= 0.99999 or stop):
            break

        ET0 = ET
        ET = (alphaT >= 0) / C + (alphaT < 0) / c

        if min(n, s) > 1e3:
            d = _my_cg(QT, yT, ET, np.concatenate([gzT, [alyT]]), 1e-10, 50, np.zeros(s + 1))
            dT = d[:s]
            dend = d[-1]
        else:
            if s <= n:
                if iter_idx == 1 or flag:
                    PTT0 = QtT @ QT
                PTT = PTT0 + np.diag(ET)
                rhs = np.concatenate([gzT, [alyT]])
                d = np.linalg.solve(np.block([[PTT, yT.reshape(-1, 1)], [ytT, [[0]]]]), rhs)
                dT = d[:s]
                dend = d[-1]
            else:
                ETinv = 1.0 / ET
                flag1 = np.count_nonzero(ET0) != np.count_nonzero(ET)
                flag2 = np.count_nonzero(ET0) == np.count_nonzero(ET) and np.count_nonzero(ET0 - ET) == 0
                if iter_idx == 1 or flag or flag1 or not flag2:
                    EQtT = ETinv[:, None] * QtT
                    P0 = np.eye(n) + QT @ EQtT
                Ey = ETinv * yT
                Hy = Ey - EQtT @ np.linalg.solve(P0, QT @ Ey)
                denom = float(ytT @ Hy)
                dend = (float(gzT @ Hy) - alyT) / denom if denom != 0 else 0.0
                tem = ETinv * (gzT - dend * yT)
                dT = tem - EQtT @ np.linalg.solve(P0, QT @ tem)

        alpha = np.zeros(m)
        alphaT = alphaT + dT
        alpha[T] = alphaT
        b = b + dend

        w = QT @ alphaT
        Qtw = Qt @ w
        tmp = y * Qtw
        gz = Qtw - 1 + b * y
        ET1 = (alphaT >= 0) / C + (alphaT < 0) / c
        gz[T] = alphaT * ET1 + gz[T]

        ACC[iter_idx] = 1 - np.count_nonzero(np.sign(tmp + b) - y) / m

        if m <= 1e7:
            bb0 = np.mean(yT - tmp[T])
            acc0 = 1 - np.count_nonzero(np.sign(tmp + bb0) - y) / m
            if ACC[iter_idx] >= acc0:
                bb = b
            else:
                ACC[iter_idx] = acc0
                bb = bb0
        else:
            bb = b

        if m < 6e6 and ACC[iter_idx] < 0.5:
            opt_max_iter = 10 if m >= 1e6 else 20
            best_b = _best_bias(tmp, y, bb)
            acc_tmp = 1 - np.count_nonzero(np.sign(tmp + best_b) - y) / m
            if acc_tmp > ACC[iter_idx]:
                bb = best_b
                ACC[iter_idx] = acc_tmp

        if ACC[iter_idx] >= maxACC:
            maxACC = ACC[iter_idx]
            alpha0 = alpha.copy()
            tmp0 = tmp.copy()
            maxwb = np.concatenate([w, [bb]])

        T0 = T.copy()
        mark = 0
        if tune and (err < tol or iter_idx % 10 == 0) and iter_idx > iter0 + 2 and count < 10:
            count0 = count
            count += 1
            s = min(m, int(math.ceil(r * s)))
            iter0 = iter_idx
            if count > (1 if m >= 1e6 or n < 3 else 0) + (1 if m < 1e6 and n >= 5 else 0):
                alpha = np.zeros(m)
                gz = -np.ones(m)
                mark = 1
        else:
            count0 = count

        if s != m:
            idx = _maxk_indices(np.abs(alpha - eta * gz), s)
            T = np.sort(idx)
            if mark:
                nT = np.count_nonzero(y[T] == 1)
                if nT == s:
                    if nT2 <= 0.75 * s:
                        T = np.concatenate([T[: s - int(math.ceil(nT2 / 2))], T2[: int(math.ceil(nT2 / 2))]])
                    else:
                        T = np.concatenate([T[: int(math.ceil(s / 4))], T2[: s - int(math.ceil(s / 4))]])
                elif nT == 0:
                    if nT1 <= 0.75 * s:
                        T = np.concatenate([T[: s - int(math.ceil(nT1 / 2))], T1[: int(math.ceil(nT1 / 2))]])
                    else:
                        T = np.concatenate([T[: int(math.ceil(s / 4))], T1[: s - int(math.ceil(s / 4))]])
                T = np.sort(T[:s])
        else:
            T = np.arange(m)

        flag = True
        flag3 = T0.size == s and np.array_equal(T, T0)
        if flag3 or T0.size == m:
            flag = False
            T = T0

    wb = np.concatenate([w, [bb]])
    acc = ACC[min(iter_idx, ACC.size - 1)]

    if m <= 1e7 and iter_idx > 1:
        best_b = _best_bias(tmp0, y, maxwb[-1])
        acc0 = 1 - np.count_nonzero(np.sign(tmp0 + best_b) - y) / m
        if acc < acc0:
            wb = np.concatenate([maxwb[:-1], [best_b]])
            acc = acc0

    if acc < maxACC - 1e-4:
        alpha = alpha0
        wb = maxwb
        acc = maxACC

    if disp:
        print(' ------------------------------------------')

    return {
        's': s,
        'w': wb,
        'sv': s,
        'ACC': acc,
        'iter': iter_idx,
        'time': time.perf_counter() - t0,
        'alpha': alpha,
    }


def _get_parameters(m: int, n: int, pars: Dict):
    maxit = int(pars.get('maxit', 1_000))
    alpha = np.asarray(pars.get('alpha', np.zeros(m)), dtype=float)
    tune = int(pars.get('tune', 0))
    disp = int(pars.get('disp', 1))
    tol = float(pars.get('tol', 1e-6))
    eta = float(pars.get('eta', min(1 / m, 1e-4)))
    if max(m, n) < 1e4:
        beta = 1
    elif m <= 5e5:
        beta = 0.05
    elif m <= 1e8:
        beta = 10
    else:
        beta = 1
    s0 = int(math.ceil(beta * n * (math.log2(max(m / n, 2))) ** 2))
    if 's0' in pars:
        s0 = min(m, int(pars['s0']))
    C = float(pars.get('C', math.log10(m) if m > 5e6 else 0.5))
    c = 0.01 * C
    return maxit, alpha, tune, disp, tol, eta, s0, C, c


def _my_cg(QT: np.ndarray, yT: np.ndarray, ET: np.ndarray, b: np.ndarray, cgtol: float, cgit: int, x0: np.ndarray):
    x = x0.copy()
    r = b.copy()
    e = float(np.dot(r, r))
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        p1 = p[:-1]
        y_val = yT
        w = np.concatenate([QT.T @ (QT @ p1) + ET * p1 + p[-1] * y_val, [float(np.dot(y_val, p1))]])
        denom = float(np.dot(p, w))
        a = e / denom if denom != 0 else 0.0
        x = x + a * p
        r = r - a * w
        e0 = e
        e = float(np.dot(r, r))
        p = r + (e / e0 if e0 != 0 else 0.0) * p
    return x


def _maxk_indices(values: np.ndarray, k: int):
    k = min(k, values.size)
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-values, k - 1)[:k]
    return idx


def _best_bias(tmp: np.ndarray, y: np.ndarray, guess: float) -> float:
    candidates = np.unique(-tmp)
    candidates = np.concatenate((candidates, candidates + 1e-6, candidates - 1e-6, [guess]))
    best_b = guess
    best_err = np.count_nonzero(np.sign(tmp + guess) - y)
    for b in candidates:
        err = np.count_nonzero(np.sign(tmp + b) - y)
        if err < best_err:
            best_b = b
            best_err = err
    return best_b
