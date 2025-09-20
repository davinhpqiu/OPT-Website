import numpy as np
import time
from typing import Callable, Dict, Optional, Tuple, Union

ArrayLike = np.ndarray
FuncType = Callable[[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]], Union[Tuple[float, np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]], Callable[[np.ndarray], np.ndarray]]]


def NL0R(func: FuncType, n: int, lam: float, pars: Optional[Dict] = None):
    """Python port of NL0R.m."""
    pars = pars or {}
    x, eta, tol, maxit, disp, update, uppf, rate = setparameters(n, pars)
    x = x.astype(float, copy=True).reshape(n)

    Err = np.zeros(maxit)
    Obj = np.zeros(maxit)
    Nzx = np.zeros(maxit)
    TMP = np.zeros(3)

    def call_fg(vec: np.ndarray) -> Tuple[float, np.ndarray]:
        obj, grad = func(vec, 'fg', None, None)
        grad = np.asarray(grad).reshape(-1)
        return float(obj), grad

    def call_h(vec: np.ndarray, T1: np.ndarray, T2: Optional[np.ndarray]):
        return func(vec, 'h', T1, T2)

    obj, g = call_fg(x)
    if np.linalg.norm(g) == 0:
        if disp:
            print('Starting point is a good stationary point, stop !!!')
        return {
            'sol': x,
            'obj': obj,
            'iter': 0,
            'time': 0.0,
            'sparsity': int(np.count_nonzero(x)),
            'Obj': Obj,
        }

    if np.isnan(g).any():
        x = np.zeros(n)
        rind = np.random.randint(n)
        x[rind] = np.random.rand()
        obj, g = call_fg(x)

    t = 0
    xtg = np.abs(x - eta * g)
    while t < 20:
        T = np.flatnonzero(xtg > np.sqrt(2 * eta * lam))
        nT = T.size
        if nT == 0:
            lam /= 1.25
        elif nT > 0.12 * n:
            lam *= 1.25
        else:
            break
        t += 1

    maxlam = (np.max(np.abs(g)) ** 2) / eta / 2
    nx = 0
    nx0_prev = None
    pcgit = 5
    pcgtol = 1e-5
    beta = 0.5
    sigma = 5e-5
    delta = 1e-10
    T0 = np.array([], dtype=int)

    if disp:
        print(' Start to run the solver -- NL0R ')
        print(' --------------------------------------------------------')
        print('  Iter      Error       Objective    Sparsity    Time(sec)')
        print(' --------------------------------------------------------')

    start = time.perf_counter()
    for iter_idx in range(maxit):
        x0 = x.copy()
        xtg = x0 - eta * g

        while True:
            T = np.flatnonzero(np.abs(xtg) > np.sqrt(2 * eta * lam))
            nT = T.size
            if nT > 0:
                break
            lam /= 1.05

        if iter_idx > 0 and 0 <= nT - T0.size <= 5 and Err[iter_idx - 1] < tol:
            lam = lam0
            while True:
                T = np.flatnonzero(np.abs(xtg) > np.sqrt(2 * eta * lam))
                nT = T.size
                if nT > 0:
                    break
                lam /= 1.05

        if iter_idx >= 0 and nT > max(0.12, 0.2 / np.log2(1 + iter_idx + 1)) * n:
            Tnew = SparseApprox(xtg[T], T)
            nTnew = Tnew.size
            if Tnew.size and nT / nTnew < 20 and nT != nTnew:
                T = Tnew
                nT = nTnew

        TTc = np.setdiff1d(T, T0, assume_unique=False)
        flag = TTc.size == 0

        FxT = np.sqrt(np.linalg.norm(g[T]) ** 2 + np.linalg.norm(x[TTc]) ** 2)
        Err[iter_idx] = FxT / np.sqrt(n)
        Nzx[iter_idx] = nx
        if disp and ((iter_idx + 1) % 10 == 0 or iter_idx + 1 < 100):
            print(f" {iter_idx + 1:4d}      {FxT:8.2e}     {obj:9.2e}      {nx:4d}      {time.perf_counter() - start:5.3f}sec")

        TMP = np.roll(TMP, -1)
        TMP[-1] = obj
        stop1 = Err[iter_idx] < tol and np.std(TMP) < 1e-8 * (1 + abs(obj))
        stop1 = stop1 and nx == nT and flag
        stop2 = (iter_idx + 1) > 3 and obj < uppf and nx <= int(np.ceil(n / 4))
        stop3 = np.linalg.norm(g) < tol and nx <= int(np.ceil(n / 4))
        if iter_idx > 0 and (stop1 or stop2 or stop3):
            if disp and not ((iter_idx + 1) % 10 == 0 or iter_idx + 1 < 100):
                print(f" {iter_idx + 1:4d}      {FxT:8.2e}     {obj:9.2e}      {nx:4d}      {time.perf_counter() - start:5.3f}sec")
            break

        if iter_idx == 0 or flag:
            H = call_h(x0, T, None)
            if callable(H):
                d = my_cg(H, -g[T], pcgtol, pcgit, np.zeros(nT))
            else:
                d = np.linalg.solve(H, -g[T])

            dg = np.dot(d, g[T])
            ngT = np.linalg.norm(g[T]) ** 2
            if dg > max(-delta * np.linalg.norm(d) ** 2, -ngT) or np.isnan(dg):
                d = -g[T]
                dg = ngT
        else:
            res = call_h(x0, T, TTc)
            H, D = res if isinstance(res, tuple) else (res, None)
            rhs = (D(x0[TTc]) if callable(D) else D @ x0[TTc]) - g[T]
            if callable(H):
                d = my_cg(H, rhs, pcgtol, pcgit, np.zeros(nT))
            else:
                d = np.linalg.solve(H, rhs)

            Fnz = np.linalg.norm(x[TTc]) ** 2 / (4 * eta)
            dgT = np.dot(d, g[T])
            dg = dgT - np.dot(x0[TTc], g[TTc])
            delta0 = 1e-4 if Fnz > 1e-4 else delta
            ngT = np.linalg.norm(g[T]) ** 2
            if dgT > max(-delta0 * np.linalg.norm(d) ** 2 + Fnz, -ngT) or np.isnan(dg):
                d = -g[T]
                dg = ngT

        alpha = 1.0
        obj0 = obj
        x_candidate = np.zeros_like(x)
        for _ in range(6):
            x_candidate.fill(0.0)
            x_candidate[T] = x0[T] + alpha * d
            obj_candidate, _ = call_fg(x_candidate)
            if obj_candidate < obj0 + alpha * sigma * dg:
                obj = obj_candidate
                x = x_candidate.copy()
                break
            alpha *= beta
        else:
            x = x_candidate.copy()
            obj, _ = call_fg(x)

        T0 = T.copy()
        obj, g = call_fg(x)
        Obj[iter_idx] = obj

        if (iter_idx + 1) % 10 == 0:
            recent = Obj[max(0, iter_idx - 9):iter_idx + 1]
            if Err[iter_idx] > 1 / (iter_idx + 1) ** 2 or np.sum(recent[1:] > 1.5 * recent[:-1]) >= 2:
                eta = eta / (1.25 if iter_idx + 1 < 1500 else 1.5)
            else:
                eta = eta * 1.25

        nx = int(np.count_nonzero(x))
        if (iter_idx + 1) > 5:
            prev_vals = Nzx[:iter_idx]
            prev_max = np.max(prev_vals) if prev_vals.size else 0
        else:
            prev_max = 0

        if (iter_idx + 1) > 5 and nx > 2 * prev_max and Err[iter_idx] < 1e-2:
            rate0 = 2 / rate
            x = x0
            nx = int(np.count_nonzero(x0))
            nx0_prev = Nzx[iter_idx - 1] if iter_idx > 0 else None
            obj, g = call_fg(x)
            rate = 1.1
        else:
            rate0 = rate

        if nx0_prev is not None and nx < nx0_prev:
            rate0 = 1

        lam0 = lam
        if update:
            lam = min(maxlam, lam * (2 * (nx >= 0.1 * n) + rate0))

    final_obj, g = call_fg(x)
    elapsed = time.perf_counter() - start
    out = {
        'sol': x,
        'obj': final_obj,
        'iter': iter_idx + 1,
        'time': elapsed,
        'sparsity': int(np.count_nonzero(x)),
        'Obj': Obj,
    }

    if disp:
        print(' --------------------------------------------------------')
        normgrad = np.linalg.norm(g)
        if normgrad < 1e-6:
            print(' A global optimal solution might be found')
            print(f' because of ||gradient|| = {normgrad:5.2e}!')
        print(' --------------------------------------------------------')

    return out


def setparameters(n: int, pars: Dict):
    if n <= 1e3:
        rate0 = 0.5
    else:
        rate0 = 1.0 / np.exp(3 / np.log10(n))

    x0 = np.asarray(pars.get('x0', np.zeros(n))).reshape(-1)
    eta = float(pars.get('eta', 1))
    rate = float(pars.get('rate', rate0))
    disp = int(pars.get('disp', 1))
    maxit = int(pars.get('maxit', 2000))
    uppf = float(pars.get('uppf', -np.inf))
    tol = float(pars.get('tol', 1e-6))
    update = int(pars.get('update', 1))

    return x0, eta, tol, maxit, disp, update, uppf, rate


def SparseApprox(x0: np.ndarray, T0: np.ndarray) -> np.ndarray:
    x = np.abs(x0)
    nz = x[x != 0]
    if nz.size == 0:
        return T0
    sx = np.sort(nz)
    if sx.size <= 2:
        th = sx[-1]
    else:
        ratios = sx[1:] / sx[:-1]
        if ratios.size == 0:
            th = sx[-1]
        else:
            mean = ratios.mean()
            std = ratios.std(ddof=0)
            std = std if std != 0 else 1.0
            scores = (ratios - mean) / std
            idx = int(np.argmax(scores))
            th = 0.0
            if scores[idx] > 10 and idx >= 1:
                th = sx[idx]
            else:
                th = sx[-1]
    mask = x > th
    return T0[mask]


def my_cg(fx: Union[Callable[[np.ndarray], np.ndarray], np.ndarray], b: np.ndarray, cgtol: float, cgit: int, x0: np.ndarray) -> np.ndarray:
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        return np.zeros_like(x0)

    if not callable(fx):
        mat = np.asarray(fx)
        fx = lambda v: mat @ v

    x = x0.copy()
    r = b - fx(x) if np.count_nonzero(x) else b.copy()
    e = np.linalg.norm(r) ** 2
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        w = fx(p)
        pw = np.dot(p, w)
        if pw == 0:
            break
        a = e / pw
        x = x + a * p
        r = r - a * w
        e0 = e
        e = np.linalg.norm(r) ** 2
        if e0 == 0:
            break
        p = r + (e / e0) * p
    return x
