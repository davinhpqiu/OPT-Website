import time
from typing import Callable, Dict, Any
import numpy as np


def _fnorm2(v):
    v = np.asarray(v)
    return float(np.dot(v.ravel(), v.ravel()))


def _maxk_abs(v: np.ndarray, k: int):
    a = np.abs(v)
    if k >= a.size:
        idx = np.argsort(-a)
    else:
        part = np.argpartition(-a, k - 1)[:k]
        idx = part[np.argsort(-a[part])]
    return v[idx], idx


def my_cg(fx, b, cgtol, cgit, x):
    # fx can be ndarray (matrix) or function handle
    def apply_fx(v):
        if callable(fx):
            return fx(v)
        else:
            return fx @ v

    if np.linalg.norm(b) == 0:
        return np.zeros_like(x)
    r = b.copy()
    if np.count_nonzero(x) > 0:
        r = b - apply_fx(x)
    e = float(np.linalg.norm(r) ** 2)
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        w = apply_fx(p)
        pw = p * w
        a = e / float(np.sum(pw))
        x = x + a * p
        r = r - a * w
        e0 = e
        e = float(np.linalg.norm(r) ** 2)
        p = r + (e / e0) * p
    return x


def NHTP(func: Callable, n: int, s: int, pars: Dict[str, Any] = None):
    if pars is None:
        pars = {}

    def Funcfg(x):
        return func(x, 'fg', None, None)

    def FuncH(x, T, J):
        return func(x, 'h', T, J)

    def getparameters(n, s, funcfg, pars):
        disp = pars.get('disp', 1)
        maxit = int(pars.get('maxit', 2000))
        tol = pars.get('tol', 1e-6)
        x0 = pars.get('x0', np.zeros((n,)))
        uppf = pars.get('uppf', -np.inf)
        if 'eta' in pars:
            eta = pars['eta']
        else:
            _, g1 = funcfg(np.ones((n,)))
            abg1 = np.abs(g1)
            T = np.flatnonzero(abg1 > 1e-8)
            if T.size == 0:
                eta = 10 * (1 + s / n) / min(10, np.log(n))
            else:
                maxe = np.sum(1.0 / (abg1[T] + np.finfo(float).eps)) / T.size
                if maxe > 2:
                    eta = (np.log2(1 + maxe) / np.log2(maxe)) * np.exp((s / n) ** (1 / 3))
                elif maxe < 1:
                    eta = (np.log2(1 + maxe)) * (n / s) ** 0.5
                else:
                    eta = (np.log2(1 + maxe)) * np.exp((s / n) ** (1 / 3))
        return x0, eta, disp, maxit, tol, uppf

    x0, eta, disp, maxit, tol, uppf = getparameters(n, s, Funcfg, pars)
    x = x0.copy()
    beta = 0.5
    sigma = 5e-5
    delta = 1e-10
    pcgtol = 0.1 * tol * s

    T0 = np.array([], dtype=int)
    Error = np.zeros((maxit,))
    Obj = np.zeros((maxit,))
    OBJ = np.zeros((5,))
    xo = np.zeros((n,))
    obj, g = Funcfg(x0)

    t0 = time.time()
    if disp:
        print(' Start to run the solver -- NHTP ')
        print(' -------------------------------------------')
        print('  Iter     Error      Objective       Time ')
        print(' -------------------------------------------')

    if _fnorm2(g) < 1e-20 and np.count_nonzero(x) <= s:
        print(' Starting point is a good solution. Stop NHTP')
        return {'sol': x, 'obj': obj, 'time': time.time() - t0, 'iter': 0}

    if np.isnan(g).any():
        x0 = np.zeros((n,))
        rind = np.random.randint(n)
        x0[rind] = np.random.rand()
        obj, g = Funcfg(x0)

    for iter in range(1, maxit + 1):
        xtg = x0 - eta * g
        _, T = _maxk_abs(x0 - eta * g, s)
        T = np.sort(T)
        TTc = np.setdiff1d(T0, T)
        flag = TTc.size == 0
        gT = g[T]

        xtaus = max(0.0, float(np.max(np.abs(g)) - np.min(np.abs(x[T])) / eta)) if T.size > 0 else float(np.max(np.abs(g)))
        if flag:
            FxT = np.sqrt(_fnorm2(gT))
            Error[iter - 1] = xtaus + FxT
        else:
            FxT = np.sqrt(_fnorm2(gT) + abs(_fnorm2(x) - _fnorm2(x[T])))
            Error[iter - 1] = xtaus + FxT

        if disp and (iter <= 10 or iter % 10 == 0):
            print(f" {iter:4d}     {Error[iter-1]:5.2e}    {obj:9.2e}     {time.time()-t0:5.3f}sec")

        OBJ = np.concatenate([OBJ[1:], [obj]])
        stop1 = Error[iter - 1] < tol and (np.std(OBJ) < 1e-8 * (1 + abs(obj)))
        stop2 = np.sqrt(_fnorm2(g)) < tol
        stop3 = obj < uppf
        if iter > 1 and (stop1 or stop2 or stop3):
            if disp and not (iter <= 10 or iter % 10 == 0):
                print(f" {iter:4d}     {Error[iter-1]:5.2e}    {obj:9.2e}     {time.time()-t0:5.3f}sec")
            break

        # Update next iterate
        if iter == 1 or flag:
            H = FuncH(x0, T, None)
            if callable(H):
                d = my_cg(H, -gT, pcgtol, 25, np.zeros((s,)))
            else:
                d = np.linalg.solve(H, -gT)
            dg = float(np.sum(d * gT))
            ngT = _fnorm2(gT)
            if dg > max(-delta * _fnorm2(d), -ngT) or np.isnan(dg):
                d = -gT
                dg = ngT
        else:
            H, D = FuncH(x0, T, TTc)
            if callable(D):
                rhs = D(x0[TTc]) - gT
            else:
                rhs = D @ x0[TTc] - gT
            if callable(H):
                d = my_cg(H, rhs, pcgtol, 25, np.zeros((s,)))
            else:
                d = np.linalg.solve(H, rhs)
            Fnz = _fnorm2(x[TTc]) / 4.0 / eta if TTc.size > 0 else 0.0
            dgT = float(np.sum(d * gT))
            dg = dgT - float(np.sum(x0[TTc] * g[TTc]))
            delta0 = delta
            if Fnz > 1e-4:
                delta0 = 1e-4
            ngT = _fnorm2(gT)
            if dgT > max(-delta0 * _fnorm2(d) + Fnz, -ngT) or np.isnan(dg):
                d = -gT
                dg = ngT

        alpha = 1.0
        x = xo.copy()
        obj0 = obj
        Obj[iter - 1] = obj
        for _ in range(6):
            x[T] = x0[T] + alpha * d
            obj = Funcfg(x)[0]
            if obj < obj0 + alpha * sigma * dg:
                break
            alpha = beta * alpha

        fhtp = 0
        if obj > obj0:
            x[T] = xtg[T]
            obj = Funcfg(x)[0]
            fhtp = 1

        flag1 = (abs(obj - obj0) < 1e-6 * (1 + abs(obj)) and fhtp)
        flag2 = (abs(obj - obj0) < 1e-8 * (1 + abs(obj)) and Error[iter - 1] < 1e-2)
        if iter > 10 and (flag1 or flag2):
            if obj > obj0:
                iter = iter - 1
                x = x0
                T = T0
            print(f" {iter:4d}     {Error[iter-1]:5.2e}    {obj:9.2e}     {time.time()-t0:5.3f}sec")
            break

        T0 = T
        x0 = x.copy()
        obj, g = Funcfg(x)

        if iter % 50 == 0:
            if Error[iter - 1] > 1.0 / (iter ** 2):
                if iter < 1500:
                    eta = eta / 1.25
                else:
                    eta = eta / 1.5
            else:
                eta = eta * 1.15

    out = {
        'time': time.time() - t0,
        'iter': iter,
        'sol': x,
        'obj': obj,
    }
    if disp:
        normgrad = np.sqrt(_fnorm2(g))
        print(' -------------------------------------------')
        if normgrad < 1e-5:
            print(' A global optimal solution might be found')
            print(f' because of ||gradient|| = {normgrad:5.2e}!')
            if out['iter'] > 1500:
                print(f"\n Since the number of iterations reaches to {out['iter']}")
                print(' Try to rerun the solver with setting a smaller pars.eta ')
            print(' -------------------------------------------')
    return out
