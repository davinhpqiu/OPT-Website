import time
from typing import Callable, Dict, Any
import numpy as np


def _fnorm2(v):
    v = np.asarray(v)
    return float(np.dot(v.ravel(), v.ravel()))


def _maxk(v: np.ndarray, k: int, by_abs: bool = False):
    if by_abs:
        a = np.abs(v)
    else:
        a = v
    if k >= a.size:
        idx = np.argsort(-a)
    else:
        part = np.argpartition(-a, k - 1)[:k]
        idx = part[np.argsort(-a[part])]
    return v[idx], idx


def IIHT(func: Callable, n: int, s: int, pars: Dict[str, Any] = None):
    """
    Python port of IIHT.m
    """
    if pars is None:
        pars = {}
    disp = pars.get('disp', 1)
    maxit = int(pars.get('maxit', 5e3))
    tol = pars.get('tol', 1e-6 * np.sqrt(n))
    neg = pars.get('neg', 0)
    uppf = pars.get('uppf', -np.inf)

    def Func(x):
        return func(x, 'fg', None, None)

    t0 = time.time()
    x = np.zeros((n,))
    xo = np.zeros((n,))
    OBJ = np.zeros((5,))
    sigma0 = 1e-4

    if disp:
        print(' Start to run the sover -- IIHT ')
        print(' -------------------------------------------')
        print('  Iter     Error      Objective       Time ')
        print(' -------------------------------------------')

    f, g = Func(x)
    scale = (max(f, np.linalg.norm(g)) > n)
    scal = n if scale else 1
    fs = f / scal
    gs = g / scal

    error = None
    normg = np.linalg.norm(g)
    for iter in range(1, maxit + 1):
        x_old = x.copy()
        fx_old = fs
        alpha = np.log1p(iter)
        for _ in range(10):
            tp = x_old - alpha * gs
            if neg:
                tp = np.maximum(0.0, tp)
                mx, T = _maxk(tp, s, by_abs=False)
            else:
                mx, T = _maxk(tp, s, by_abs=True)
            x = xo.copy()
            x[T] = mx
            fs = Func(x)[0] / scal
            if fs < fx_old - 0.5 * sigma0 * _fnorm2(x - x_old):
                break
            alpha = alpha / 2.0

        f, g = Func(x)
        fs = f / scal
        gs = g / scal
        OBJ = np.concatenate([OBJ[1:], [fs]])
        error = scal * np.linalg.norm(gs[T]) / max(1.0, np.linalg.norm(mx))
        normg = np.linalg.norm(g)
        if disp and (iter <= 10 or iter % 10 == 0):
            print(f" {iter:4d}     {error:5.2e}    {fs*scal:9.2e}     {time.time()-t0:5.3f}sec")

        stop1 = error < tol and (np.std(OBJ) < 1e-8 * (1 + abs(fs)))
        stop2 = normg < tol
        stop3 = fs * scal < uppf
        if iter > 1 and (stop1 or stop2 or stop3):
            if disp and not (iter <= 10 or iter % 10 == 0):
                print(f" {iter:4d}     {error:5.2e}    {fs*scal:9.2e}     {time.time()-t0:5.3f}sec")
            break

    if disp:
        print(' -------------------------------------------')

    out = {
        'sol': x,
        'obj': fs * scal,
        'iter': iter,
        'time': time.time() - t0,
        'error': error,
    }
    if normg < 1e-5 and disp:
        print(' A global optimal solution might be found')
        print(f' because of ||gradient||={normg:5.2e}!')
        print(' -------------------------------------------')
    return out
