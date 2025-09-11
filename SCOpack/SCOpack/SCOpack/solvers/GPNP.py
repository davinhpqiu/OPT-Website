import time
from typing import Callable, Dict, Any, Tuple
import numpy as np


def _fnorm2(v):
    # Squared Frobenius/Euclidean norm for any shape
    v = np.asarray(v)
    return float(np.dot(v.ravel(), v.ravel()))


def _maxk_abs(v: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.abs(v)
    if k >= a.size:
        idx = np.argsort(-a)
    else:
        part = np.argpartition(-a, k - 1)[:k]
        idx = part[np.argsort(-a[part])]
    return v[idx], idx


def my_cg(fx, b, cgtol, cgit, x):
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


def GPNP(func: Callable, n: int, s: int, pars: Dict[str, Any] = None):
    if pars is None:
        pars = {}

    def Funcfg(x):
        return func(x, 'fg', None, None)

    def FuncH(x, T, J):
        return func(x, 'h', T, J)

    def set_parameters(n, s, pars):
        sigma = 1e-8
        J = 1
        flag = 1
        alpha0 = 5.0
        gamma = 0.5
        if s / n <= 0.05 and n >= 1e4:
            alpha0 = 1.0
            gamma = 0.1
        if s / n <= 0.05:
            thd = int(np.ceil(np.log2(2 + s) * 50))
        else:
            if n > 1e3:
                thd = 100
            elif n > 500:
                thd = 500
            else:
                thd = int(np.ceil(np.log2(2 + s) * 750))
        disp = pars.get('disp', 1)
        tol = pars.get('tol', 1e-6)
        uppf = pars.get('uppf', -np.inf)
        maxit = int(pars.get('maxit', 1e4))
        return sigma, J, flag, alpha0, gamma, thd, disp, tol, uppf, maxit

    sigma, J, flag, alpha0, gamma, thd, disp, tol, uppf, maxit = set_parameters(n, s, pars)
    x = np.zeros((n,))
    xo = np.zeros((n,))
    fx, gx = Funcfg(x)
    _, Tx = _maxk_abs(gx, s)
    Tx = np.sort(Tx)
    minobj = np.zeros((maxit + 1,))
    minobj[0] = fx
    OBJ = np.zeros((5,))

    t0 = time.time()
    if disp:
        print(' Start to run the solver -- GPNP ')
        print(' -------------------------------------------')
        print('  Iter     Error      Objective       Time ')
        print(' -------------------------------------------')

    for iter in range(1, maxit + 1):
        alpha = alpha0
        for _ in range(J):
            subu, Tu = _maxk_abs(x - alpha * gx, s)
            u = xo.copy()
            u[Tu] = subu
            fu = Funcfg(u)[0]
            if fu < fx - sigma * _fnorm2(u - x):
                break
            alpha = alpha * gamma

        fu, gx = Funcfg(u)
        normg = _fnorm2(gx)
        x = u
        fx = fu

        sT = np.sort(Tu)
        mark = (np.count_nonzero(sT - Tx) == 0)
        Tx = sT
        eps = 1e-4
        if (mark or normg < 1e-4 or alpha0 == 1) and s <= 5e4:
            v = xo.copy()
            H = FuncH(u, Tu, None)
            if s < 200 and not callable(H):
                subv = subu + np.linalg.solve(H, -gx[Tu])
                eps = 1e-8
            else:
                cgit = min(25, 5 * iter)
                subv = subu + my_cg(H, -gx[Tu], 1e-10 * n, cgit, np.zeros((s,)))
            v[Tu] = subv
            fv, gv = Funcfg(v)
            if fv <= fu - sigma * _fnorm2(subu - subv):
                x = v
                fx = fv
                subu = subv
                gx = gv
                normg = _fnorm2(gx)

        error = np.sqrt(_fnorm2(gx[Tu]))
        obj = fx
        OBJ = np.concatenate([OBJ[1:], [obj]])
        if disp and (iter <= 10 or iter % 10 == 0):
            print(f" {iter:4d}     {error:5.2e}    {fx:9.2e}     {time.time()-t0:5.3f}sec")

        maxg = float(np.max(np.abs(gx)))
        minx = float(np.min(np.abs(subu))) if subu.size > 0 else 1.0
        J = 8
        if error ** 2 < tol * 1e3 and normg > 1e-2 and iter < maxit - 10:
            J = int(min(8, max(1, np.ceil(maxg / max(minx, np.finfo(float).eps)) - 1)))

        if 'uppf' in pars and obj <= uppf and flag:
            maxit = int(iter + 100 * s / n)
            flag = 0

        minobj[iter] = min(minobj[iter - 1], fx)
        if fx < minobj[iter - 1]:
            xmin = x.copy()
            fmin = fx

        if iter > thd:
            count = np.std(minobj[iter - thd:iter + 1]) < 1e-10
        else:
            count = False

        stop1 = error < tol and (np.std(OBJ) < eps * (1 + abs(obj)))  # eps from MATLAB
        stop2 = np.sqrt(normg) < tol
        stop3 = fx < uppf
        if iter > 1 and (stop1 or stop2 or stop3 or count):
            if count and 'fmin' in locals() and fmin < fx:
                x = xmin
                fx = fmin
            if disp and not (iter <= 10 or iter % 10 == 0):
                print(f" {iter:4d}     {error:5.2e}    {fx:9.2e}     {time.time()-t0:5.3f}sec")
            break

    out = {
        'sol': x,
        'obj': fx,
        'iter': iter,
        'error': error,
        'time': time.time() - t0,
    }
    if disp:
        print(' -------------------------------------------')
    if normg < 1e-10 and disp:
        print(' A global optimal solution may be found')
        print(f' because of ||gradient|| = {np.sqrt(normg):5.3e}!')
        print(' -------------------------------------------')
    return out
