# This Python translation of the NM01 solver preserves structure, comments, and variable names.
# It assumes availability of helper functions like `GetParameters`, `Ttau`, and `my_cg` defined below.

import numpy as np
import time


def NM01(func, B, b, lam, pars):
    # -------------------------------------------------------------------------
    # This code aims at solving the support vector machine with form
    #
    #      min  f(x) + lam * ||(Bx+b)_+||_0
    #
    # where f is twice continuously differentiable
    # lam > 0, B\in\R^{m x n}, b\in\R^{m x 1}
    # (z)_+ = (max{0,z_1},...,max{0,z_m})^T
    # ||(z)_+ ||_0 counts the number of positive entries of z
    # -------------------------------------------------------------------------
    # Inputs:
    #   func: A function handle defines (objective,gradient,Hessain) (REQUIRED)
    #   B   : A matrix \R^{m x n}                                    (REQUIRED)      
    #   b   : A vector \R^{m x 1}                                    (REQUIRED)
    #   lam : The penalty parameter                                  (REQUIRED)
    #   pars: Parameters are all OPTIONAL
    #         pars.x0     -- The initial point             (default zeros(n,1))
    #         pars.tau    -- A useful paramter                   (default 1.00)
    #         pars.mu0    -- A smoothing parameter               (default 0.01)
    #         pars.maxit  -- Maximum number of iterations        (default 1000)  
    #         pars.tol    -- Tolerance of halting conditions   (1e-7*sqrt(n*m)) 
    #         pars.strict -- = 0, loosely meets halting conditions  (default 0)
    #                        = 1, strictly meets halting conditions  
    #                        pars.strict=1 is useful for low dimensions  
    # -------------------------------------------------------------------------
    # Outputs:
    #   out.sol:  The solution 
    #   out.obj:  The objective function value
    #   out.time: CPU time
    #   out.iter: Number of iterations
    # -------------------------------------------------------------------------
    # Send your comments and suggestions to <<< slzhou2021@163.com >>>    
    # WARNING: Accuracy may not be guaranteed!!!!!  
    # -------------------------------------------------------------------------
    if any(arg is None for arg in [func, B, b, lam]):
        print(" Inputs are not enough !!!")
        return None
    
    if pars is None:
        pars = {}

    t0 = time.time()
    m, n = B.shape
    # Ensure b is a 1-D vector to avoid broadcasting to (m, m)
    if b is not None:
        b = np.asarray(b).reshape(-1)
    maxit, tau, mu, x, tol, tolcg, strict = GetParameters(m, n, pars)

   # Fnorm = lambda var: np.linalg.norm(var, 'fro') ** 2
    Fnorm = lambda var: np.sum(var**2)
    obje = lambda var: func(var, 'f')
    grad = lambda var: func(var, 'g')
    Hess = lambda var: func(var, 'h')
    fBxb = lambda var: B @ var + b

    accu = lambda var: func(var, 'a')
    if accu(x) is None:
        accu = lambda var: (1 - np.count_nonzero(fBxb(var) > 0) / m)

    z = np.ones(m)
    u = x
    Bxb = fBxb(x)
    Bxz = Bxb + tau * z
    lam = max(1 / (2 * tau), lam)
    Acc = np.zeros(maxit)
    Obj = np.zeros(maxit)
    T = None

    #lam = Initialam(m,n,Axz,tau);

    print(" Start to run the solver -- NM01")
    print(" -----------------------------------------------------------")
    print("  Iter     Accuracy     Objective      Error       Time(sec) ")
    print(" -----------------------------------------------------------")

    for iter in range(maxit):
        T0 = T
        T, empT, lam = Ttau(Bxz, Bxb, tau, lam)

        if iter > 3 and np.std(Acc[iter - 3:iter]) < 1e-8:
            T = np.unique(np.union1d(T0, T))

        nT = np.count_nonzero(T)
        if np.count_nonzero(T0) == nT and np.count_nonzero(T0 - T) == 0:
            flag = 0
        else:
            flag = 1

        g = grad(x)
        if empT:
            error = Fnorm(g) + Fnorm(z)
        else:
            if flag:
                BT = B[T, :]
                fBT = lambda var: BT @ var
                fBTt = lambda var: BT.T @ var
            zT = z[T]
            rhs1 = g + fBTt(zT)
            rhs2 = Bxb[T]
            error = Fnorm(rhs1) + Fnorm(rhs2) + Fnorm(z) - Fnorm(z[T])

        error = error / np.sqrt(n * m)
        Acc[iter] = accu(x) * 100
        Obj[iter] = obje(x)

        if iter < 10 or iter % 10 == 0:
            print(f"  {iter:3d}     {Acc[iter]:8.3f}      {Obj[iter]:8.4f}     {error:8.3e}      {time.time() - t0:.3f}sec")

        stop = 0
        if iter > 5 and strict:
            stop = max(error, Fnorm(u)) < tol
        elif iter > 5 and not strict:
            stop1 = min(error, Fnorm(u)) < tol
            stop2 = np.std(Acc[iter - 5:iter + 1]) < 1e-10
            stop3 = np.std(Obj[iter - 5:iter + 1]) < 1e-4 * (1 + np.sqrt(abs(Obj[iter])))
            stop = stop1 or (stop2 and stop3)

        if stop:
            if iter > 10 and iter % 10 != 0:
                print(f"  {iter:3d}     {Acc[iter]:8.3f}      {Obj[iter]:8.4f}     {error:8.3e}      {time.time() - t0:.3f}sec")
            break 

        if empT:
            u = -g
            v = -z
        else:
            H = Hess(x)
            if not callable(H) and n <= 1e3 and nT <= 1e4:
                u = np.linalg.solve(BT.T @ BT + mu * H, -mu * rhs1 - fBTt(rhs2))
                v = -z
                v[T] = (fBT(u) + rhs2) / mu
            elif not callable(H) and n > 1e3 and np.allclose(H, np.diag(np.diag(H))) and n <= 5e3:
                invH = np.diag(H)
                invH[np.abs(invH) < 1e-8] = 1e-4 / iter
                invH = 1.0 / invH
                D = BT @ (invH[:, None] * BT.T)
                D[np.diag_indices_from(D)] += mu
                vT = np.linalg.solve(D, rhs2 - fBT(invH * rhs1))
                v = -z
                v[T] = vT
                u = -invH * (rhs1 + fBTt(vT))
            else:
                if callable(H):
                    fx = (lambda var: fBTt(fBT(var)) + mu * H(var))
                else:
                    fx = (lambda var: fBTt(fBT(var)) + mu * (H @ var))
                u = my_cg(fx, -mu * rhs1 - fBTt(rhs2), tolcg, 20, np.zeros(n))
                v = -z
                v[T] = (fBT(u) + rhs2) / mu

        alpha = 1
        x0 = x.copy()
        z0 = z.copy()
        obj0 = obje(x)
        for _ in range(4):
            x = x0 + alpha * u
            if obje(x) < obj0:
                break
            alpha *= 0.8
        z = z0 + alpha * v
        Bxb = fBxb(x)
        if iter % 5 == 0:
            mu = max(1e-8, mu / 1.1)
            tau = max(1e-4, tau / 1.1)
            lam = lam * 1.1
        Bxz = Bxb + tau * z

    print(" -----------------------------------------------------------")
    out = {
        'sol': x,
        'obj': obje(x),
        'time': time.time() - t0,
        'iter': iter
    }
    return out

#--------------------------------------------------------------------------
def GetParameters(m, n, pars):
    maxit = int(1e3)
    mn = m * n
    tolcg = 1e-8 * np.sqrt(mn)
    tol = 1e-7 * np.sqrt(mn)
    tau = 1.00
    mu = 0.01
    x0 = np.zeros(n)
    strict = 0
    if 'maxit' in pars: maxit = pars['maxit']
    if 'tau' in pars: tau = pars['tau']
    if 'x0' in pars: x0 = pars['x0']
    if 'tol' in pars: tol = pars['tol']
    if 'mu0' in pars: mu = pars['mu0']
    if 'strict' in pars: strict = pars['strict']
    return maxit, tau, mu, x0, tol, tolcg, strict

#select the index set T
def Ttau(Bxz, Bxb, tau, lam):
    tl = np.sqrt(tau * lam / 2)
    T = np.where(np.abs(Bxz - tl) <= tl)[0]
    empT = len(T) == 0
    if empT:
        zp = Bxb[Bxb >= 0]
        if zp.size > 0:
            # MATLAB: s = ceil(0.01*nnz(zp)); use 1-based index -> Python 0-based
            s = max(0, min(zp.size - 1, int(np.ceil(0.01 * zp.size)) - 1))
            tau = (zp[s]) ** 2 / (2 * lam)
            tl = np.sqrt(tau * lam / 2)
            T = np.where(np.abs(Bxb - tl) < tl)[0]
        empT = len(T) == 0
    return T, empT, lam

# Set intial lam
def Initialam(m, n, z, tau):
    zp = z[z > 0]
    if zp.size == 0:
        return 5
    s = min(m, 20 * n, np.count_nonzero(zp)) - 1  # -1 for Python indexing
    zp_sorted = np.sort(zp)
    return max(5, max(zp_sorted[s]**2, 1) / (2 * tau))

# conjugate gradient
def my_cg(fx, b, cgtol, cgit, x):
    #if np.linalg.norm(b, 'fro') == 0:
    if np.linalg.norm(b) == 0:
        return np.zeros_like(x)
    if not callable(fx):
        fx = lambda v: fx @ v
    r = b.copy()
    if np.count_nonzero(x) > 0:
        r = b - fx(x)
    #e = np.linalg.norm(r, 'fro') ** 2
    e = np.linalg.norm(r) ** 2
    t = e
    p = r.copy()
    for _ in range(cgit):
        if e < cgtol * t:
            break
        w = fx(p)
        pw = p * w
        a = e / np.sum(pw)
        x = x + a * p
        r = r - a * w
        e0 = e
        #e = np.linalg.norm(r, 'fro') ** 2
        e = np.linalg.norm(r) ** 2
        p = r + (e / e0) * p
    return x
