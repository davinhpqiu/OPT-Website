from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .solvers import GPNP, IIHT, MIRL1, NHTP, NL0R, PSNP


class CSpackError(ValueError):
    """Custom error for CSpack."""


SOLVERS_REQUIRING_S = {'NHTP', 'GPNP', 'IIHT'}


def CSpack(A, At, b, n: int, s, solver: str, pars: Optional[Dict[str, Any]] = None):
    """Python translation of ``CSpack.m``."""

    if solver is None:
        raise CSpackError('Inputs are not enough !!!')
    if pars is None:
        pars = {}

    solver = solver.strip().upper()
    if callable(A):
        if At is None:
            raise CSpackError('A is a function handle. Its transpose is missing!')
        data = {'A': A, 'At': At, 'b': np.asarray(b, dtype=float), 'n': n}
    else:
        A = np.asarray(A, dtype=float)
        data = {'A': A, 'b': np.asarray(b, dtype=float), 'n': n}
        if At is not None:
            data['At'] = At

    if s is not None and s != []:
        s = int(np.ceil(float(s)))
    elif solver in SOLVERS_REQUIRING_S:
        solver = 'PSNP'

    if 'lambda' in pars:
        lam = float(pars['lambda'])
        lam_nl0r = lam
    else:
        if callable(data['A']):
            lam = 0.005 * np.linalg.norm(data['At'](data['b']), ord=np.inf)
        else:
            lam = 0.005 * np.linalg.norm(data['b'] @ data['A'], ord=np.inf)
        lam_nl0r = 10 * lam

    report_opt = bool(pars.get('report_opt', True))

    if solver == 'NL0R':
        func = _make_ls_func(data)
        out = NL0R(func, n, lam_nl0r, pars.copy())
    elif solver == 'PSNP':
        out = PSNP(data, n, lam, pars.copy())
    elif solver == 'MIRL1':
        out = MIRL1(data, n, lam, pars.copy())
    elif solver in SOLVERS_REQUIRING_S:
        if s is None:
            raise CSpackError('Sparsity level is required for this solver.')
        func = _make_ls_func(data)
        solver_map = {'NHTP': NHTP, 'GPNP': GPNP, 'IIHT': IIHT}
        out = solver_map[solver](func, n, s, pars.copy())
    else:
        raise CSpackError(f'Unknown solver {solver}')

    obj = out.get('obj') if isinstance(out, dict) else None
    if obj is not None and obj < 1e-10 and report_opt:
        print(' -------------------------------------')
        print(' A global minimizer may be found')
        print(f" since (1/2)||Ax-b||^2 = {obj:5.2e}")
        print(' -------------------------------------')

    return out


def _make_ls_func(data: Dict[str, Any]):
    A = data['A']
    b = data['b']
    n = data['n']
    is_func = callable(A)

    if is_func:
        At = data['At']

        def func(x, key, T1, T2):
            if key == 'fg':
                Axb = A(x) - b
                obj = 0.5 * float(Axb.T @ Axb)
                grad = At(Axb)
                return obj, grad
            if key == 'h':
                if T1 is None:
                    raise CSpackError('T1 is required for Hessian access')

                def h11(v):
                    z = np.zeros(n)
                    z[T1] = v
                    return At(A(z))[T1]

                if T2 is None:
                    return h11

                def h12(v):
                    z = np.zeros(n)
                    z[T2] = v
                    return At(A(z))[T1]

                return h11, h12
            raise ValueError("key must be 'fg' or 'h'")

    else:
        A = np.asarray(A, dtype=float)

        def func(x, key, T1, T2):
            if key == 'fg':
                if np.count_nonzero(x) >= 0.8 * x.size:
                    Axb = A @ x - b
                else:
                    Tx = np.nonzero(x)[0]
                    Axb = A[:, Tx] @ x[Tx] - b
                obj = 0.5 * float(Axb.T @ Axb)
                grad = A.T @ Axb
                return obj, grad
            if key == 'h':
                if T1 is None:
                    raise CSpackError('T1 is required for Hessian access')
                AT = A[:, T1]
                if AT.size == 0:
                    return np.zeros((0, 0))
                if len(T1) <= 1000 and A.shape[0] <= 5000:
                    H11 = AT.T @ AT
                else:
                    H11 = lambda v: AT.T @ (AT @ v)
                if T2 is None:
                    return H11
                AT2 = A[:, T2]
                H12 = lambda v: AT.T @ (AT2 @ v)
                return H11, H12
            raise ValueError("key must be 'fg' or 'h'")

    return func
