import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray


class SNSQPResult(dict):
    """Result container matching the MATLAB struct output."""


def snsqp(
    n: int,
    s: int,
    Q0: ArrayLike,
    q0: ArrayLike,
    Qi: Optional[Sequence[ArrayLike]] = None,
    qi: Optional[ArrayLike] = None,
    ci: Optional[ArrayLike] = None,
    ineqA: Optional[ArrayLike] = None,
    ineqb: Optional[ArrayLike] = None,
    eqA: Optional[ArrayLike] = None,
    eqb: Optional[ArrayLike] = None,
    lb: Optional[ArrayLike] = None,
    ub: Optional[ArrayLike] = None,
    pars: Optional[Dict[str, Any]] = None,
) -> SNSQPResult:
    """Python translation of ``SNSQP.m`` for sparse QCQP."""

    pars = dict(pars or {})
    Q0 = np.asarray(Q0, dtype=float)
    q0 = np.asarray(q0, dtype=float).reshape(n)

    Qi_list = [np.asarray(mat, dtype=float) for mat in (Qi or [])]
    dim0 = len(Qi_list)

    if qi is None:
        qi_mat = np.zeros((n, dim0))
    else:
        qi_mat = np.asarray(qi, dtype=float)
        if qi_mat.ndim == 1:
            qi_mat = qi_mat.reshape(n, 1)

    if ci is None:
        ci_vec = np.zeros(dim0)
    else:
        ci_vec = np.asarray(ci, dtype=float).reshape(dim0)

    ineqA = _ensure_matrix(ineqA, n)
    ineqb = _ensure_vector(ineqb, ineqA.shape[0])
    eqA = _ensure_matrix(eqA, n)
    eqb = _ensure_vector(eqb, eqA.shape[0])

    lb_vec = _ensure_bounds(lb, n, -np.inf)
    ub_vec = _ensure_bounds(ub, n, np.inf)

    (
        dim0,
        dim1,
        dim2,
        existcons,
        flagbd,
        lenf,
        show,
        x,
        dualquad,
        dualineq,
        dualeq,
        dualbd,
        tau,
        tol,
        itmax,
        itlser,
        gamma,
        sigma,
        alpha0,
        lb_vec,
        ub_vec,
    ) = _set_parameters(n, s, Qi_list, ineqA, eqA, lb_vec, ub_vec, pars)

    x = x.astype(float, copy=True).reshape(n)
    Index = np.arange(n)
    tau0 = float(tau)

    T0 = _topk_indices(np.abs(x), s)
    xT = x[T0]
    obj = _func_obj(xT, Q0, q0, T0)

    if existcons[0]:
        Qxq = _func_qxq(xT, Qi_list, qi_mat, dim0, T0)
        Qqc = _func_qxqc(xT, Qi_list, qi_mat, ci_vec, dim0, T0)
        Ncpqual = _func_ncp_quad(Qqc, dualquad)
    else:
        Qxq = np.zeros((n, 0))
        Qqc = np.zeros(0)
        Ncpqual = np.zeros(0)

    if existcons[1]:
        Axb = _func_axb(xT, ineqA, ineqb, T0)
        Ncpineq = _func_ncp_ineq(Axb, dualineq)
    else:
        Axb = np.zeros(0)
        Ncpineq = np.zeros(0)

    if existcons[2]:
        Lineq = _func_axb(xT, eqA, eqb, T0)
    else:
        Lineq = np.zeros(0)

    if existcons[3]:
        dualbdT = dualbd[T0]
        xPT = xT - _proj_box(xT + dualbdT, lb_vec[T0], ub_vec[T0])
    else:
        dualbdT = np.zeros(0)
        xPT = np.zeros(0)

    GradL = _grad_lagrangian(
        xT,
        Q0,
        q0,
        Qxq,
        ineqA,
        eqA,
        dualquad,
        dualineq,
        dualeq,
        T0,
        existcons,
    )
    if existcons[3]:
        GradL[T0] += dualbdT

    Indx = np.arange(s)
    Indqual = np.arange(s, s + dim0)
    Indineq = np.arange(s + dim0, s + dim0 + dim1)
    Indeq = np.arange(s + dim0 + dim1, s + dim0 + dim1 + dim2)
    offset_bnd = s + dim0 + dim1 + dim2

    z = np.zeros(n)
    nz_prev = 0
    Err = float('inf')

    if show:
        print("\n Start to run the sover -- SNSQP")
        print(" -------------------------------------------------")
        print(" Iter        Error        Objective      Time(sec)")
        print(" -------------------------------------------------")

    start = time.perf_counter()

    for it in range(1, itmax + 1):
        xt = x.copy()
        T = _topk_indices(np.abs(xt - tau * GradL), s)
        Tc = np.setdiff1d(Index, T, assume_unique=False)
        TTc = np.setdiff1d(T0, T, assume_unique=False)
        flagT = TTc.size == 0

        if existcons[3] and not flagT:
            xPT = x[T] - _proj_box(x[T] + dualbd[T], lb_vec[T], ub_vec[T])

        StationEq = _concat(GradL[T], Ncpqual, Ncpineq, Lineq, xPT)
        Err = np.linalg.norm(StationEq) + np.linalg.norm(x[Tc])

        if show:
            elapsed = time.perf_counter() - start
            print(f"{it:4d}       {Err:8.2e}      {obj:12.3e}    {elapsed:7.3f}sec")

        if Err <= tol:
            break

        HessTT = _hess_lagrangian_tt(Q0, Qi_list, dualquad, T, dim0)

        if existcons[0]:
            QxqT = Qxq[T, :]
            JPquad, JDquad = _jac_ncp_quad(Qqc, dualquad, dim0)
            Hquad = (JPquad[:, None] * QxqT.T)
        else:
            QxqT = np.zeros((T.size, 0))
            JPquad = np.zeros(0)
            JDquad = np.zeros(0)
            Hquad = np.zeros((0, T.size))

        if existcons[1]:
            IneqAT = ineqA[:, T]
            JPineq, JDineq = _jac_ncp_ineq(Axb, dualineq, dim1)
            Hineq = (JPineq[:, None] * IneqAT)
        else:
            IneqAT = np.zeros((0, T.size))
            JPineq = np.zeros(0)
            JDineq = np.zeros(0)
            Hineq = np.zeros((0, T.size))

        if existcons[2]:
            EqAT = eqA[:, T]
        else:
            EqAT = np.zeros((0, T.size))

        if existcons[3]:
            if flagT:
                projbd = _jac_proj_box(xT + dualbdT, lb_vec[T0], ub_vec[T0])
            else:
                projbd = _jac_proj_box(x[T] + dualbd[T], lb_vec[T], ub_vec[T])
            U = np.diag(projbd)
            eU = np.diag(1 - projbd)
            sbd = s
            eye_bd = np.eye(s)
        else:
            U = np.zeros((0, 0))
            eU = np.zeros((0, 0))
            sbd = 0
            eye_bd = np.zeros((T.size, 0))

        row1 = [
            HessTT,
            QxqT if dim0 else np.zeros((T.size, 0)),
            IneqAT.T if dim1 else np.zeros((T.size, 0)),
            EqAT.T if dim2 else np.zeros((T.size, 0)),
            eye_bd,
        ]
        rows = [row1]

        if dim0:
            rows.append(
                [
                    Hquad,
                    np.diag(JDquad),
                    np.zeros((dim0, dim1)),
                    np.zeros((dim0, dim2)),
                    np.zeros((dim0, sbd)),
                ]
            )
        if dim1:
            rows.append(
                [
                    Hineq,
                    np.zeros((dim1, dim0)),
                    np.diag(JDineq),
                    np.zeros((dim1, dim2)),
                    np.zeros((dim1, sbd)),
                ]
            )
        if dim2:
            rows.append(
                [
                    EqAT,
                    np.zeros((dim2, dim0)),
                    np.zeros((dim2, dim1)),
                    np.zeros((dim2, dim2)),
                    np.zeros((dim2, sbd)),
                ]
            )
        if sbd:
            rows.append(
                [
                    eU,
                    np.zeros((sbd, dim0)),
                    np.zeros((sbd, dim1)),
                    np.zeros((sbd, dim2)),
                    -U,
                ]
            )

        HessL = np.block(rows) if rows else np.zeros((0, 0))
        lenf_curr = HessL.shape[0]

        if it == 1 or flagT or flagbd:
            STEq = -StationEq
        else:
            HessTc = _hess_lagrangian_ttc(Q0, Qi_list, dualquad, T, TTc, dim0)
            if existcons[0]:
                STcquad = (JPquad[:, None] * Qxq[TTc, :].T)
            else:
                STcquad = np.zeros((dim0, TTc.size))
            if existcons[1]:
                STcineq = (JPineq[:, None] * ineqA[:, TTc])
            else:
                STcineq = np.zeros((dim1, TTc.size))
            if existcons[2]:
                STceq = eqA[:, TTc]
            else:
                STceq = np.zeros((dim2, TTc.size))
            if existcons[3]:
                STcbd = np.zeros((s, TTc.size))
            else:
                STcbd = np.zeros((0, TTc.size))

            extra = _concat(
                HessTc @ x[TTc],
                STcquad @ x[TTc],
                STcineq @ x[TTc],
                STceq @ x[TTc],
                STcbd @ x[TTc],
            )
            STEq = -StationEq + extra

        if lenf_curr < 1000:
            try:
                d = np.linalg.solve(HessL, STEq)
            except np.linalg.LinAlgError:
                regularizer = HessL.T @ HessL + (0.01 / it) * np.eye(lenf_curr)
                rhs = HessL.T @ STEq
                d = np.linalg.solve(regularizer, rhs)
        else:
            d = _conjugate_gradient(HessL, STEq, 1e-16, 20, np.zeros(lenf_curr))

        if np.isnan(d).any():
            regularizer = HessL.T @ HessL + (0.01 / it) * np.eye(lenf_curr)
            rhs = HessL.T @ STEq
            d = np.linalg.solve(regularizer, rhs)

        dT = d[Indx]
        dqual = d[Indqual] if dim0 else np.zeros(0)
        dineq = d[Indineq] if dim1 else np.zeros(0)
        deq = d[Indeq] if dim2 else np.zeros(0)
        dbd = d[offset_bnd : offset_bnd + sbd] if sbd else np.zeros(0)

        mark = 0
        while True:
            xT1 = x[T] + dT
            dualquad1 = dualquad + dqual
            dualineq1 = dualineq + dineq
            dualeq1 = dualeq + deq

            if existcons[0]:
                Qxq = _func_qxq(xT1, Qi_list, qi_mat, dim0, T)
                Qqc = _func_qxqc(xT1, Qi_list, qi_mat, ci_vec, dim0, T)
                Ncpqual = _func_ncp_quad(Qqc, dualquad1)
            if existcons[1]:
                Axb = _func_axb(xT1, ineqA, ineqb, T)
                Ncpineq = _func_ncp_ineq(Axb, dualineq1)
            if existcons[2]:
                Lineq = _func_axb(xT1, eqA, eqb, T)
            if existcons[3]:
                dualbdT1 = dualbd[T] + dbd
                xPT = xT1 - _proj_box(xT1 + dualbdT1, lb_vec[T], ub_vec[T])
            else:
                dualbdT1 = np.zeros(0)
                xPT = np.zeros(0)

            gradL1 = _grad_lagrangian(
                xT1,
                Q0,
                q0,
                Qxq,
                ineqA,
                eqA,
                dualquad1,
                dualineq1,
                dualeq1,
                T,
                existcons,
            )
            if existcons[3]:
                gradL1[T] += dualbdT1

            F1 = _concat(gradL1[T], Ncpqual, Ncpineq, Lineq, xPT)

            if np.linalg.norm(F1) < 1e4 * Err or mark == 2:
                break
            elif mark == 0:
                regularizer = HessL.T @ HessL + (0.01 / it) * np.eye(lenf_curr)
                rhs = HessL.T @ STEq
                d = np.linalg.solve(regularizer, rhs)
            else:
                d = STEq.copy()
            dT = d[Indx]
            dqual = d[Indqual] if dim0 else np.zeros(0)
            dineq = d[Indineq] if dim1 else np.zeros(0)
            deq = d[Indeq] if dim2 else np.zeros(0)
            dbd = d[offset_bnd : offset_bnd + sbd] if sbd else np.zeros(0)
            mark += 1

        alpha = alpha0
        tmp = np.linalg.norm(StationEq) ** 2 + np.linalg.norm(x[Tc]) ** 2
        for _ in range(itlser):
            if np.linalg.norm(F1) < (1 - 2 * sigma * alpha) * tmp:
                break
            alpha *= gamma
            xT1 = x[T] + alpha * dT
            dualquad1 = dualquad + alpha * dqual
            dualineq1 = dualineq + alpha * dineq
            dualeq1 = dualeq + alpha * deq

            if existcons[0]:
                Qxq = _func_qxq(xT1, Qi_list, qi_mat, dim0, T)
                Qqc = _func_qxqc(xT1, Qi_list, qi_mat, ci_vec, dim0, T)
                Ncpqual = _func_ncp_quad(Qqc, dualquad1)
            if existcons[1]:
                Axb = _func_axb(xT1, ineqA, ineqb, T)
                Ncpineq = _func_ncp_ineq(Axb, dualineq1)
            if existcons[2]:
                Lineq = _func_axb(xT1, eqA, eqb, T)
            if existcons[3]:
                dualbdT1 = dualbd[T] + alpha * dbd
                proj = _proj_box(xT1 + dualbdT1, lb_vec[T], ub_vec[T])
                xPT = xT1 - proj
            else:
                dualbdT1 = np.zeros(0)
                xPT = np.zeros(0)

            gradL1 = _grad_lagrangian(
                xT1,
                Q0,
                q0,
                Qxq,
                ineqA,
                eqA,
                dualquad1,
                dualineq1,
                dualeq1,
                T,
                existcons,
            )
            if existcons[3]:
                gradL1[T] += dualbdT1
            F1 = _concat(gradL1[T], Ncpqual, Ncpineq, Lineq, xPT)

        obj = _func_obj(xT1, Q0, q0, T)
        x = z.copy()
        x[T] = xT1
        xT = xT1
        dualquad = dualquad1
        dualineq = dualineq1
        dualeq = dualeq1
        if existcons[3]:
            dualbd = z.copy()
            dualbd[T] = dualbdT1
            dualbdT = dualbdT1
        else:
            dualbdT = np.zeros(0)
        GradL = gradL1
        T0 = T

        if it % 200 == 0:
            tau = max(0.1 * tau0, tau / 1.5)

        nz_curr = int(np.count_nonzero(x))
        if nz_curr > 2 * max(nz_prev, 1) and Err < 1e-2:
            tau /= 1.1
        nz_prev = nz_curr

    elapsed = time.perf_counter() - start
    result = SNSQPResult()
    result['sol'] = x
    result['sparsity'] = int(np.count_nonzero(x))
    result['error'] = float(Err)
    result['time'] = float(elapsed)
    result['iter'] = it
    result['obj'] = float(obj)

    if show:
        print(' -------------------------------------------------')
        print(f" Iter:       {result['iter']:5d} ")
        print(f" Obj :     {result['obj']:7.4f}")
        print(f" Time:     {result['time']:7.4f} seconds")

    return result


def _ensure_matrix(mat: Optional[ArrayLike], n: int) -> ArrayLike:
    if mat is None or (isinstance(mat, (list, tuple)) and len(mat) == 0):
        return np.zeros((0, n), dtype=float)
    arr = np.asarray(mat, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _ensure_vector(vec: Optional[ArrayLike], length: int) -> ArrayLike:
    if length == 0:
        return np.zeros(0)
    if vec is None:
        return np.zeros(length)
    return np.asarray(vec, dtype=float).reshape(length)


def _ensure_bounds(val: Optional[ArrayLike], n: int, fill: float) -> ArrayLike:
    if val is None:
        return np.full(n, fill, dtype=float)
    arr = np.asarray(val, dtype=float)
    if arr.size == 1:
        return np.full(n, float(arr), dtype=float)
    return arr.reshape(n)


def _set_parameters(
    n: int,
    s: int,
    Qi: Sequence[ArrayLike],
    ineqA: ArrayLike,
    eqA: ArrayLike,
    lb: ArrayLike,
    ub: ArrayLike,
    pars: Dict[str, Any],
) -> Tuple[int, int, int, np.ndarray, bool, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int, float, float, float, ArrayLike, ArrayLike]:
    dim0 = len(Qi)
    dim1 = ineqA.shape[0]
    dim2 = eqA.shape[0]

    existcons = np.ones(4, dtype=bool)
    existcons[0] = dim0 > 0
    existcons[1] = dim1 > 0
    existcons[2] = dim2 > 0
    existcons[3] = not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))

    lenf = s + dim0 + dim1 + dim2 + (s if existcons[3] else 0)
    flagbd = bool(np.any((lb == 0) | (ub == 0)))

    show = int(pars.get('show', 1))
    itmax = int(pars.get('itmax', 10_000))
    x0 = np.asarray(pars.get('x0', np.zeros(n)), dtype=float).reshape(n)
    tau = float(pars.get('tau', 1.0))
    tol = float(pars.get('tol', 1e-6))
    itlser = int(pars.get('itlser', 5))
    gamma = float(pars.get('gamma', 0.5))
    sigma = float(pars.get('sigma', 1e-4))
    alpha0 = float(pars.get('alpha0', 1.0))

    dualquad = np.asarray(pars.get('dualquad', np.zeros(dim0)), dtype=float).reshape(dim0)
    if dualquad.size != dim0:
        dualquad = np.zeros(dim0)

    dualineq = np.asarray(pars.get('dualineq', np.zeros(dim1)), dtype=float).reshape(dim1)
    if dualineq.size != dim1:
        dualineq = np.zeros(dim1)

    dualeq = np.asarray(pars.get('dualeq', np.zeros(dim2)), dtype=float).reshape(dim2)
    if dualeq.size != dim2:
        dualeq = np.zeros(dim2)

    if existcons[3]:
        dualbd = np.asarray(pars.get('dualbd', np.zeros(n)), dtype=float).reshape(n)
        if dualbd.size != n:
            dualbd = np.zeros(n)
    else:
        dualbd = np.zeros(0)

    return (
        dim0,
        dim1,
        dim2,
        existcons,
        flagbd,
        lenf,
        show,
        x0,
        dualquad,
        dualineq,
        dualeq,
        dualbd,
        tau,
        tol,
        itmax,
        itlser,
        gamma,
        sigma,
        alpha0,
        lb,
        ub,
    )


def _topk_indices(values: ArrayLike, k: int) -> np.ndarray:
    k = min(k, values.size)
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-values, k - 1)[:k]
    return np.sort(idx)


def _func_obj(xT: ArrayLike, Q0: ArrayLike, q0: ArrayLike, T: np.ndarray) -> float:
    return float(0.5 * xT @ Q0[np.ix_(T, T)] @ xT + q0[T] @ xT)


def _grad_lagrangian(
    xT: ArrayLike,
    Q0: ArrayLike,
    q0: ArrayLike,
    Qxq: ArrayLike,
    ineqA: ArrayLike,
    eqA: ArrayLike,
    mu: ArrayLike,
    lamb1: ArrayLike,
    lamb2: ArrayLike,
    T: np.ndarray,
    existcons: np.ndarray,
) -> ArrayLike:
    g = Q0[:, T] @ xT + q0
    if existcons[2]:
        g += eqA.T @ lamb2
    if existcons[1]:
        g += ineqA.T @ lamb1
    if existcons[0]:
        g += Qxq @ mu
    return g


def _hess_lagrangian_tt(
    Q0: ArrayLike,
    Qi: Sequence[ArrayLike],
    mu: ArrayLike,
    T: np.ndarray,
    k: int,
) -> ArrayLike:
    h = Q0[np.ix_(T, T)].astype(float)
    for i in range(k):
        h += mu[i] * Qi[i][np.ix_(T, T)]
    return h


def _hess_lagrangian_ttc(
    Q0: ArrayLike,
    Qi: Sequence[ArrayLike],
    mu: ArrayLike,
    T: np.ndarray,
    TTc: np.ndarray,
    k: int,
) -> ArrayLike:
    hc = Q0[np.ix_(T, TTc)].astype(float)
    for i in range(k):
        hc += mu[i] * Qi[i][np.ix_(T, TTc)]
    return hc


def _func_axb(xT: ArrayLike, A: ArrayLike, b: ArrayLike, T: np.ndarray) -> ArrayLike:
    if A.size == 0:
        return np.zeros(0)
    return A[:, T] @ xT - b


def _func_qxq(
    xT: ArrayLike,
    Qi: Sequence[ArrayLike],
    qi: ArrayLike,
    k: int,
    T: np.ndarray,
) -> ArrayLike:
    if k == 0:
        return np.zeros((qi.shape[0], 0))
    Qxq = qi.copy()
    for i in range(k):
        Qxq[:, i] += Qi[i][:, T] @ xT
    return Qxq


def _func_qxqc(
    xT: ArrayLike,
    Qi: Sequence[ArrayLike],
    qi: ArrayLike,
    ci: ArrayLike,
    k: int,
    T: np.ndarray,
) -> ArrayLike:
    if k == 0:
        return np.zeros(0)
    tmp = xT @ qi[T, :]
    Qqc = ci.copy()
    for i in range(k):
        Qqc[i] = 0.5 * xT @ Qi[i][np.ix_(T, T)] @ xT + tmp[i] + Qqc[i]
    return Qqc


def _func_ncp_quad(Qqc: ArrayLike, dualquad: ArrayLike) -> ArrayLike:
    return np.sqrt(Qqc**2 + dualquad**2) + Qqc - dualquad


def _func_ncp_ineq(Axb: ArrayLike, dualineq: ArrayLike) -> ArrayLike:
    return np.sqrt(Axb**2 + dualineq**2) + Axb - dualineq


def _jac_ncp_quad(
    Qqc: ArrayLike,
    dualquad: ArrayLike,
    k: int,
) -> Tuple[ArrayLike, ArrayLike]:
    if k == 0:
        return np.zeros(0), np.zeros(0)
    z = np.sqrt(Qqc**2 + dualquad**2)
    jprim = np.divide(Qqc, z, out=np.zeros_like(Qqc), where=z != 0) + 1
    jdual = np.divide(dualquad, z, out=np.zeros_like(dualquad), where=z != 0) - 1
    zeros = z == 0
    if np.any(zeros):
        angles = 2 * np.pi * np.random.rand(np.count_nonzero(zeros))
        radii = np.sqrt(np.random.rand(np.count_nonzero(zeros)))
        jprim[zeros] = radii * np.cos(angles) + 1
        jdual[zeros] = radii * np.sin(angles) - 1
    return jprim, jdual


def _jac_ncp_ineq(
    Axb: ArrayLike,
    dualineq: ArrayLike,
    m: int,
) -> Tuple[ArrayLike, ArrayLike]:
    if m == 0:
        return np.zeros(0), np.zeros(0)
    z = np.sqrt(Axb**2 + dualineq**2)
    jprim = np.divide(Axb, z, out=np.zeros_like(Axb), where=z != 0) + 1
    jdual = np.divide(dualineq, z, out=np.zeros_like(dualineq), where=z != 0) - 1
    zeros = z == 0
    if np.any(zeros):
        angles = 2 * np.pi * np.random.rand(np.count_nonzero(zeros))
        radii = np.sqrt(np.random.rand(np.count_nonzero(zeros)))
        jprim[zeros] = radii * np.cos(angles) + 1
        jdual[zeros] = radii * np.sin(angles) - 1
    return jprim, jdual


def _proj_box(xT: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> ArrayLike:
    return np.minimum(np.maximum(xT, lb), ub)


def _jac_proj_box(xT: ArrayLike, lb: ArrayLike, ub: ArrayLike) -> ArrayLike:
    jpbd = np.zeros_like(xT)
    mid = (xT > lb) & (xT < ub)
    jpbd[mid] = 1.0
    edge = (xT == lb) | (xT == ub)
    jpbd[edge] = 0.5
    return jpbd


def _concat(*arrays: ArrayLike) -> ArrayLike:
    non_empty = [arr for arr in arrays if arr.size]
    if not non_empty:
        return np.zeros(0)
    return np.concatenate(non_empty)


def _conjugate_gradient(
    A: ArrayLike,
    b: ArrayLike,
    tol: float,
    maxiter: int,
    x0: ArrayLike,
) -> ArrayLike:
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    rs0 = rs_old
    for _ in range(maxiter):
        if rs_old < tol * rs0:
            break
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x
