from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .solvers.NM01 import NM01
from .solvers.NSSVM import NSSVM


def SSVMpack(A, y, solver: str, pars: Optional[Dict[str, Any]] = None):
    """Python port of ``SSVMpack.m``."""

    if solver is None:
        raise ValueError(' Inputs are not enough !!! ')
    if pars is None:
        pars = {}

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m, n = A.shape

    solver_upper = solver.strip().upper()
    if m < 0.1 * n:
        print(' Suggest solver <NM01> !!! ')
        solver_upper = 'NM01'

    if solver_upper == 'NSSVM':
        return NSSVM(A, y, pars)
    if solver_upper == 'NM01':
        if 'C' in pars:
            pars = pars.copy()
            pars['lam'] = max(0.1, pars['C'])
        return NM01(A, y, pars)
    raise ValueError(f'Unknown solver {solver}')
