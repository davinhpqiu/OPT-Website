from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .solvers.GPSP import GPSP
from .solvers.NM01 import NM01


def OneBCSpack(A, b, s, k, solver: str, pars: Dict[str, Any] | None = None):
    """Python port of ``OBCSpack.m``."""

    if solver is None:
        raise ValueError(' Inputs are not enough !!! ')
    if pars is None:
        pars = {}

    solver = solver.strip().upper()
    if s is None or s == []:
        solver = 'NM01'

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if solver == 'GPSP':
        if k is None:
            raise ValueError('k is required for GPSP solver')
        return GPSP(A, b, int(s), int(k), pars)
    if solver == 'NM01':
        return NM01(A, b, s, pars)
    raise ValueError(f'Unknown solver {solver}')
