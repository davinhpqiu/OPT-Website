import importlib
from typing import Callable, Dict, Any


def SCOpack(func: Callable, n: int, s: int, solvername: str, pars: Dict[str, Any] = None):
    """
    Python port of SCOpack.m

    Solves sparsity constrained optimization (SCO) or nonnegative SCO (NSCO):
        min f(x) s.t. ||x||_0 <= s  [and optionally x >= 0]

    Inputs:
      - func: callable implementing (objective, gradient, sub-Hessian) interface via
               func(x, key, T1, T2, ...), where key in {'fg','h'}
      - n: dimension of x
      - s: sparsity level (1..n-1)
      - solvername: one of {'NHTP','GPNP','IIHT'}
      - pars: optional parameters dict

    Returns a dict with keys: sol, obj, iter, time (solver dependent)
    """
    if func is None or n is None or s is None or solvername is None:
        raise ValueError('Inputs are not enough !!!')
    if pars is None:
        pars = {}

    name = solvername.strip()
    if name not in {"NHTP", "GPNP", "IIHT"}:
        raise ValueError(f"Unknown solver '{solvername}'. Use one of 'NHTP','GPNP','IIHT'.")

    # Dynamic import from local solvers package
    module = importlib.import_module(f".solvers.{name}", package=__package__)
    solver = getattr(module, name)
    out = solver(func, n, s, pars)
    return out

