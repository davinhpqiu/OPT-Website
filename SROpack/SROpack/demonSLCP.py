import importlib.util
import sys
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for path in (str(_PARENT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from SROpack.SROpack.solver.NL0R import NL0R
from SROpack.SROpack.examples.linear_complementarity_problem.funcLCP import funcLCP
from SROpack.SROpack.examples.linear_complementarity_problem.generationLCPdata import LCPdata


def _load_plot_recovery():
    module_path = Path(__file__).resolve().parent / 'SROpack' / 'useful funcs' / 'PlotRecovery.py'
    spec = importlib.util.spec_from_file_location('sropack_plot_recovery', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PlotRecovery


PlotRecovery = _load_plot_recovery()


def main():
    n = 1000
    s = int(np.ceil(0.05 * n))
    examp = 2
    mattype = ['z-mat', 'sdp', 'sdp-non']
    data = LCPdata(mattype[examp - 1], n, s)
    f = lambda x, key, T1, T2: funcLCP(x, key, T1, T2, data)

    lam = 0.01
    out = NL0R(f, n, lam)

    print(f" Objective:         {out['obj']:5.2e}")
    print(f" CPU time:          {out['time']:.3f}sec")
    print(f" Sample size:       {n}x{n}")

    xopt = data.get('xopt') if isinstance(data, dict) else getattr(data, 'xopt', None)
    if xopt is not None:
        PlotRecovery(xopt, out['sol'], (900, 500, 500, 250), True)


if __name__ == '__main__':
    main()
