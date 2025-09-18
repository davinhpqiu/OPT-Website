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
from SROpack.SROpack.examples.funcLinReg import funcLinReg


def _load_plot_recovery():
    module_path = Path(__file__).resolve().parent / 'SROpack' / 'useful funcs' / 'PlotRecovery.py'
    spec = importlib.util.spec_from_file_location('sropack_plot_recovery', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PlotRecovery


PlotRecovery = _load_plot_recovery()


def main():
    n = 1000
    m = int(np.ceil(0.25 * n))
    s = int(np.ceil(0.05 * n))

    Tx = np.random.permutation(n)[:s]
    xopt = np.zeros(n)
    xopt[Tx] = (0.25 + np.random.rand(s)) * np.sign(np.random.randn(s))
    A = np.random.randn(m, n) / np.sqrt(m)
    b = A @ xopt

    f = lambda x, key, T1, T2: funcLinReg(x, key, T1, T2, A, b)
    lam = 0.01
    pars = {'eta': 1.0}
    out = NL0R(f, n, lam, pars)
    PlotRecovery(xopt, out['sol'], (900, 500, 500, 250), True)


if __name__ == '__main__':
    main()
