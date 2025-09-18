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
from SROpack.SROpack.examples.compressed_sensing.funcCS import funcCS


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
    nf = 0.0

    Tx = np.random.permutation(n)[:s]
    xopt = np.zeros(n)
    xopt[Tx] = (0.25 + np.random.rand(s)) * np.sign(np.random.randn(s))
    A = np.random.randn(m, n)
    scale = np.sqrt(m)
    data = {
        'A': A / scale,
        'b': (A / scale) @ xopt + nf * np.random.randn(m),
    }

    f = lambda x, key, T1, T2: funcCS(x, key, T1, T2, data)
    pars = {'eta': 1.0}
    lam = 0.01
    out = NL0R(f, n, lam, pars)

    print(f" CPU time:          {out['time']:.3f}sec")
    print(f" Objective:         {out['obj']:5.2e}")
    true_obj = 0.5 * np.linalg.norm(data['A'] @ xopt - data['b']) ** 2
    print(f" True Objective:    {true_obj:5.2e}")
    print(f" Sample size:       {m}x{n}")
    PlotRecovery(xopt, out['sol'], (900, 500, 500, 250), True)


if __name__ == '__main__':
    main()
