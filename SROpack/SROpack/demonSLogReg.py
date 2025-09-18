import sys
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for path in (str(_PARENT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from SROpack.SROpack.solver.NL0R import NL0R
from SROpack.SROpack.examples.logistic_regression.funcLogReg import funcLogReg
from SROpack.SROpack.examples.logistic_regression.generationLRdata import LogitRegdata

def main():
    test = 2
    if test == 1:
        n = 10000
        m = int(np.ceil(n / 5))
        s = int(np.ceil(0.05 * n))
        rho = 0.5
        data, _ = LogitRegdata('Correlated', m, n, s, rho)
    else:
        prob = 'colon-cancer'
        raise NotImplementedError(f'Real dataset {prob} is not bundled with this demo')

    f = lambda x, key, T1, T2: funcLogReg(x, key, T1, T2, data)
    lam = 0.01
    pars = {'eta': 1.0}
    out = NL0R(f, n, lam, pars)

    print(f" Logistic Loss:  {out['obj']:5.2e}")
    print(f" CPU time:       {out['time']:.3f}sec")
    print(f" Sample size:    {m}x{n}")


if __name__ == '__main__':
    main()
