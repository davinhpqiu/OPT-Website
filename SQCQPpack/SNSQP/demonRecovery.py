import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for candidate in (_PARENT, _HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from SNSQP.solver import snsqp
from SNSQP.solver.PlotRecovery import plot_recovery
from SNSQP.solver.data.DataRecovery import generate_recovery_data


def main():
    rng = np.random.RandomState()

    n = 1000
    k = int(np.ceil(0.01 * n))
    m = int(np.ceil(0.01 * n))
    s = int(np.ceil(0.05 * n))

    test = 1
    if test == 1:
        lb = -np.inf
        ub = np.inf
        xT = rng.randn(s)
    elif test == 2:
        lb = -2.0
        ub = 2.0
        xT = rng.uniform(lb, ub, size=s)
    else:
        lb = 0.0
        ub = np.inf
        xT = rng.rand(s)

    support = rng.permutation(n)[:s]
    xopt = np.zeros(n)
    xopt[support] = xT

    data = generate_recovery_data(n, k, m, xopt, support, rng)

    pars = {
        'x0': np.zeros(n),
        'tau': 3.0,
        'dualquad': np.full(k, 0.001),
        'dualineq': np.full(m, 0.001),
        'itlser': 1,
    }

    out = snsqp(
        n,
        s,
        data['Q0'],
        data['q0'],
        data['Qi'],
        data['qi'],
        data['ci'],
        data['A'],
        data['b'],
        None,
        None,
        lb,
        ub,
        pars,
    )

    relerr = np.linalg.norm(out['sol'] - xopt) / (np.linalg.norm(xopt) + 1e-12)
    print(f" Relerr:    {relerr:7.3e} ")

    try:
        plot_recovery(xopt, out['sol'])
    except RuntimeError:
        pass


if __name__ == '__main__':
    main()
