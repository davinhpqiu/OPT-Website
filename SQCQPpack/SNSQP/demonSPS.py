import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for candidate in (_PARENT, _HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from SNSQP.solver import snsqp


def main():
    rng = np.random.RandomState()

    n = 1000
    s = 10

    B = 0.01 * rng.random((int(np.ceil(n / 4)), n))
    D = np.diag(0.01 * rng.random(n))
    Q0 = 2.0 * (B.T @ B + D)
    q0 = np.zeros(n)
    Qi = [2.0 * D]
    qi = np.zeros(n)
    ci = np.array([-0.001])
    ineqA = -0.5 * rng.normal(size=(1, n))
    ineqb = np.array([-0.002])
    eqA = np.ones((1, n))
    eqb = np.array([1.0])
    lb = 0.0
    ub = 0.3

    pars = {
        'x0': np.full(n, (lb + ub) / 2.0),
        'tau': 1.0,
        'dualquad': np.zeros(ci.size),
        'dualineq': np.full(ineqb.size, 0.001),
        'dualeq': np.full(eqb.size, 0.001),
    }

    out = snsqp(
        n,
        s,
        Q0,
        q0,
        Qi,
        qi,
        ci,
        ineqA,
        ineqb,
        eqA,
        eqb,
        lb,
        ub,
        pars,
    )


if __name__ == '__main__':
    main()
