import sys
from math import ceil
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for candidate in (_PARENT, _HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from SNSQP.solver import snsqp
from SNSQP.solver.PlotSCCA import plot_sps
from SNSQP.solver.data.DataSCCA import generate_scca_data


def main():
    nx = 200
    ny = 300
    N = 50
    s = 10
    n = nx + ny

    data = generate_scca_data(nx, ny, N)

    pars = {
        'x0': data['x0'],
        'tau': 0.5,
        'dualquad': np.full(len(data['ci']), 0.01),
    }

    out = snsqp(
        n,
        s,
        data['Q0'],
        data['q0'],
        data['Qi'],
        data['qi'],
        data['ci'],
        None,
        None,
        None,
        None,
        None,
        None,
        pars,
    )

    print(f" Corr:      {-out['obj']:.4f} \n")

    try:
        plot_sps(out['sol'], ceil(nx / 200))
    except RuntimeError:
        pass


if __name__ == '__main__':
    main()
