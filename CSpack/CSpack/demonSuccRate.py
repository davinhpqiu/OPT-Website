import io
from contextlib import redirect_stdout

import numpy as np

try:
    from .CSpack import CSpack as run_cspack
    from .CSpack.funcs.normalization import normalization
except Exception:  # pragma: no cover
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    from CSpack.CSpack import CSpack as run_cspack
    from CSpack.CSpack.funcs.normalization import normalization


def main():
    test = 1  # 1 -> success rate vs sparsity; 2 -> vs measurement ratio
    n = 256
    m = int(np.ceil(0.25 * n))
    s = int(np.ceil(0.05 * n))
    no_trials = 100

    if test == 1:
        values = np.arange(10, 40, 2)
    elif test == 2:
        values = np.linspace(0.06, 0.28, 12)
    else:
        raise ValueError('Unknown test type')

    suc_rate = np.zeros((2, len(values)), dtype=int)
    pars = {'disp': 0, 'report_opt': False}
    np.random.seed(0)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    for j, val in enumerate(values):
        m_cur = m
        s_cur = s
        if test == 1:
            s_cur = int(val)
        else:
            m_cur = int(np.ceil(float(val) * n))

        rate = np.zeros(2, dtype=int)
        for _ in range(no_trials):
            idx = np.random.permutation(n)[:s_cur]
            xopt = np.zeros(n)
            xopt[idx] = np.random.randn(s_cur)
            A_mat = normalization(np.random.randn(m_cur, n), 3)
            b = A_mat[:, idx] @ xopt[idx]

            with redirect_stdout(io.StringIO()):
                out = run_cspack(A_mat, None, b, n, s_cur, 'NHTP', pars)
            if np.linalg.norm(xopt) > 0 and np.linalg.norm(out['sol'] - xopt) / np.linalg.norm(xopt) < 1e-2:
                rate[0] += 1

            with redirect_stdout(io.StringIO()):
                out = run_cspack(A_mat, None, b, n, s_cur, 'GPNP', pars)
            if np.linalg.norm(xopt) > 0 and np.linalg.norm(out['sol'] - xopt) / np.linalg.norm(xopt) < 1e-2:
                rate[1] += 1

        suc_rate[:, j] = rate

        print('SucRate =')
        for row in suc_rate[:, : j + 1]:
            print('  ' + '   '.join(f'{val:3d}' for val in row))
        print(flush=True)

    print('Final SucRate =')
    for row in suc_rate:
        print('  ' + '   '.join(f'{val:3d}' for val in row))

    if plt is not None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        x_label = ['s', 'm/n'][test - 1]
        ax.plot(values, suc_rate[0] / no_trials, 'r*-', label='NHTP')
        ax.plot(values, suc_rate[1] / no_trials, 'bo-', label='GPSP')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Success Rate')
        ax.set_xlim(values[0], values[-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return suc_rate


if __name__ == '__main__':
    main()
