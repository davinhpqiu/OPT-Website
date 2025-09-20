import numpy as np

try:
    from .OneBCSpack import OneBCSpack as run_obcspack
    from .OneBCSpack.funcs import random1bcs
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from OBCSpack.OneBCSpack import OneBCSpack as run_obcspack
    from OBCSpack.OneBCSpack.OneBCSpack.funcs import random1bcs


def main():
    n = 500
    m = int(np.ceil(0.5 * n))
    s = int(np.ceil(0.01 * n))
    r = 0.05
    k = int(np.ceil(r * m))
    v = 0.5

    kind = 'Ind'
    test = 's'
    if test == 'm':
        test0 = np.linspace(0.2, 2, 10)
    elif test == 's':
        test0 = np.arange(2, 11)
    elif test == 'r':
        test0 = np.arange(0.01, 0.11, 0.01)
    elif test == 'v':
        test0 = np.arange(0.1, 1.0, 0.1)
        kind = 'Cor'
    elif test == 'n':
        test0 = np.arange(5, 21) * 1000
    else:
        raise ValueError('Unknown test type')

    varying_n = test == 'n'
    S = 10 if varying_n else 20
    recd = np.zeros((len(test0), 4))
    pars = {'disp': 0}

    for j, val in enumerate(test0):
        m_cur, n_cur, s_cur, r_cur, v_cur = m, n, s, r, v
        if test == 'm':
            m_cur = int(np.ceil(val * n_cur))
        elif test == 's':
            s_cur = int(val)
        elif test == 'r':
            r_cur = float(val)
        elif test == 'v':
            v_cur = float(val)
        elif test == 'n':
            n_cur = int(val)
            s_cur = int(np.ceil(0.01 * n_cur))
            m_cur = int(np.ceil(n_cur / 2))

        k_cur = int(np.ceil(r_cur * m_cur))

        for _ in range(S):
            A, b, bo, xo = random1bcs(kind, m_cur, n_cur, s_cur, r_cur, 0.01, v_cur)
            out = run_obcspack(A, b, s_cur, k_cur, 'GPSP', pars)
            err = np.linalg.norm(xo - out['sol'])
            recd[j, 0] += -20 * np.log10(err + 1e-12)
            recd[j, 1] += np.count_nonzero(np.sign(A @ out['sol']) - b) / m_cur
            recd[j, 2] += np.count_nonzero(np.sign(A @ out['sol']) - bo) / m_cur
            recd[j, 3] += out['time']

    recd /= S

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return recd

    fig, axs = plt.subplots(1, 4, figsize=(9, 2))
    ylabels = ['SNR', 'HD', 'HE', 'TIME']
    for j, ax in enumerate(axs):
        ax.plot(test0, recd[:, j], 'b.-', linewidth=0.75)
        if j == 0:
            ax.set_ylim(0, max(0.35, np.max(recd[:, j]) + 0.05))
        elif j == 3:
            ax.set_ylim(0, max(0.004, np.max(recd[:, j]) + 0.001))
        else:
            ax.set_ylim(0, max(0.35, np.max(recd[:, j]) + 0.05))
        ax.set_xlabel(test)
        ax.set_ylabel(ylabels[j])
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    return recd


if __name__ == '__main__':
    main()
