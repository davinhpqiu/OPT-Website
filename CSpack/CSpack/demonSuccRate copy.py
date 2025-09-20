import io
import os
from contextlib import redirect_stdout
from functools import partial
from multiprocessing import Pool

import numpy as np

try:
    from .CSpack import CSpack as run_cspack
    from .CSpack.funcs.normalization import normalization
except Exception:  # pragma: no cover
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    from CSpack.CSpack import CSpack as run_cspack
    from CSpack.CSpack.funcs.normalization import normalization


_SUCCESS_THRESHOLD = 1e-2


def _run_single_trial(seed, n, m_cur, s_cur, pars):
    np.random.seed(seed)
    idx = np.random.permutation(n)[:s_cur]
    xopt = np.zeros(n)
    xopt[idx] = np.random.randn(s_cur)
    A_mat = normalization(np.random.randn(m_cur, n), 3)
    b = A_mat[:, idx] @ xopt[idx]

    norm_x = np.linalg.norm(xopt)
    success = [0, 0]
    if norm_x == 0:
        return tuple(success)

    with redirect_stdout(io.StringIO()):
        out = run_cspack(A_mat, None, b, n, s_cur, 'NHTP', pars.copy())
    if np.linalg.norm(out['sol'] - xopt) / norm_x < _SUCCESS_THRESHOLD:
        success[0] = 1

    with redirect_stdout(io.StringIO()):
        out = run_cspack(A_mat, None, b, n, s_cur, 'GPNP', pars.copy())
    if np.linalg.norm(out['sol'] - xopt) / norm_x < _SUCCESS_THRESHOLD:
        success[1] = 1

    return tuple(success)


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
    base_rng = np.random.default_rng(0)

    use_parallel = os.cpu_count() and os.cpu_count() > 1 and no_trials >= 4
    pool = Pool(processes=max(1, os.cpu_count() - 1)) if use_parallel else None

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

        seeds = base_rng.integers(0, 2**32, size=no_trials, dtype=np.uint32)
        rate = np.zeros(2, dtype=int)
        progress_step = max(1, no_trials // 10)
        processed = 0

        print(f"Running trials for value {val} ({j + 1}/{len(values)})", flush=True)

        if pool is not None:
            worker = partial(_run_single_trial, n=n, m_cur=m_cur, s_cur=s_cur, pars=pars)
            for succ_nhtp, succ_gpnp in pool.imap(worker, seeds, chunksize=4):
                rate[0] += succ_nhtp
                rate[1] += succ_gpnp
                processed += 1
                if processed % progress_step == 0 or processed == no_trials:
                    print(
                        f"  processed {processed}/{no_trials} trials",
                        flush=True,
                    )
        else:
            for seed in seeds:
                succ_nhtp, succ_gpnp = _run_single_trial(int(seed), n, m_cur, s_cur, pars)
                rate[0] += succ_nhtp
                rate[1] += succ_gpnp
                processed += 1
                if processed % progress_step == 0 or processed == no_trials:
                    print(
                        f"  processed {processed}/{no_trials} trials",
                        flush=True,
                    )

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

    if pool is not None:
        pool.close()
        pool.join()

    return suc_rate


if __name__ == '__main__':
    main()
