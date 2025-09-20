import numpy as np

try:
    from .CSpack import CSpack as run_cspack
    from .CSpack.funcs.PlotRecovery import plot_recovery
except Exception:  # pragma: no cover
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    from CSpack.CSpack import CSpack as run_cspack
    from CSpack.CSpack.funcs.PlotRecovery import plot_recovery


def main():
    n = 1000
    m = int(np.ceil(0.25 * n))
    s = int(np.ceil(0.05 * n))

    idx = np.random.permutation(n)[:s]
    xopt = np.zeros(n)
    xopt[idx] = (0.1 + np.random.rand(s)) * np.sign(np.random.randn(s))
    A_mat = np.random.randn(m, n)
    scale = np.sqrt(m)
    A_mat = A_mat / scale
    b = A_mat[:, idx] @ xopt[idx] + 0.0 * np.random.randn(m)

    A = lambda v: A_mat @ v
    At = lambda v: A_mat.T @ v

    solver = ['GPNP', 'NHTP', 'IIHT', 'NL0R', 'PSNP', 'MIRL1']
    out = run_cspack(A, At, b, n, s, solver[0])

    print(f" Objective at xopt:       {0.5 * np.linalg.norm(A(xopt) - b) ** 2:.2e}")
    print(f" Objective at out.sol:    {out['obj']:.2e}")
    print(f" Sparsity of out.sol:     {np.count_nonzero(out['sol']):2d}")
    print(f" Computational time:      {out['time']:.3f}sec")

    plot_recovery(xopt, out['sol'], [1000, 500, 500, 250], True)


if __name__ == '__main__':
    main()
