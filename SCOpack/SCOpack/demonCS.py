import numpy as np
import time

# Prefer package-relative imports; fall back to absolute when run by path
try:
    from .SCOpack.SCOpack import SCOpack as run_SCOpack
    from .SCOpack.examples.compressed_sensing.funcCS import funcCS
    from .SCOpack.examples.compressed_sensing.PlotRecovery import PlotRecovery
except Exception:  # running as a script without -m
    import os, sys
    # Ensure the repo root (containing top-level 'SCOpack') is first on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from SCOpack.SCOpack.SCOpack import SCOpack as run_SCOpack
    from SCOpack.SCOpack.SCOpack.examples.compressed_sensing.funcCS import funcCS
    from SCOpack.SCOpack.SCOpack.examples.compressed_sensing.PlotRecovery import PlotRecovery


def main():
    n = 10000
    m = int(np.ceil(0.25 * n))
    s = int(np.ceil(0.025 * n))
    nf = 0.00

    Tx = np.random.permutation(n)[:s]
    xopt = np.zeros((n,))
    xopt[Tx] = np.random.randn(s)
    A = np.random.randn(m, n)
    scale = (np.log(m) if False else np.sqrt(m))  # mimic MATLAB issparse(A)
    data = {}
    data['A'] = A / scale
    data['b'] = data['A'] @ xopt + nf * np.random.randn(m)

    f = lambda x, key, T1, T2: funcCS(x, key, T1, T2, data)
    pars = {'tol': 1e-6}
    if nf > 0:
        pars['eta'] = 0.5
    solver = ['NHTP', 'GPNP', 'IIHT']
    t0 = time.time()
    out = run_SCOpack(f, n, s, solver[1], pars)

    print(f" CPU time:          {out['time']:.3f}sec")
    print(f" Objective:         {out['obj']:5.2e}")
    print(f" True Objective:    {0.5*np.linalg.norm(data['A']@xopt - data['b'])**2:5.2e}")
    print(f" Sample size:       {m}x{n}")
    PlotRecovery(xopt, out['sol'], [900, 500, 500, 250], 1)


if __name__ == '__main__':
    main()
