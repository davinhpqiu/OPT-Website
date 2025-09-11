import numpy as np
try:
    from .SCOpack.SCOpack import SCOpack as run_SCOpack
    from .SCOpack.examples.funcLinReg import funcLinReg
    from .SCOpack.examples.compressed_sensing.PlotRecovery import PlotRecovery
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from SCOpack.SCOpack.SCOpack import SCOpack as run_SCOpack
    from SCOpack.SCOpack.SCOpack.examples.funcLinReg import funcLinReg
    from SCOpack.SCOpack.SCOpack.examples.compressed_sensing.PlotRecovery import PlotRecovery


def main():
    n = 20000
    m = int(np.ceil(0.25 * n))
    s = int(np.ceil(0.025 * n))

    Tx = np.random.permutation(n)[:s]
    xopt = np.zeros((n,))
    xopt[Tx] = np.random.randn(s)
    A = np.random.randn(m, n) / np.sqrt(m)
    b = A @ xopt

    f = lambda x, key, T1, T2: funcLinReg(x, key, T1, T2, A, b)
    pars = {'tol': 1e-6}
    solver = ['NHTP', 'GPNP', 'IIHT']
    out = run_SCOpack(f, n, s, solver[1], pars)
    PlotRecovery(xopt, out['sol'], [900, 500, 500, 250], 1)


if __name__ == '__main__':
    main()
