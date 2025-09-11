import numpy as np
try:
    from .SCOpack.SCOpack import SCOpack as run_SCOpack
    from .SCOpack.examples.logistic_regression.funcLogReg import funcLogReg
    from .SCOpack.examples.logistic_regression.generationLRdata import LogitRegdata
    from .SCOpack.examples.compressed_sensing.normalization import normalization
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from SCOpack.SCOpack.SCOpack import SCOpack as run_SCOpack
    from SCOpack.SCOpack.SCOpack.examples.logistic_regression.funcLogReg import funcLogReg
    from SCOpack.SCOpack.SCOpack.examples.logistic_regression.generationLRdata import LogitRegdata
    from SCOpack.SCOpack.SCOpack.examples.compressed_sensing.normalization import normalization


def main():
    test = 1
    if test == 1:
        n = 10000
        m = int(np.ceil(n / 5))
        s = int(np.ceil(0.05 * n))
        rho = 0.5
        I = np.random.permutation(n)
        T = I[:s]
        A = np.random.randn(m, n)
        q = 1.0 / (1.0 + np.exp(-A[:, T] @ np.random.randn(s)))
        b = np.zeros((m,))
        for i in range(m):
            b[i] = np.random.choice([0, 1], p=[1 - q[i], q[i]])
        data = type('obj', (), {'A': A, 'b': b})
    else:
        prob = 'colon-cancer'
        # For real data loading, translate MATLAB .mat loading if present in Python env.
        raise NotImplementedError('Real data path not wired in this demo')

    f = lambda x, key, T1, T2: funcLogReg(x, key, T1, T2, data)
    pars = {'tol': 1e-6}
    solver = ['NHTP', 'GPNP', 'IIHT']
    out = run_SCOpack(f, n, s, solver[1], pars)
    print(f" Logistic Loss:  {out['obj']:5.2e}")
    print(f" CPU time:       {out['time']:.3f}sec")
    print(f" Sample size:    {m}x{n}")


if __name__ == '__main__':
    main()
