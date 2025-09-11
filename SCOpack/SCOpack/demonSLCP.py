import numpy as np
try:
    from .SCOpack.SCOpack import SCOpack as run_SCOpack
    from .SCOpack.examples.linear_complementarity_problem.funcLCP import funcLCP
    from .SCOpack.examples.linear_complementarity_problem.generationLCPdata import generationLCPdata
    from .SCOpack.examples.compressed_sensing.PlotRecovery import PlotRecovery
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from SCOpack.SCOpack.SCOpack import SCOpack as run_SCOpack
    from SCOpack.SCOpack.SCOpack.examples.linear_complementarity_problem.funcLCP import funcLCP
    from SCOpack.SCOpack.SCOpack.examples.linear_complementarity_problem.generationLCPdata import generationLCPdata
    from SCOpack.SCOpack.SCOpack.examples.compressed_sensing.PlotRecovery import PlotRecovery


def main():
    n = 10000
    s = int(np.ceil(0.01 * n))
    examp = 2  # 1,2,3
    mattype = ['z-mat', 'sdp', 'sdp-non']
    data = generationLCPdata(mattype[examp - 1], n, s)
    f = lambda x, key, T1, T2: funcLCP(x, key, T1, T2, data)
    pars = {'eta': 1 + 4 * (n <= 1000), 'neg': 1, 'tol': 1e-6}
    solver = ['NHTP', 'GPNP', 'IIHT']
    out = run_SCOpack(f, n, s, solver[1], pars)

    print(f" Objective:         {out['obj']:5.2e}")
    print(f" CPU time:          {out['time']:.3f}sec")
    print(f" Sample size:       {n}x{n}")
    if 'xopt' in data:
        PlotRecovery(data['xopt'].reshape(-1), out['sol'], [900, 500, 500, 250], 1)


if __name__ == '__main__':
    main()
