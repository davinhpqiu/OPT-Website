import numpy as np
try:
    from .SCOpack.SCOpack import SCOpack as run_SCOpack
    from .SCOpack.examples.funcSimpleEx import funcSimpleEx
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from SCOpack.SCOpack.SCOpack import SCOpack as run_SCOpack
    from SCOpack.SCOpack.SCOpack.examples.funcSimpleEx import funcSimpleEx


def main():
    n = 2
    s = 1
    func = funcSimpleEx
    solver = ['NHTP', 'GPNP', 'IIHT']
    pars = {'eta': 0.1}
    out = run_SCOpack(func, n, s, solver[1], pars)
    print(f" Objective:      {out['obj']:.4f}")
    print(f" CPU time:      {out['time']:.3f}sec")
    print(f" Iterations:        {out['iter']:4d}")


if __name__ == '__main__':
    main()
