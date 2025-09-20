import numpy as np

try:
    from .SSVMpack import SSVMpack as run_ssvmpack
    from .SSVMpack.funcs.randomData import randomData
    from .SSVMpack.funcs.accuracy import accuracy
    from .SSVMpack.funcs.plot2D import plot2D
except Exception:  # pragma: no cover
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    from SSVMpack.SSVMpack import SSVMpack as run_ssvmpack
    from SSVMpack.SSVMpack.funcs.randomData import randomData
    from SSVMpack.SSVMpack.funcs.accuracy import accuracy
    from SSVMpack.SSVMpack.funcs.plot2D import plot2D


def main():
    kind_idx = 0  # 0 -> '2D', 1 -> '3D', 2 -> 'nD'
    kinds = ['2D', '3D', 'nD']
    m0 = int(4e2)
    n0 = 100
    Atrain, ytrain, Atest, ytest = randomData(kinds[kind_idx], m0, n0, 0.0)
    m, n = Atrain.shape

    solver = ['NM01', 'NSSVM']
    pars = {'C': 0.25}
    out = run_ssvmpack(Atrain, ytrain, solver[0], pars)

    acc, _, _ = accuracy(Atrain, out['x'], ytrain)
    tacc, _, _ = accuracy(Atest, out['x'], ytest)

    print(f" Training  Time:             {out['time']:.3f}sec")
    print(f" Training  Size:             {m}x{n}")
    print(f" Training  Accuracy:         {acc * 100:5.2f}%")
    print(f" Testing   Size:             {Atest.shape[0]}x{n}")
    print(f" Testing   Accuracy:         {tacc * 100:5.2f}%")
    print(f" Number of Support Vectors:  {out['sv']}")

    if kinds[kind_idx] == '2D' and m < 400:
        plot2D(Atrain, ytrain, out['x'], solver[0], acc)
        plot2D(Atest, ytest, out['x'], solver[0], tacc)


if __name__ == '__main__':
    main()
