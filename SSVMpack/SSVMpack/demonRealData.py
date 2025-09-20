import numpy as np
from pathlib import Path

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('scipy is required to load .mat files for this demo') from exc

try:
    from .SSVMpack import SSVMpack as run_ssvmpack
    from .SSVMpack.funcs.normalization import normalization
    from .SSVMpack.funcs.accuracy import accuracy
except Exception:  # pragma: no cover
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    from SSVMpack.SSVMpack import SSVMpack as run_ssvmpack
    from SSVMpack.SSVMpack.funcs.normalization import normalization
    from SSVMpack.SSVMpack.funcs.accuracy import accuracy


def main():
    data_dir = Path(__file__).resolve().parent / 'SSVMpack' / 'data'
    data = loadmat(data_dir / 'dhrb.mat')
    labels = loadmat(data_dir / 'dhrbclass.mat')
    A = np.asarray(data['A'], dtype=float)
    y = np.asarray(labels['y']).reshape(-1).astype(float)

    m0, n = A.shape
    A = normalization(A, 2)
    m = int(np.ceil(0.9 * m0))
    perm = np.random.permutation(m0)
    train_idx = perm[:m]
    test_idx = perm[m:]

    Atrain = A[train_idx, :]
    Atest = A[test_idx, :]
    ytrain = y[train_idx]
    ytest = y[test_idx]

    solver = ['NM01', 'NSSVM']
    pars = {'C': 0.25}
    out = run_ssvmpack(Atrain, ytrain, solver[0], pars)
    acc, _, _ = accuracy(Atrain, out['x'], ytrain)
    tacc, _, _ = accuracy(Atest, out['x'], ytest)

    print(f" Training  Time:             {out['time']:.3f}sec")
    print(f" Training  Size:             {Atrain.shape[0]}x{Atrain.shape[1]}")
    print(f" Training  Accuracy:         {acc * 100:5.2f}%")
    print(f" Testing   Size:             {Atest.shape[0]}x{Atest.shape[1]}")
    print(f" Testing   Accuracy:         {tacc * 100:5.2f}%")
    print(f" Number of Support Vectors:  {out['sv']}")


if __name__ == '__main__':
    main()
