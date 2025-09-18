import numpy as np
from time import time
from pathlib import Path
import importlib.util


def _load_normalization():
    module_name = 'sropack_useful_normalization'
    module_path = Path(__file__).resolve().parents[2] / 'useful funcs' / 'normalization.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.normalization


try:
    from ..useful_funcs.normalization import normalization  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback when package name is not a valid identifier
    normalization = _load_normalization()


def CSdata(problemname: str, m: int, n: int, s: int, nf: float):
    """Python port of generationCSdata.m."""
    start = time()
    print(' Please wait for CS data generation ...')

    if problemname == 'GaussianMat':
        A = np.random.randn(m, n)
    elif problemname == 'PartialDCTMat':
        r = np.random.rand(m)
        column = np.arange(n)
        A = np.cos(2 * np.pi * r[:, None] * column)
    elif problemname == 'ToeplitzCorMat':
        t = np.arange(n)
        Sig = np.array([np.power(0.5, np.abs(i - t)) for i in range(n)])
        Sig = np.real(np.linalg.cholesky(Sig))
        A = np.random.randn(m, n) @ Sig
    else:
        raise ValueError('problemname must be GaussianMat, PartialDCTMat, or ToeplitzCorMat')

    I = np.random.permutation(n)[:s]
    xopt = np.zeros(n)
    while np.count_nonzero(xopt) != s:
        xopt[I] = np.random.randn(s)
    xopt[I] = xopt[I] + 2 * nf * np.sign(xopt[I])

    data = {}
    data['A'] = normalization(A, 3)
    data['b'] = data['A'][:, I] @ xopt[I] + nf * np.random.randn(m)
    data['xopt'] = xopt

    print(f' Data generation used {time() - start:.4f} seconds.\n')
    return data
