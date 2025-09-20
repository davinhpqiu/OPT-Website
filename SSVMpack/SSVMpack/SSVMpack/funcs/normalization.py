from __future__ import annotations

import time
from typing import Sequence

import numpy as np


def normalization(X: Sequence[Sequence[float]], normal_type: int):
    """Python port of the dataset normalisation helper."""

    start = time.perf_counter()
    X = np.asarray(X, dtype=float)
    if normal_type == 0:
        NX = X.copy()
    elif normal_type == 1:
        Xc = X - X.mean(axis=1, keepdims=True)
        row_std = np.std(X, axis=1, ddof=0, keepdims=True)
        row_std[row_std == 0] = 1.0
        Yrow = Xc / row_std
        Y = Yrow.T
        Yc = Y - Y.mean(axis=1, keepdims=True)
        col_std = np.std(Y, axis=1, ddof=0, keepdims=True)
        col_std[col_std == 0] = 1.0
        NX = (Yc / col_std).T
        if np.isnan(NX).any():
            scale = 1.0 / np.sqrt(np.sum(X * X, axis=0))
            scale[np.isinf(scale)] = 0.0
            NX = X * scale
    else:
        if normal_type == 2:
            scale = 1.0 / np.max(np.abs(X), axis=0)
        else:
            scale = 1.0 / np.sqrt(np.sum(X * X, axis=0))
        scale[np.isinf(scale)] = 0.0
        lX = scale.size
        NX = X * scale
        if lX > 10_000:
            k = 5_000
            if np.count_nonzero(X) / (lX * lX) < 1e-4:
                k = 100_000
            K = int(np.ceil(lX / k))
            NX = X.copy()
            for i in range(K - 1):
                T = slice(i * k, (i + 1) * k)
                NX[:, T] = X[:, T] * scale[T]
            T = slice((K - 1) * k, lX)
            NX[:, T] = X[:, T] * scale[T]
    NX = np.nan_to_num(NX, nan=0.0)
    print(f' Data nomorlization took {time.perf_counter() - start:2.4f} seconds')
    return NX
