from __future__ import annotations

import numpy as np


def accuracy(X, x, y):
    """Python port of ``accuracy.m``."""

    if X is None or len(X) == 0:
        return float('nan'), float('nan'), float('nan')

    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    z = X @ x[:-1] + x[-1]
    sz = np.sign(z)
    sz[sz == 0] = 1
    mis = np.count_nonzero(sz - y)
    acc = 1 - mis / y.size
    return acc, mis, sz
