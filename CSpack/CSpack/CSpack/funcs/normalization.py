from __future__ import annotations

import numpy as np


def normalization(X, normal_type: int):
    """Normalise matrix as in MATLAB helper."""

    X = np.asarray(X, dtype=float)
    if normal_type == 0:
        return X.copy()

    if normal_type == 1:
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
        return np.nan_to_num(NX)

    if normal_type == 2:
        scale = 1.0 / np.max(np.abs(X), axis=0)
    else:
        scale = 1.0 / np.sqrt(np.sum(X * X, axis=0))
    scale[np.isinf(scale)] = 0.0

    NX = X * scale
    return np.nan_to_num(NX)
