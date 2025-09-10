import numpy as np
from scipy import sparse as sp

def normalization(X, normal_type):
    """
    Normalize input matrix X.

    normal_type:
      0: no normalization
      1: row-wise standardize, then column-wise standardize
      2: feature-wise scale to [-1, 1] (by dividing each column by its max abs)
      3: feature-wise scale to unit-norm columns
    """
    # Ensure dense float array (MATLAB .mat may store sparse)
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    if normal_type == 0:
        NX = X.copy()
    elif normal_type == 1:
        # Row-wise
        row_mean = X.mean(axis=1, keepdims=True)
        row_std = X.std(axis=1, ddof=0, keepdims=True)
        row_std[row_std == 0] = 1.0
        Yrow = (X - row_mean) / row_std
        # Column-wise on transposed then transpose back
        Y = Yrow.T
        col_mean = Y.mean(axis=1, keepdims=True)
        col_std = Y.std(axis=1, ddof=0, keepdims=True)
        col_std[col_std == 0] = 1.0
        Ycol = (Y - col_mean) / col_std
        NX = Ycol.T
        # Fallback if NaNs appear
        if np.isnan(NX).any():
            nX = 1.0 / np.sqrt((X * X).sum(axis=0))
            nX[~np.isfinite(nX)] = 0.0
            NX = X * nX
    else:
        if normal_type == 2:
            # Scale each feature by max abs to fit into [-1, 1]
            nX = 1.0 / np.maximum(np.abs(X), 1e-12).max(axis=0)
        else:
            # Unit norm columns
            nX = 1.0 / np.sqrt((X * X).sum(axis=0))
        nX[~np.isfinite(nX)] = 0.0
        NX = X * nX

    NX = np.nan_to_num(NX, nan=0.0, posinf=0.0, neginf=0.0)
    return NX
