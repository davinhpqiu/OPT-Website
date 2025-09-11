import numpy as np
from scipy import sparse as sp


def normalization(X: np.ndarray, normal_type: int):
    """
    Normalize input matrix X.
    normal_type:
      0: no normalization
      1: sample(row)-wise then feature(column)-wise standardization
      2: feature-wise scale columns to [-1, 1] via 1/max(|X|)
      3: feature-wise scale to unit-norm columns
    """
    if normal_type == 0:
        return X

    if normal_type == 1:
        # Sample-wise normalization
        row_mean = X.mean(axis=1, keepdims=True)
        C = X - row_mean
        row_std = X.std(axis=1, ddof=0, keepdims=True)
        # Avoid divide-by-zero
        row_std = np.where(row_std == 0, 1.0, row_std)
        Yrow = C / row_std

        # Feature-wise normalization
        Y = Yrow.T
        col_mean = Y.mean(axis=1, keepdims=True)
        D = Y - col_mean
        col_std = Y.std(axis=1, ddof=0, keepdims=True)
        col_std = np.where(col_std == 0, 1.0, col_std)
        Ycol = D / col_std
        return Ycol.T

    # normal_type 2 or 3
    if normal_type == 2:
        nX = 1.0 / np.maximum(np.abs(X), axis=0)
    else:
        nX = 1.0 / np.sqrt((X * X).sum(axis=0))

    # Replace inf with 0 for constant columns
    nX = np.where(np.isfinite(nX), nX, 0.0)
    lX = nX.shape[0]
    if lX <= 10000:
        D = sp.diags(nX, offsets=0, shape=(lX, lX), format='csr')
        return X @ D
    else:
        # Chunked multiplication for very wide matrices
        k = int(5e3)
        if (np.count_nonzero(X) / (lX * lX)) < 1e-4:
            k = int(1e5)
        K = int(np.ceil(lX / k))
        Xc = X.copy()
        for i in range(1, K):
            T = slice((i - 1) * k, i * k)
            D = sp.diags(nX[T], 0, shape=(k, k), format='csr')
            Xc[:, T] = Xc[:, T] @ D
        T = slice((K - 1) * k, lX)
        k0 = Xc[:, T].shape[1]
        D = sp.diags(nX[T], 0, shape=(k0, k0), format='csr')
        Xc[:, T] = sp.csr_matrix(Xc[:, T]) @ D
        return Xc

