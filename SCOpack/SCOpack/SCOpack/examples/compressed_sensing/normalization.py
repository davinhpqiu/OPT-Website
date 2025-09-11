import numpy as np
from scipy.sparse import diags


def normalization(X, normal_type):
    if normal_type == 0:
        NX = X

    elif normal_type == 1:
        C = X - np.mean(X, axis=1, keepdims=True)
        Yrow = C / np.std(X, axis=1, ddof=0, keepdims=True)
        Y = Yrow.T
        D = Y - np.mean(Y, axis=1, keepdims=True)
        Ycol = D / np.std(Y, axis=1, ddof=0, keepdims=True)
        NX = Ycol.T

    else:
        if normal_type == 2:
            nX = 1.0 / np.max(np.abs(X), axis=0)
        else:
            nX = 1.0 / np.sqrt(np.sum(X * X, axis=0))

        lX = len(nX)
        if lX <= 10000:
            NX = X @ diags(nX)
        else:
            k = int(5e3)
            if np.count_nonzero(X) / lX / lX < 1e-4:
                k = int(1e5)
            K = int(np.ceil(lX / k))
            for i in range(K - 1):
                T = slice(i * k, (i + 1) * k)
                X[:, T] = X[:, T] @ diags(nX[T])
            T = slice((K - 1) * k, lX)
            X[:, T] = X[:, T] @ diags(nX[T])
            NX = X

    return np.nan_to_num(NX)