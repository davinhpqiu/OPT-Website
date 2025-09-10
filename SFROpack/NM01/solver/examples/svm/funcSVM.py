import numpy as np
from scipy.sparse import eye as speye

def funcSVM(x, key, w, A=None, c=None):
    n = len(x)
    if key == 'f':
        # sum(x^2) - (1-w)*x_n^2 == sum_{i<n} x_i^2 + w*x_n^2
        out = np.dot(x, x) - (1 - w) * x[n - 1] ** 2
    elif key == 'g':
        out = x.copy()
        out[n - 1] = w * out[n - 1]
    elif key == 'h':
        # Return a Hessian-vector product to avoid allocating an n x n matrix.
        # H is diagonal with ones, except H[-1,-1] = w.
        def Hv(v):
            v = np.asarray(v).copy()
            v[n - 1] = w * v[n - 1]
            return v
        out = Hv
    elif key == 'a':
        acc = lambda var: np.count_nonzero(np.sign(A @ var[:n - 1] + var[n - 1]) - c)
        out = 1 - acc(x) / len(c)
    else:
        out = None  # 'Otherwise' is REQIURED
    return out
