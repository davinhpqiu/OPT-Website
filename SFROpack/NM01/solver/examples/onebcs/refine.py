import numpy as np

def refine(x, sp, A0=None, c=None):
    # x:  a vector in R^{n x 1}               (REQUIRED)
    # sp: a positive integer in [1,n] or [].  (REQUIRED)
    # AO: a matrix in R^{m x n}. If sp ~=[], then A0 is REQUIRED
    # c:  a vector in R^{m x 1}. If sp ~=[], then c  is REQUIRED

    x = x.flatten()  # guarantee x is a 1-D array

    # Determine whether to use sparsity level `sp`
    if isinstance(sp, (int, np.integer)):
        use_sp = sp > 0
    else:
        try:
            use_sp = sp is not None and len(sp) > 0
        except TypeError:
            use_sp = bool(sp)

    if use_sp:
        if A0 is None or c is None:
            raise ValueError("A0 and c must be provided when sp is specified")
        m, n = A0.shape
        K = 6
        kk = min(x.size, max(1, int(sp) + K - 1))
        Ts = np.argsort(np.abs(x))[::-1][:kk]
        sx = np.sort(np.abs(x))[::-1][:kk]
        HD = np.ones(K)
        X = np.zeros((A0.shape[1], K))
        if kk > int(sp) and (sx[int(sp) - 1] - sx[int(sp)]) <= 5e-2:
            tem = Ts[int(sp) - 1:]
            for i in range(min(K, tem.size)):
                X[:, i] = 0
                X[Ts[:int(sp) - 1], i] = x[Ts[:int(sp) - 1]]
                X[tem[i], i] = x[tem[i]]
                nz = np.linalg.norm(X[:, i])
                if nz > 0:
                    X[:, i] = X[:, i] / nz
                HD[i] = np.count_nonzero(np.sign(A0 @ X[:, i]) - c) / m
            i = np.argmin(HD)
            refx = X[:, i]
        else:
            refx = np.zeros(A0.shape[1])
            topk = Ts[:int(sp)]
            nz = np.linalg.norm(x[topk])
            if nz > 0:
                refx[topk] = x[topk] / nz
            else:
                refx[topk] = x[topk]
    else:
        refx = SparseApprox(x)
        nrm = np.linalg.norm(refx)
        if nrm > 0:
            refx = refx / nrm

    if np.isnan(refx).any():
        refx = SparseApprox(x)
        nrm = np.linalg.norm(refx)
        if nrm > 0:
            refx = refx / nrm

    return refx

# get the sparse approximation of x----------------------------------------
def SparseApprox(x0):
    n = len(x0)
    x = np.abs(x0[np.abs(x0) > 1e-2 / n])
    sx = np.sort(x[x != 0])
    if len(sx) < 2:
        return x0

    # Match MATLAB's normalize(...,'zscore') behavior: z = (ratio - mean)/std
    ratio = sx[1:] / sx[:-1]
    mu = np.mean(ratio)
    sigma = np.std(ratio)
    if sigma == 0:
        z = np.zeros_like(ratio)
    else:
        z = (ratio - mu) / sigma
    mx = np.max(z)
    it = int(np.argmax(z))

    th = 0
    if mx > 10 and it > 0:
        th = sx[it]
    x0[np.abs(x0) <= th] = 0
    return x0
