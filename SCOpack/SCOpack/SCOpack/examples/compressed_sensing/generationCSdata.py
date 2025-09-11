import numpy as np
from time import time
from .normalization import normalization


def CSdata(problemname, m, n, s, nf):
    """
    Generate data for compressed sensing problems using different types of measurement matrices.

    Inputs:
        problemname -- 'GaussianMat', 'PartialDCTMat', or 'ToeplitzCorMat'
        m, n        -- dimensions of A
        s           -- sparsity level of xopt, integer between 1 and n-1
        nf          -- noise ratio

    Outputs:
        data: dict with keys
            'A'      -- m x n measurement matrix (required)
            'b'      -- m x 1 observation vector (required)
            'xopt'   -- n x 1 true sparse solution (optional)
    """
    start = time()
    print(" Please wait for CS data generation ...")

    A = np.zeros((m, n))

    if problemname == 'GaussianMat':
        A = np.random.randn(m, n)
        I0 = np.random.permutation(n)
        I = I0[:s]

    elif problemname == 'PartialDCTMat':
        r = np.random.rand(m)
        column = np.arange(n)
        for i in range(m):
            A[i, :] = np.cos(2 * np.pi * r[i] * (column))
        I0 = np.random.permutation(n)
        I = I0[:s]

    elif problemname == 'ToeplitzCorMat':
        Sig = np.zeros((n, n))
        t = np.arange(n)
        for i in range(n):
            Sig[i, :] = np.power(0.5, np.abs(i - t))
        Sig = np.real(np.linalg.cholesky(Sig))  # MATLAB's Sig^(1/2)
        A = np.random.randn(m, n) @ Sig
        I0 = np.random.permutation(n)
        I = I0[:s]

    else:
        print('input a problem name')
        return None

    xopt = np.zeros(n)
    while np.count_nonzero(xopt) != s:
        xopt[I] = np.random.randn(s)
    
    xopt[I] = xopt[I] + 2 * nf * np.sign(xopt[I])

    data = {}
    data['A'] = normalization(A, mode=3)  # You must define `normalization`
    data['b'] = data['A'][:, I] @ xopt[I] + nf * np.random.randn(m)
    data['xopt'] = xopt

    print(f' Data generation used {time() - start:.4f} seconds.\n')
    return data
