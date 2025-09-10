import numpy as np
from scipy.stats import multivariate_normal

def datageneration1BCS(type_, m, n, s, nf, r, v):
    
# This file aims at generating data of 2 types of Guassian samples 
# for 1-bit compressed sensing
# Inputs:
#       type    -- can be 'Ind','Cor' 
#       m       -- number of samples
#       n       -- number of features
#       s       -- sparsity of the true singnal, e.g., s=0.01n
#       nf      -- nosie factor 
#       r       -- flipping ratio, 0~1, e.g., r=0.05
#       v       -- corrolation factor, 0~1, e.g., v=0.5
# Outputs:
#       X       --  samples data, m-by-n matrix
#       xopt    --  n-by-1 vector, i.e., the ture singnal
#       y       --  m-by-1 vector, i.e., sign(X*xopt+noise)
#       yf      --  m-by-1 vector, y after flapping some signs

    if type_ == 'Ind':
        X = np.random.randn(m, n)
    elif type_ == 'Cor':
        indices = np.arange(n)
        S = v ** np.abs(np.subtract.outer(indices, indices))
        X = multivariate_normal.rvs(mean=np.zeros(n), cov=S, size=m)
        if m == 1:
            X = X.reshape(1, -1)
    else:
        raise ValueError("Invalid type. Must be 'Ind' or 'Cor'.")

    xopt, T = generate_sparse_vector(n, s)
    y = np.sign(X[:, T] @ xopt[T] + nf * np.random.randn(m))
    yf = flip_signs(y, r)

    return X, yf, y, xopt


def generate_sparse_vector(n, s):
    # generate a sparse vector
    T = np.random.permutation(n)[:s]
    x = np.zeros(n)
    x[T] = np.random.randn(s)
    x[T] += np.sign(x[T])        
    x[T] /= np.linalg.norm(x[T])  
    return x, T


def flip_signs(y, r):
    # flip the signs of a vector
    yf = y.copy()
    m = len(y)
    T = np.random.permutation(m)[:int(np.ceil(r * m))]
    yf[T] = -yf[T]
    return yf