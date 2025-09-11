import numpy as np

def funcCS(x, key, T1, T2, data):
    """
    Least-squares objective: f(x) = 0.5 * ||A x - b||^2
    Returns (obj, grad) for key=='fg', and (H_T1T1, H_T1T2) for key=='h'.
    Supports matrix A or callable A with provided At.
    """
    A = data['A']
    b = data['b']

    if not callable(A):  # A is a matrix
        if key == 'fg':
            # Efficient Ax using sparsity of x
            if np.count_nonzero(x) >= 0.8 * len(x):
                Axb = A @ x - b
            else:
                Tx = np.nonzero(x)[0]
                Axb = A[:, Tx] @ x[Tx] - b
            obj = 0.5 * float(Axb.T @ Axb)
            grad = A.T @ Axb
            return obj, grad
        elif key == 'h':
            AT = A[:, T1]
            if len(T1) <= 1e3 and AT.shape[0] <= 5e3:
                H11 = AT.T @ AT  # subHessian
            else:
                H11 = lambda var: (AT.T @ (AT @ var))
            if T2 is None:
                return H11
            H12 = lambda var: (AT.T @ (A[:, T2] @ var))
            return H11, H12
        else:
            raise ValueError("key must be 'fg' or 'h'")

    # A is a function handle
    if 'At' not in data:
        raise ValueError('The transpose data["At"] is missing for function-handle A')
    if 'n' not in data:
        raise ValueError('The dimension data["n"] is missing for function-handle A')

    if key == 'fg':
        Axb = A(x) - b
        obj = 0.5 * float(Axb.T @ Axb)
        grad = data['At'](Axb)
        return obj, grad
    elif key == 'h':
        func = fgH(data)
        H11 = lambda var: func(var, T1, T1)
        if T2 is None:
            return H11
        H12 = lambda var: func(var, T1, T2)
        return H11, H12
    else:
        raise ValueError("key must be 'fg' or 'h'")

def fgH(data):
    def supp(n, x, T):
        z = np.zeros(n)
        z[T] = x
        return z
    
    def sub(z, t):
        return z[t]

    def Hess(z, t1, t2):
        return sub(data['At'](data['A'](supp(data['n'], z, t2))), t1)
    
    return Hess
