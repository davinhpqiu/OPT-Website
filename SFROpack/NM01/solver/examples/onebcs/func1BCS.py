import numpy as np

def func1BCS(x, key, eps, q, A=None, c=None):
    if key == 'f':
        return np.sum((x**2 + eps)**(q / 2))
    elif key == 'g':
        return q * x * (x**2 + eps)**(q / 2 - 1)
    elif key == 'h':
        x2 = x**2
        diag_vals = (x2 + eps)**(q / 2 - 2) * ((q - 1) * x2 + eps)
        return np.diag(diag_vals)
    elif key == 'a':
        if A is None or c is None:
            raise ValueError("A and c must be provided when key='a'")
        sign_mismatches = np.count_nonzero(np.sign(A @ x) - c)
        return 1 - sign_mismatches / len(c)
    else:
        return None