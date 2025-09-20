import numpy as np

try:
    from .SSVMpack import SSVMpack as run_ssvmpack
    from .SSVMpack.funcs.accuracy import accuracy
except Exception:  # pragma: no cover
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    from SSVMpack.SSVMpack import SSVMpack as run_ssvmpack
    from SSVMpack.SSVMpack.funcs.accuracy import accuracy


def main():
    a = 10
    A = np.array([[0, 0], [0, 1], [1, 0], [1, a]], dtype=float)
    y = np.array([-1, -1, 1, 1], dtype=float)
    pars = {'C': 1.0}
    out = run_ssvmpack(A, y, 'NM01', pars)

    err, _, _ = accuracy(A, out['x'], y)
    print(f" Training accuracy: {err * 100:5.2f}%")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(3.5, 3.3))
    plt.scatter([1, 1], [0, a], 80, marker='+', color='m')
    plt.scatter([0, 0], [0, 1], 80, marker='x', color='b')
    x_vals = np.array([-out['x'][2] / out['x'][0], -out['x'][2] / out['x'][0]])
    y_vals = np.array([-1, 1.1 * a])
    plt.plot(x_vals, y_vals, color='r')
    plt.axis([-0.1, 1.1, -1, 1.1 * a])
    plt.grid(True)
    plt.box(True)
    ld = f"NM01: {err * 100:.0f}%"
    plt.legend(['Positive', 'Negative', ld], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
