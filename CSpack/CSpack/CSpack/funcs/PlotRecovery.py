from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_recovery(x_true: Sequence[float], x_est: Sequence[float], pos, show_stats: bool = True):
    """Python port of the MATLAB recovery plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('matplotlib is required for plotting') from exc

    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)

    width = pos[2] / 100 if isinstance(pos, (list, tuple)) and len(pos) >= 4 else 5
    height = pos[3] / 100 if isinstance(pos, (list, tuple)) and len(pos) >= 4 else 3
    fig, ax = plt.subplots(figsize=(width, height))

    nz_true = np.flatnonzero(x_true)
    nz_est = np.flatnonzero(x_est)
    ax.stem(nz_true + 1, x_true[nz_true], linefmt='C1-', markerfmt='C1o', basefmt='k-')
    ax.stem(nz_est + 1, x_est[nz_est], linefmt='C0:', markerfmt='C0o', basefmt='k-')
    ax.grid(True)

    combined = np.concatenate([x_true, x_est])
    neg = combined[combined < 0]
    pos_vals = combined[combined > 0]
    ymin = (neg.min() if neg.size else -0.1) - 0.1
    ymax = (pos_vals.max() if pos_vals.size else 0.2) + 0.1
    ax.set_xlim(1, len(x_true))
    ax.set_ylim(ymin, ymax)

    if show_stats:
        relerr = np.linalg.norm(x_est - x_true) / (np.linalg.norm(x_est) + 1e-12)
        overlap = np.logical_and(x_est != 0, x_true != 0)
        mis_support = max(int(np.count_nonzero(x_true)) - int(np.count_nonzero(overlap)), 0)
        ax.set_title(f'Recovery accuracy ={relerr:0.4g}, Number of mis-supports ={mis_support}')
        ax.legend(['Ground-Truth', 'Recovered'])

    plt.show()
    return fig, ax
