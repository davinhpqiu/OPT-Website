from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_recovery(x_true: Sequence[float], x_est: Sequence[float], show_title: bool = True):
    """Python port of ``PlotRecovery.m``."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('matplotlib is required for plotting') from exc

    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.stem(np.flatnonzero(x_true), x_true[x_true != 0], linefmt='C1-', markerfmt='C1o', basefmt='k-')
    ax.stem(np.flatnonzero(x_est), x_est[x_est != 0], linefmt='C0:', markerfmt='C0o', basefmt='k-')
    ax.grid(True)

    neg_vals = np.concatenate([x_true[x_true < 0], x_est[x_est < 0]])
    pos_vals = np.concatenate([x_true[x_true > 0], x_est[x_est > 0]])
    ymin = (neg_vals.min() if neg_vals.size else 0.0) - 0.1
    ymax = (pos_vals.max() if pos_vals.size else 0.0) + 0.1
    ax.set_xlim(0, len(x_true))
    ax.set_ylim(ymin, ymax)

    if show_title:
        relerr = np.linalg.norm(x_est - x_true) / (np.linalg.norm(x_est) + 1e-12)
        ax.set_title(f'Recovery accuracy = {relerr:0.4g}')
        ax.legend(['Ground-Truth', 'Recovered'])

    try:
        plt.show()
    except Exception:
        pass

    return fig, ax
