from __future__ import annotations

from typing import Iterable

import numpy as np


def plot_sps(vector: Iterable[float], block_size: int = 1):
    """Python port of ``PlotSCCA.m``."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('matplotlib is required for plotting') from exc

    v = np.asarray(list(vector), dtype=float)
    n = v.size
    keep = np.concatenate([np.arange(min(50 * block_size, n)), np.arange(max(n - 150 * block_size, 0), n)])
    v = v[keep]

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(np.arange(v.size), np.zeros_like(v), '-', linewidth=1, color='#f26419')
    ax.stem(np.flatnonzero(v), v[v != 0], linefmt='C0-', markerfmt='C0o', basefmt='k-')
    ax.grid(True)

    if np.any(v != 0):
        y = np.max(np.abs(v[v != 0]))
    else:
        y = 1.0
    ax.set_ylim(-1.05 * y, 1.05 * y)

    if block_size == 1:
        ax.set_xticks([0, min(49, v.size - 1), min(149, v.size - 1), min(199, v.size - 1)])
        ax.set_xticklabels(['1', '50', '450', '500'])
    else:
        ax.set_xticks([0, min(249, v.size - 1), min(749, v.size - 1), min(999, v.size - 1)])
        ax.set_xticklabels(['1', '250', '2250', '2500'])

    try:
        plt.show()
    except Exception:
        pass

    return fig, ax
