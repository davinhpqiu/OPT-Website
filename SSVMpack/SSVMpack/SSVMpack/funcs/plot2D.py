from __future__ import annotations

from typing import Sequence

import numpy as np

_figure_counter = 0

def plot2D(Atr: Sequence[Sequence[float]], ctr: Sequence[float], x=None, label: str | None = None, acc: float | None = None):
    """Python translation of the 2D plotting helper."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('matplotlib is required for plotting') from exc

    Atr = np.asarray(Atr, dtype=float)
    ctr = np.asarray(ctr, dtype=float)

    global _figure_counter
    _figure_counter += 1
    fig = plt.figure(figsize=(3, 3))
    try:
        fig.canvas.manager.set_window_title(f'Figure {_figure_counter}')
    except Exception:
        pass
    siz = 50
    at1 = Atr[ctr == 1]
    at2 = Atr[ctr == -1]
    x0 = np.array([-2, 2])
    y0 = 2.5 * x0

    plt.scatter(at1[:, 0], at1[:, 1], siz, marker='+', color='m', linewidths=1.5)
    plt.scatter(at2[:, 0], at2[:, 1], siz, marker='x', color='b', linewidths=1.5)
    plt.plot(x0, y0, color='black', linestyle=':', linewidth=1.5)
    plt.axis([-2, 2, min(Atr[:, 1]), max(Atr[:, 1])])
    plt.grid(True)
    plt.box(True)

    if x is not None:
        x = np.asarray(x, dtype=float)
        y_line = -x[0] / x[1] * x0 - x[2] / x[1]
        plt.plot(x0, y_line, color='g', linewidth=1.5)
        legend_labels = ['Positive', 'Negative', 'Bayes']
        if label:
            legend_labels.append(label)
        plt.legend(legend_labels, loc='upper right')
        if acc is not None:
            plt.title(f'Accuracy: {acc * 100:7.2f}%')
        plt.axis([-2, 2, Atr[:, 1].min() - 0.1, Atr[:, 1].max() + 1.5])
    plt.show()
