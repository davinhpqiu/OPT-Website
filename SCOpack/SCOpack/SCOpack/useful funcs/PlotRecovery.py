import numpy as np
import matplotlib.pyplot as plt


def PlotRecovery(xo: np.ndarray, x: np.ndarray, pos=None, ind: int = 1):
    if pos is None:
        pos = [900, 500, 500, 250]
    plt.figure(figsize=(pos[2] / 100, pos[3] / 100))
    ax = plt.gca()
    ax.set_position([0.05, 0.1, 0.9, 0.8])
    xo = np.asarray(xo).reshape(-1)
    x = np.asarray(x).reshape(-1)
    idx_xo = np.flatnonzero(xo)
    idx_x = np.flatnonzero(x)
    ax.stem(idx_xo + 1, xo[idx_xo], linefmt='-', markerfmt='o', basefmt=' ', label='Ground-Truth',
            use_line_collection=True)
    ax.stem(idx_x + 1, x[idx_x], linefmt=':', markerfmt='o', basefmt=' ', label='Recovered',
            use_line_collection=True)
    ax.grid(True)
    ymin, ymax = -0.1, 0.2
    xx = np.concatenate([xo, x])
    if np.any(xx < 0):
        ymin = float(np.min(xx[xx < 0]) - 0.1)
    if np.any(xx > 0):
        ymax = float(np.max(xx[xx > 0]) + 0.1)
    ax.set_xlim(1, len(x))
    ax.set_ylim(ymin, ymax)
    if ind:
        snr = np.linalg.norm(x - xo) / max(1e-16, np.linalg.norm(x))
        title = f"Recovery accuracy = {snr:.4g}"
        ax.set_title(title, fontweight='normal')
        ax.legend()
    plt.show()

