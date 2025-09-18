import numpy as np
import matplotlib.pyplot as plt


def plot_recovery(xo, x, pos=(100, 100, 800, 400), ind=True, show_info=None):
    """Python port of PlotRecovery.m for sparse recovery visualisation."""
    if show_info is not None:
        ind = bool(show_info)

    xo = np.asarray(xo).ravel()
    x = np.asarray(x).ravel()
    n = x.size

    fig = plt.figure(figsize=(pos[2] / 100, pos[3] / 100))
    try:
        fig.canvas.manager.set_window_title('Plot Recovery')
    except Exception:
        pass
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])

    gt_idx = np.flatnonzero(xo) + 1  # MATLAB-style indexing in the plot
    rc_idx = np.flatnonzero(x) + 1

    gt_marker, gt_stems, _ = ax.stem(gt_idx, xo[gt_idx - 1],
                                     linefmt='o-', markerfmt='o', basefmt=' ',
                                     label='Ground-Truth')
    rc_marker, rc_stems, _ = ax.stem(rc_idx, x[rc_idx - 1],
                                     linefmt='o:', markerfmt='o', basefmt=' ',
                                     label='Recovered')

    plt.setp(gt_stems, color='#f26419', linewidth=1)
    plt.setp(gt_marker, markersize=7)
    plt.setp(rc_stems, color='#1c8ddb', linewidth=1)
    plt.setp(rc_marker, markersize=4)

    xx = np.concatenate((xo, x))
    ymin_candidates = [-0.1]
    ymax_candidates = [0.2]
    if np.any(xx < 0):
        ymin_candidates.append(np.min(xx[xx < 0]) - 0.1)
    if np.any(xx > 0):
        ymax_candidates.append(np.max(xx[xx > 0]) + 0.1)
    ax.set_xlim([1, n])
    ax.set_ylim([min(ymin_candidates), max(ymax_candidates)])
    ax.grid(True)

    if ind:
        rec_acc = np.linalg.norm(x - xo) / max(np.linalg.norm(x), 1e-16)
        ax.set_title(f'Recovery accuracy = {rec_acc:.4g}', fontweight='normal')
        ax.legend()

    plt.show()


def PlotRecovery(xo, x, pos=(100, 100, 800, 400), ind=True):
    """Backward-compatible alias for MATLAB-style calls."""
    return plot_recovery(xo, x, pos=pos, ind=ind)
