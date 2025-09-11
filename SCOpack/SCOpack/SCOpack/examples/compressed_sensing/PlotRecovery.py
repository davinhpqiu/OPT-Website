import numpy as np
import matplotlib.pyplot as plt

def plot_recovery(xo, x, pos=(100, 100, 800, 400), ind=True, show_info=None):
    # show_info is an alias for ind for compatibility with demos
    if show_info is not None:
        ind = bool(show_info)

    xo = np.asarray(xo).flatten()
    x = np.asarray(x).flatten()
    n = len(x)

    fig = plt.figure(figsize=(pos[2]/100, pos[3]/100))
    try:
        fig.canvas.manager.set_window_title('Plot Recovery')
    except Exception:
        pass
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])

    # Plot ground truth and recovered stems with 1-based indices to match MATLAB axes
    gt_idx = np.where(xo != 0)[0] + 1
    rc_idx = np.where(x != 0)[0] + 1

    gt_marker, gt_stems, gt_base = ax.stem(gt_idx, xo[gt_idx - 1],
                                           linefmt='o-', markerfmt='o', basefmt=' ',
                                           label='Ground-Truth')
    rc_marker, rc_stems, rc_base = ax.stem(rc_idx, x[rc_idx - 1],
                                           linefmt='o:', markerfmt='o', basefmt=' ',
                                           label='Recovered')

    # Style using setp to support both Line2D and LineCollection cases
    import matplotlib.pyplot as _plt
    _plt.setp(gt_stems, color='#f26419', linewidth=1)
    _plt.setp(gt_marker, markersize=7)
    _plt.setp(rc_stems, color='#1c8ddb', linewidth=1)
    _plt.setp(rc_marker, markersize=4)

    xx = np.concatenate((xo, x))
    ymin = min(np.min(xx[xx < 0]) - 0.1 if np.any(xx < 0) else -0.1, -0.1)
    ymax = max(np.max(xx[xx > 0]) + 0.1 if np.any(xx > 0) else 0.2, 0.2)
    ax.set_xlim([1, n])
    ax.set_ylim([ymin, ymax])
    ax.grid(True)

    if ind:
        # Match MATLAB: Recovery accuracy = ||x - xo|| / ||x|| (lower is better)
        rec_acc = np.linalg.norm(x - xo) / max(1e-16, np.linalg.norm(x))
        title = f"Recovery accuracy = {rec_acc:.4g}"
        ax.set_title(title, fontweight='normal')
        ax.legend()

    plt.show()


# Backward-compatible alias used by demo scripts
def PlotRecovery(xo, x, pos=(100, 100, 800, 400), ind=True):
    return plot_recovery(xo, x, pos=pos, ind=ind)
