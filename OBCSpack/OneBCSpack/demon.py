import numpy as np

try:
    from .OneBCSpack import OneBCSpack as run_obcspack
    from .OneBCSpack.funcs import plot_recovery
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from OBCSpack.OneBCSpack import OneBCSpack as run_obcspack
    from OBCSpack.OneBCSpack.OneBCSpack.funcs import plot_recovery


def main():
    rng = np.random.default_rng()

    n = 2000
    m = int(np.ceil(0.5 * n))
    s = int(np.ceil(0.01 * n))
    nf = 0.05
    r = 0.02
    k = int(np.ceil(r * m))

    A = rng.standard_normal((m, n))
    T = rng.permutation(n)[:s]
    xo = np.zeros(n)
    xo[T] = (1 + rng.random(s)) * np.sign(rng.standard_normal(s))
    norm = np.linalg.norm(xo[T])
    if norm > 0:
        xo[T] /= norm

    bo = np.sign(A[:, T] @ xo[T] + nf * rng.standard_normal(m))
    bo[bo == 0] = 1.0
    h = np.ones(m)
    idx = rng.permutation(m)[:k]
    h[idx] *= -1
    b = bo * h

    solver = ['GPSP', 'NM01']
    out = run_obcspack(A, b, s, k, solver[0])

    err = np.linalg.norm(xo - out['sol'])
    snr = -10 * np.log10(err**2 + 1e-12)
    hd = np.count_nonzero(np.sign(A @ out['sol']) - b) / m
    he = np.count_nonzero(np.sign(A @ out['sol']) - bo) / m

    print(f" Time:                  {out['time']:.3f} sec")
    print(f" Absolue error:         {err * 100:6.2f} %")
    print(f" Signal-to-noise ratio: {snr:6.2f}")
    print(f" Hamming distence:      {hd:6.3f}")
    print(f" Hamming error:         {he:6.3f}")
    plot_recovery(xo, out['sol'], [1000, 450, 500, 250], True)


if __name__ == '__main__':
    main()
