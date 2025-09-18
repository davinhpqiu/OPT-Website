import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for path in (str(_PARENT), str(_HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from SROpack.SROpack.solver.NL0R import NL0R
from SROpack.SROpack.examples.funcSimpleEx import funcSimpleEx


def main():
    n = 2
    lam = 0.5
    pars = {'eta': 0.1}
    f = lambda x, key, T1, T2: funcSimpleEx(x, key, T1, T2)
    out = NL0R(f, n, lam, pars)
    print(f" Objective:      {out['obj']:.4f}")
    print(f" CPU time:      {out['time']:.3f}sec")
    print(f" Iterations:        {out['iter']:4d}")


if __name__ == '__main__':
    main()
