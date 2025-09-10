# Solving support vector machine using four synthetic samples

import numpy as np
import matplotlib.pyplot as plt
from solver.examples.svm.funcSVM import funcSVM
from solver.NM01 import NM01

a = 10
A = np.array([[0, 0], [0, 1], [1, 0], [1, a]])
c = np.array([-1, -1, 1, 1])
m, n = A.shape

func = lambda x, key: funcSVM(x, key, 1e-4, A, c)
B = (-c[:, None]) * np.hstack((A, np.ones((m, 1))))
b = np.ones((m, 1))
lam = 10

pars = {}
pars['tau'] = 1
pars['strict'] = 1

out = NM01(func, B, b, lam, pars)
x = out['sol']

fig = plt.figure(figsize=(3.5, 3.3), dpi=100)
ax = fig.add_axes([0.08, 0.08, 0.88, 0.88])

ax.scatter([1, 1], [0, a], s=80, marker='+', color='m')  # Positive
ax.scatter([0, 0], [0, 1], s=80, marker='x', color='b')  # Negative

# Decision boundary: x1 * w1 + x2 * w2 + bias = 0 → x1 = -bias / w1
if x[0] != 0:
    boundary_x = -x[2] / x[0]
    ax.plot([boundary_x, boundary_x], [-1, 1.1 * a], color='r')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-1, 1.1 * a)
ax.grid(True)
ax.set_box_aspect(1)

ld = f'NM01: {func(x, "a") * 100:.0f}%'
ax.legend(['Positive', 'Negative', ld], loc='upper left')

plt.show()