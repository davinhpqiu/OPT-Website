# Solving support vector machine using real datasets

import os
import numpy as np
import scipy.io as sio
from solver.examples.svm.funcSVM import funcSVM
from solver.NM01 import NM01
from solver.examples.svm.normalization import normalization

test = 0
data = ['arce', 'fabc']
prob = data[test]

# Expect files like 'arce.mat' and 'arce_class.mat' alongside this script
script_dir = os.path.dirname(__file__)
feat_path = os.path.join(script_dir, f"{prob}.mat")
label_path = os.path.join(script_dir, f"{prob}_class.mat")
if not (os.path.exists(feat_path) and os.path.exists(label_path)):
    raise FileNotFoundError(
        f"Dataset files not found: '{feat_path}', '{label_path}'. "
        "Place them alongside this script or update the paths.")

samp = sio.loadmat(feat_path)
label = sio.loadmat(label_path)

A = normalization(samp['X'], 2)
c = label['y'].flatten()
c[c != 1] = -1
m, n0 = A.shape

func = lambda x, key: funcSVM(x, key, 1e-4, A, c)
B = (-c[:, None]) * np.hstack((A, np.ones((m, 1))))
b = np.ones(m)

pars = {}
pars['tau'] = 1
lam = 10

out = NM01(func, B, b, lam, pars)
acc = 1 - np.count_nonzero(np.sign(np.hstack((A, np.ones((m, 1)))) @ out['sol']) - c) / m

print(' Training  Size:       %d x %d' % (m, n0))
print(' Training  Time:       %5.3fsec' % out['time'])
print(' Training  Accuracy:   %5.2f%%' % (acc * 100))
print(' Training  Objective:  %5.3e' % out['obj'])
