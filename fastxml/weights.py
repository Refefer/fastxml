from __future__ import division
from builtins import range
from past.utils import old_div
from collections import Counter
import numpy as np

def metrics(y):
    Nl = Counter(yi for ys in y for yi in ys)
    N = len(y)
    return N, Nl, max(Nl) + 1

def uniform(y):
    N, Nl, ml = metrics(y)
    return np.ones(ml, dtype='float32')

def propensity(y, A=0.55, B=1.5):
    """
    Computes propensity scores based on ys
    """
    N, Nl, ml = metrics(y)
    C = (np.log(N) - 1) * (B + 1) ** A
    weights = []
    for i in range(ml):
        weights.append(1 + C * (Nl.get(i, 0) + B) ** -A)

    return np.array(weights, dtype='float32')

def nnllog(y, a=1, b=0):
    N, Nl, ml = metrics(y)
    N = float(N)

    weights = []
    for i in range(ml):
        if i in Nl:
            weights.append(a * np.log(old_div(N, Nl[i])) + b)
        else:
            weights.append(0)

    return np.array(weights, dtype='float32')

def logexp(y, a=1, b=1):
    N, Nl, ml = metrics(y)
    weights = []
    for i in range(ml):
        if i in Nl:
            weights.append(a * np.log(1 + Nl[i]) ** -b)
        else:
            weights.append(0)

    return np.array(weights, dtype='float32')
