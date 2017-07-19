from builtins import range
from builtins import object
import os
import json
from collections import OrderedDict

import scipy.sparse as sp

from .inferencer import IForest, LeafComputer, Blender, IForestBlender

class Inferencer(object):
    """
    Loads up a model for inferencing
    """
    def __init__(self, dname, gamma=30, blend=0.8, leaf_probs=False):
        with open(os.path.join(dname, 'settings'), 'rt') as f:
            self.__dict__.update(json.load(f))

        self.gamma = gamma
        self.blend = blend
        self.leaf_probs = leaf_probs

        forest = IForest(dname, self.n_trees, self.n_labels)
        if self.leaf_classifiers:
            lc = LeafComputer(dname)
            predictor = Blender(forest, lc)
        else:
            predictor = IForestBlender(forest)

        self.predictor = predictor

    def predict(self, X, fmt='sparse'):
        assert fmt in ('sparse', 'dict')
        s = []
        num = X.shape[0] if isinstance(X, sp.csr_matrix) else len(X)
        for i in range(num):
            Xi = X[i]
            mean = self.predictor.predict(Xi.data, Xi.indices, 
                    self.blend, self.gamma, self.leaf_probs)

            if fmt == 'sparse':
                s.append(mean)

            else:
                od = OrderedDict()
                for idx in reversed(mean.data.argsort()):
                    od[mean.indices[idx]] = mean.data[idx]
                
                s.append(od)

        if fmt == 'sparse':
            return sp.vstack(s)

        return s

