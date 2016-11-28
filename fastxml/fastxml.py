import multiprocessing
import time
from itertools import islice, repeat
from collections import Counter, OrderedDict, deque

import numpy as np

import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier

from .splitter import Splitter, PTree
from .proc import faux_fork_call, fork_call

class Node(object):
    is_leaf = False
    def __init__(self, left, right, w, b):
        self.left = left
        self.right = right
        self.w = w
        self.b = b

class Leaf(object):
    is_leaf = True
    def __init__(self, probs):
        self.probs = probs

class CLF(object):
    __slots__ = ('w', 'b')
    def __init__(self, w, bias):
        self.w = w
        self.b = bias

def stack(X):
    stacker = np.vstack if isinstance(X[0], np.ndarray) else sp.vstack
    return stacker(X)

class FastXML(object):

    def __init__(self, n_trees=1, max_leaf_size=10, max_labels_per_leaf=20,
            re_split=False, n_jobs=1, alpha=1e-4, n_epochs=2,
            bias=True, propensity=False, A=0.55, B=1.5, 
            data_split=1, loss='log', retry_re_split=5, sparsify=True,
            verbose=False, seed=2016):
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size
        self.max_labels_per_leaf = max_labels_per_leaf
        self.re_split = re_split
        self.n_jobs = n_jobs if n_jobs > 0 else (multiprocessing.cpu_count() + 1 + n_jobs)
        self.alpha = alpha

        if isinstance(seed, np.random.RandomState):
            seed = np.randint(0, np.iinfo(np.int32).max)

        self.seed = seed
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.bias = bias
        self.propensity = propensity
        self.A = A
        self.B = B
        self.data_split = data_split
        self.loss = loss
        self.retry_re_split = retry_re_split
        self.sparsify = sparsify
        self.roots = []

    def split_node(self, idxs, splitter, rs):
        if self.verbose and len(idxs) > 1000:
            print "Splitting {}".format(len(idxs))

        return splitter.split_node(idxs, rs)

    def compute_probs(self, y, idxs, ml):
        counter = Counter(yi for i in idxs for yi in y[i])
        total = float(len(idxs))
        i, j, v = [], [], []
        for l, val in counter.most_common(self.max_labels_per_leaf):
            i.append(0)
            j.append(l)
            v.append(val / total)

        return sp.csr_matrix((v, (i, j)), shape=(1, ml)).astype('float32')

    def train_clf(self, X, idxss, rs, tries=0):
        X_train = []
        y_train = []

        for i, idxs in enumerate(idxss):
            for idx in idxs:
                X_train.append(X[idx])

            y_train.extend([i] * len(idxs))

        clf = SGDClassifier(loss=self.loss, penalty='l1', n_iter=self.n_epochs + tries, 
                alpha=self.alpha, fit_intercept=self.bias, class_weight='balanced',
                random_state=rs)

        clf.fit(stack(X_train), y_train)
        if self.sparsify:
            clf.sparsify()

        # Halves the memory requirement
        clf.coef_ = clf.coef_.astype('float32')
        if self.bias:
            clf.intercept_ = clf.intercept_.astype('float32')

        return clf, CLF(clf.coef_, clf.intercept_)

    def __reduce__(self):
        d = self.__dict__.copy()
        if 'f_roots' in d:
            del d['f_roots']

        return (FastXML, (), d)

    def __setstate__(self, d):
        self.__dict__.update(d)
        if 'roots' in d:
            self.optimize()

    @staticmethod
    def resplit_data(X, idxs, clf, classes):
        X_train = [X[i] for i in idxs]
        new_idxs = [[] for _ in xrange(classes)]
        for i, k in enumerate(clf.predict(stack(X_train))):
            new_idxs[k].append(idxs[i])

        return new_idxs

    def split_train(self, X, idxs, splitter, rs, tries=0):
        l_idx, r_idx = self.split_node(idxs, splitter, rs)

        clf = clf_fast = None
        if l_idx and r_idx:
            # Train the classifier
            if self.verbose and len(idxs) > 1000:
                print "Training classifier"

            clf, clf_fast = self.train_clf(X, [l_idx, r_idx], rs, tries)

        return l_idx, r_idx, (clf, clf_fast)

    def grow_tree(self, X, y, idxs, rs, splitter):
        if len(idxs) <= self.max_leaf_size:
            return Leaf(self.compute_probs(y, idxs, splitter.max_label))

        l_idx, r_idx, (clf, clff) = self.split_train(X, idxs, splitter, rs)

        if not l_idx or not r_idx:
            return Leaf(self.compute_probs(y, idxs, splitter.max_label))

        # Resplit the data
        for tries in xrange(self.retry_re_split if self.re_split else 0):

            if self.verbose and len(idxs) >= 1000:
                print "Resplit-before: {}".format((len(l_idx), len(r_idx)))

            if clf is not None:
                l_idx, r_idx = self.resplit_data(X, idxs, clf, 2)

            if self.verbose and len(idxs) >= 1000:
                print "Resplit-after: {}".format((len(l_idx), len(r_idx)))

            if l_idx and r_idx: break

            if self.verbose:
                print "Re-splitting {}".format(len(idxs))

            l_idx, r_idx, (clf, clff) = self.split_train(
                    X, idxs, splitter, rs, tries)

        if not l_idx or not r_idx:
            return Leaf(self.compute_probs(y, idxs, splitter.max_label))

        lNode = self.grow_tree(X, y, l_idx, rs, splitter)
        rNode = self.grow_tree(X, y, r_idx, rs, splitter)

        return Node(lNode, rNode, clff.w, clff.b)

    def _predict_opt(self, X):
        probs = []
        for tree in self.f_roots:
            probs.append(tree.predict(X.data, X.indices))

        return sum(probs) / len(probs)

    def optimize(self):
        self.f_roots = [PTree(r) for r in self.roots]

    def predict(self, X, fmt='sparse'):
        assert fmt in ('sparse', 'dict')
        s = []
        f = self._predict_opt if hasattr(self, 'f_roots') else self._predict_classic
        for i in xrange(X.shape[0]):
            mean = f(X[i])

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
        
    def compute_weights(self, y):
        if self.propensity:
            return 1 / self.compute_propensity(y, self.A, self.B)

        return np.ones(max(yi for ys in y for yi in ys) + 1, dtype='float32')

    @staticmethod
    def compute_propensity(y, A, B):
        """
        Computes propensity scores based on ys
        """
        Nl = Counter(yi for ys in y for yi in ys)
        N = len(y)
        C = (np.log(N) - 1) * (B + 1) ** A
        weights = []
        for i in xrange(max(Nl) + 1):
            weights.append(1. / (1 + C * np.exp(-A * np.log(Nl.get(i, 0) + B))))

        return np.array(weights, dtype='float32')

    def generate_idxs(self, dataset_len):
        if self.data_split == 1:
            return repeat(range(dataset_len))

        batch_size = int(dataset_len * self.data_split) \
                if self.data_split < 1 else self.data_split

        if batch_size > dataset_len:
            raise Exception("dataset subset is larger than dataset")

        def gen(bs):
            rs = np.random.RandomState(seed=self.seed + 1000)
            idxs = range(dataset_len)
            while True:
                rs.shuffle(idxs)
                it = iter(idxs)
                data = list(islice(it, bs))
                while data:
                    yield data
                    data = list(islice(it, bs))

        return gen(batch_size)

    def fit(self, X, y):
        if self.n_jobs > 1:
            f = fork_call(self.grow_tree)
        else:
            f = faux_fork_call(self.grow_tree)

        weights = self.compute_weights(y)

        splitter = Splitter(y, weights)
        procs = []
        finished = []
        counter = iter(xrange(self.n_trees))
        idxs = self.generate_idxs(len(X))
        while len(finished) < self.n_trees:
            if len(procs) < self.n_jobs and (len(procs) + len(finished)) < self.n_trees :
                rs = np.random.RandomState(seed=self.seed + next(counter))
                procs.append(f(X, y, next(idxs), rs, splitter))
            else:
                # Check
                _procs = []
                for p in procs:
                    if p.ready():
                        finished.append(p.get())
                    else:
                        _procs.append(p)

                # No change in readyness, just sleep
                if len(procs) == len(_procs):
                    time.sleep(0.1)

                procs = _procs

        self.roots = finished
        self.optimize()

class MetricNode(object):
    __slots__ = ('left', 'right')
    is_leaf = False
    
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def idxs(self):
        return self.left.idxs + self.right.idxs

class MetricLeaf(object):
    __slots__ = ('idxs')
    is_leaf = True

    def __init__(self, idxs):
        self.idxs = idxs

def metric_cluster(y, max_leaf_size=10, propensity=False, A=0.55, B=1.5, seed=2016, verbose=False):
    rs = np.random.RandomState(seed=seed)
    n_labels = max(yi for ys in y for yi in ys) + 1
    if propensity:
        weights = 1 / FastXML.compute_propensity(y, 0.55, 1.5)
    else:
        weights = np.ones(n_labels, dtype='float32')

    # Initialize splitter
    splitter = Splitter(y, weights)

    def _metric_cluster(idxs):
        if verbose and len(idxs) > 1000:
            print "Splitting:", len(idxs)

        if len(idxs) < max_leaf_size:
            return MetricLeaf(idxs)

        left, right = splitter.split_node(idxs, rs)
        if not left or not right:
            return MetricLeaf(idxs)

        return MetricNode(_metric_cluster(left), _metric_cluster(right))

    return _metric_cluster(range(len(y)))
