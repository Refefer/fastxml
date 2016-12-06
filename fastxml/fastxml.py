import multiprocessing
import time
from itertools import islice, repeat, izip
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import scipy.sparse as sp

import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier

from .splitter import Splitter, PTree, sparsify
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
            re_split=0, n_jobs=1, alpha=1e-4, n_epochs=2, bias=True, 
            subsample=1, loss='log', sparse_multiple=25, leaf_classifiers=False,
            gamma=30, blend=0.5, verbose=False, seed=2016):

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
        self.subsample = subsample
        self.loss = loss
        self.sparse_multiple = sparse_multiple
        self.leaf_classifiers = leaf_classifiers
        self.gamma = gamma
        self.blend = blend
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

    def build_X(self, X, idxs):
        indptr = [0]
        indices = []
        data = []
        for idx in idxs:
            s = X[idx]
            indices.append(s.indices)
            data.append(s.data)
            indptr.append(indptr[-1] + s.indices.shape[0])

        X_train = sp.csr_matrix((len(data), X[0].shape[1]), dtype=X[0].dtype.name)
        X_train.indptr  = np.array(indptr, dtype=np.int32)
        X_train.indices = np.concatenate(indices)
        X_train.data    = np.concatenate(data)
        return X_train

    def build_XY(self, X, idxss):
        y_train = []

        idxes = []
        for i, idxs in enumerate(idxss):
            idxes.extend(idxs)
            y_train.extend([i] * len(idxs))

        X_train = self.build_X(X, idxes)
        return X_train, y_train


    def train_clf(self, X, idxss, rs, tries=0):
        clf = SGDClassifier(loss=self.loss, penalty='l1', n_iter=self.n_epochs + tries, 
                alpha=self.alpha, fit_intercept=self.bias, class_weight='balanced',
                random_state=rs)

        X_train, y_train = self.build_XY(X, idxss)

        clf.fit(X_train, y_train)

        # Halves the memory requirement
        clf.coef_ = sparsify(clf.coef_)
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

    def resplit_data(self, X, idxs, clf, classes):
        X_train = self.build_X(X, idxs)
        new_idxs = [[] for _ in xrange(classes)]
        for i, k in enumerate(clf.predict(X_train)):
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
        for tries in xrange(self.re_split):

            if clf is not None:
                l_idx, r_idx = self.resplit_data(X, idxs, clf, 2)

            if l_idx and r_idx: break

            if self.verbose and len(idxs) > 1000:
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

    def _add_leaf_probs(self, X, ypi):
        Xn = self._l2norm(X)
        indices = []
        lyp = []
        for yi in ypi.indices:
            ux = self.uxs_[yi]
            xr = self.xr_[yi]

            # distance
            hl = ((Xn - ux).data ** 2).sum()
            k = np.exp(self.gamma  * (hl - xr)) 
            indices.append(yi)
            lyp.append(1. / (1. + k))

        # Blend leaf and tree probabilities
        nps = self.blend * ypi.data + (1 - self.blend) * np.array(lyp)
        return sp.csr_matrix((nps, ([0] * len(lyp), indices)))

    def predict(self, X, fmt='sparse'):
        assert fmt in ('sparse', 'dict')
        s = []
        for i in xrange(X.shape[0]):
            mean = self._predict_opt(X[i])
            if self.leaf_classifiers and self.blend < 1:
                mean = self._add_leaf_probs(X[i], mean)

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
        
    def generate_idxs(self, dataset_len):
        if self.subsample == 1:
            return repeat(range(dataset_len))

        batch_size = int(dataset_len * self.subsample) \
                if self.subsample < 1 else self.subsample

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

    def _build_roots(self, X, y, weights):
        assert isinstance(X, list) and isinstance(X[0], sp.csr_matrix), "Requires list of csr_matrix"
        if self.n_jobs > 1:
            f = fork_call(self.grow_tree)
        else:
            f = faux_fork_call(self.grow_tree)

        nl = max(yi for ys in y for yi in ys) + 1
        if weights is None:
            weights = np.ones(nl, dtype='float32')
        else:
            assert weights.shape[0] == nl, "Weights need to be same as largest y class"

        splitter = Splitter(y, weights, self.sparse_multiple)

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

        return finished

    def fit(self, X, y, weights=None):
        
        self.roots = self._build_roots(X, y, weights)
        # Optimize for inference
        self.optimize()

        if self.leaf_classifiers:
            self.uxs_, self.xr_ = self._compute_leaf_probs(X, y)

    def _l2norm(self, Xi):
        return Xi / np.linalg.norm(Xi.data)

    def _compute_leaf_probs(self, X, y):
        dd = defaultdict(list)
        ml = 0
        for Xi, yis in izip(X, y):
            Xin = self._l2norm(Xi)
            for yi in yis:
                dd[yi].append(Xin)
                ml = max(yi, ml)

        if self.verbose:
            print "Computing means and radius for hard margin"

        xmeans = []
        xrs = []
        for i in xrange(ml):
            if self.verbose and i % 100 == 0:
                print "Training leaf classifier: %s of %s" % (i, ml)

            ux = sum(dd[i]) / len(dd[i])
            xmeans.append(ux)
            xrs.append(max(self._radius(ux, Xi) for Xi in dd[i]))

        return sp.vstack(xmeans), np.array(xrs, dtype=np.float32)

    @staticmethod
    def _radius(Xu, Xui):
        return ((Xu - Xui).data ** 2).sum()

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

def metric_cluster(y, weights=None, max_leaf_size=10, 
        sparse_multiple=25, seed=2016, verbose=False):
    rs = np.random.RandomState(seed=seed)
    n_labels = max(yi for ys in y for yi in ys) + 1
    if weights is None:
        weights = np.ones(n_labels, dtype='float32')

    # Initialize splitter
    splitter = Splitter(y, weights, sparse_multiple)

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
