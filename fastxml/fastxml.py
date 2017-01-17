import multiprocessing
import time
from math import ceil
from itertools import islice, repeat, izip
from contextlib import closing
from collections import Counter, OrderedDict, defaultdict, deque

import numpy as np
import scipy.sparse as sp

import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

from .splitter import Splitter, sparsify, compute_leafs, sparse_mean, radius, ITree
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

class Tree(object):
    def __init__(self, rootIdx, W, b, tree, probs):
        self.rootIdx = rootIdx
        self.W = W
        self.b = b
        self.tree = tree
        self.probs = probs
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = ITree(self.rootIdx, self.W, self.b, self.tree, self.probs)

        return self._predictor

    def __reduce__(self):
        return (Tree, (self.rootIdx, self.W, self.b, self.tree, self.probs), {})

    def __setstate__(self, d):
        pass

class FastXML(object):

    def __init__(self, n_trees=1, max_leaf_size=10, max_labels_per_leaf=20,
            re_split=0, n_jobs=1, alpha=1e-4, n_epochs=2, n_updates=100, bias=True, 
            subsample=1, loss='log', sparse_multiple=25, leaf_classifiers=False,
            gamma=30, blend=0.8, leaf_eps=1e-5, optimization="fastxml", 
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
        assert isinstance(n_epochs, int) or n_epochs == 'auto'
        self.n_epochs = n_epochs
        self.n_updates = n_updates
        self.verbose = verbose
        self.bias = bias
        self.subsample = subsample
        self.loss = loss
        self.sparse_multiple = sparse_multiple
        self.leaf_classifiers = leaf_classifiers
        self.gamma = gamma
        self.blend = blend
        self.leaf_eps = leaf_eps
        assert optimization in ('fastxml', 'dsimec')
        self.optimization = optimization
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

    def build_XY(self, X, idxss, rs):
        """
        Faster sparse building
        """
        y_train = []
        idxes = []
        for i, idxs in enumerate(idxss):
            idxes.extend(idxs)
            y_train.extend([i] * len(idxs))

        # Shuffle the flattened data
        idxes, y_train = shuffle(idxes, y_train, random_state=rs)

        X_train = self.build_X(X, idxes)
        return X_train, y_train

    def compute_epochs(self, N):
        if isinstance(self.n_epochs, int):
            return self.n_epochs

        # Rules of Thumb state that SGD needs ~1mm updates to converge
        # That would take _forever_, so we set it 100 by default
        n_epochs = int(ceil(self.n_updates / N))
        assert n_epochs > 0
        return n_epochs

    def train_clf(self, X, idxss, rs):
        n_epochs = self.compute_epochs(len(idxss))
        if self.verbose and len(idxss) > 1000:
            print "Epochs:", n_epochs

        if self.optimization == 'fastxml':
            penalty = 'l1'
        else:
            penalty = 'l2'
        clf = SGDClassifier(loss=self.loss, penalty=penalty, n_iter=n_epochs, 
                alpha=self.alpha, fit_intercept=self.bias, class_weight='balanced',
                random_state=rs)

        X_train, y_train = self.build_XY(X, idxss, rs)

        clf.fit(X_train, y_train)

        # Halves the memory requirement
        eps = 1e-6 if self.optimization == 'fastxml' else 1e-2
        clf.coef_ = sparsify(clf.coef_, eps)
        if self.bias:
            clf.intercept_ = clf.intercept_.astype('float32')

        return clf, CLF(clf.coef_, clf.intercept_)

    def __reduce__(self):
        d = self.__dict__.copy()
        return (FastXML, (), d)

    def __setstate__(self, d):
        self.__dict__.update(d)

    def resplit_data(self, X, idxs, clf, classes):
        X_train = self.build_X(X, idxs)
        new_idxs = [[] for _ in xrange(classes)]
        for i, k in enumerate(clf.predict(X_train)):
            new_idxs[k].append(idxs[i])

        return new_idxs

    def split_train(self, X, idxs, splitter, rs):
        l_idx, r_idx = self.split_node(idxs, splitter, rs)

        clf = clf_fast = None
        if l_idx and r_idx:
            # Train the classifier
            if self.verbose and len(idxs) > 1000:
                print "Training classifier"

            clf, clf_fast = self.train_clf(X, [l_idx, r_idx], rs)

        return l_idx, r_idx, (clf, clf_fast)

    def grow_root(self, X, y, idxs, rs, splitter):
        node = self.grow_tree(X, y, idxs, rs, splitter)
        return self.compact(node, X[0].shape[1])

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
                    X, idxs, splitter, rs)

        if not l_idx or not r_idx:
            return Leaf(self.compute_probs(y, idxs, splitter.max_label))

        lNode = self.grow_tree(X, y, l_idx, rs, splitter)
        rNode = self.grow_tree(X, y, r_idx, rs, splitter)

        return Node(lNode, rNode, clff.w, clff.b)

    def _predict_opt(self, X):
        probs = []
        for tree in self.roots:
            probs.append(tree.predictor.predict(X.data, X.indices))

        return sum(probs) / len(probs)

    def _add_leaf_probs(self, X, ypi):
        Xn = l2norm(X)
        indices = ypi.indices.tolist()
        uxs = self.uxs_[indices]
        xrs = self.xr_[indices]

        lyp = compute_leafs(self.gamma, Xn.data, Xn.indices, uxs, xrs)

        # Blend leaf and tree probabilities
        def f():
            nps = self.blend * np.log(ypi.data) + (1 - self.blend) * np.log(np.array(lyp))
            return nps
        return sp.csr_matrix((f(), ([0] * len(lyp), indices)))

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
                yield idxs[:bs]

        return gen(batch_size)

    def _build_roots(self, X, y, weights):
        assert isinstance(X, list) and isinstance(X[0], sp.csr_matrix), "Requires list of csr_matrix"
        if self.n_jobs > 1:
            f = fork_call(self.grow_root)
        else:
            f = faux_fork_call(self.grow_root)

        nl = max(yi for ys in y for yi in ys) + 1
        if weights is None:
            weights = np.ones(nl, dtype='float32')
        else:
            assert weights.shape[0] == nl, "Weights need to be same as largest y class"

        # Initialize cython splitter
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
    
    def compact(self, root, dims):
        #CLS
        Ws = []
        bs = []

        # Tree: index, left, right, isLeaf
        tree = []

        # Payload
        probs = []

        def f(node):
            if node.is_leaf:
                treeIdx = len(probs)
                probs.append(node.probs)
                tree.append([treeIdx, 0, 0, 1])
            else:
                leftIndex = f(node.left)
                rightIndex = f(node.right)

                clfIdx = len(Ws)
                Ws.append(node.w)
                bs.append(node.b[0])
                tree.append([clfIdx, leftIndex, rightIndex, 0])

            curIdx = len(tree) - 1
            return curIdx

        rootIdx = f(root)

        if Ws:
            W_stack = sp.vstack(Ws)
        else:
            W_stack = sp.csr_matrix(([], ([], [])), shape=(0, dims)).astype('float32')

        return Tree(rootIdx, W_stack, 
                np.array(bs, dtype='float32'), 
                np.array(tree, dtype='uint32'), 
                probs)

    def fit(self, X, y, weights=None):
        
        self.roots = self._build_roots(X, y, weights)
        if self.leaf_classifiers:
            self.uxs_, self.xr_ = self._compute_leaf_probs(X, y)

    def _compute_leaf_probs(self, X, y):
        dd = defaultdict(list)
        ml = 0
        for Xi, yis in izip(X, y):
            Xin = l2norm(Xi)
            for yi in yis:
                dd[yi].append(Xin)
                ml = max(yi, ml)

        if self.verbose:
            print "Computing means and radius for hard margin"

        xmeans = []
        xrs = []
        with closing(multiprocessing.Pool(processes=self.n_jobs)) as p:
            it = ((i, dd[i], self.leaf_eps) for i in xrange(ml + 1))
            for k, ux, r in p.imap(compute_leaf_metrics, it, 100):
                if self.verbose and k % 100 == 0:
                    print "Training leaf classifier: %s of %s" % (k, ml)

                if ux is None:
                    ux = sp.csr_matrix((1, X[0].shape[1])).astype('float64')

                xmeans.append(ux)
                xrs.append(r)

        return sp.vstack(xmeans), np.array(xrs, dtype=np.float32)

def l2norm(Xi):
    return Xi / np.linalg.norm(Xi.data)

def compute_leaf_metrics(data):
    i, Xs, eps = data
    if len(Xs) > 100:
        v = np.zeros(Xs[0].shape[1], dtype='float64')
        sparse_mean(Xs, v)
        ux = sparsify(v.reshape((1, -1)), eps=eps).astype('float64')

    elif len(Xs) > 1:
        ux = sum(Xs) / len(Xs)

    else:
        return i, None, 0.0

    ux = l2norm(ux)
    rad = max(radius(ux.data, ux.indices, Xi.data, Xi.indices) for Xi in Xs)
    return i, ux, rad

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
