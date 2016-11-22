import multiprocessing
import time
from itertools import islice, repeat
from collections import Counter, OrderedDict, deque

import numpy as np

import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier

from .splitter import split_node

class Node(object):
    __slots__ = ('left', 'right', 'clf')
    def __init__(self, left, right, clf):
        self.left = left
        self.right = right
        self.clf = clf

    def traverse(self, X):
        c = self.clf.predict(X)[0]
        if c == 0:
            return self.left

        return self.right

class MNode(Node):
    __slots__ = ('nodes', 'clf')
    def __init__(self, nodes, clf):
        self.nodes = nodes
        self.clf = clf

    def traverse(self, X):
        c = self.clf.predict(X)[0]
        return self.nodes[c]

class Leaf(object):
    def __init__(self, probs):
        self.probs = probs

def stack(X):
    stacker = np.vstack if isinstance(X[0], np.ndarray) else sp.vstack
    return stacker(X)

class Result(object):

    def ready(self):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

class ForkResult(Result):
    def __init__(self, queue, p):
        self.queue = queue
        self.p = p 

    def ready(self):
        return self.queue.full()

    def get(self):
        result = self.queue.get()
        self.p.join()
        self.queue.close()
        return result

class SingleResult(Result):
    def __init__(self, res):
        self.res = res

    def ready(self):
        return True

    def get(self):
        return self.res

def _remote_call(q, f, args):
    results = f(*args)
    q.put(results)

def faux_fork_call(f):
    def f2(*args):
        return SingleResult(f(*args))

    return f2

def fork_call(f):
    def f2(*args):
        queue = multiprocessing.Queue(1)
        p = multiprocessing.Process(target=_remote_call, args=(queue, f, args))
        p.start()
        return ForkResult(queue, p)

    return f2

class FastXML(object):

    def __init__(self, n_trees=1, max_leaf_size=10, max_labels_per_leaf=20,
            re_split=False, n_jobs=1, alpha=1e-4, min_binary=1, n_epochs=2,
            downsample=None, bias=True, verbose=False, seed=2016):
        assert downsample in (None, 'float32', 'float16')
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size
        self.max_labels_per_leaf = max_labels_per_leaf
        self.re_split = re_split
        self.n_jobs = n_jobs if n_jobs > 0 else (multiprocessing.cpu_count() + 1 + n_jobs)
        self.alpha = alpha

        if isinstance(seed, np.random.RandomState):
            seed = np.randint(0, np.iinfo(np.int32).max)

        self.seed = seed
        self.min_binary = min_binary
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.downsample = downsample
        self.bias = bias
        self.roots = []

    @staticmethod
    def count_labels(y, idxs):
        it = (yi for i in idxs for yi in y[i])
        return Counter(it)

    def split_node(self, X, y, idxs, rs):
        if self.verbose and len(idxs) > 1000:
            print "Splitting {}".format(len(idxs))

        return split_node(y, idxs, rs)

    def compute_probs(self, y, idxs):
        counter = self.count_labels(y, idxs)
        total = float(sum(counter.itervalues()))
        it = ((k, v / total) for k, v in counter.iteritems())
        return OrderedDict(islice(it, self.max_labels_per_leaf))

    def train_clf(self, X, idxss, rs):
        X_train = []
        y_train = []

        for i, idxs in enumerate(idxss):
            for idx in idxs:
                X_train.append(X[idx])

            y_train.extend([i] * len(idxs))

        clf = SGDClassifier(loss='log', penalty='l1', n_iter=self.n_epochs, 
                alpha=self.alpha, fit_intercept=self.bias, class_weight='balanced',
                random_state=rs)

        clf.fit(stack(X_train), y_train)
        clf.sparsify()
        # Halves the memory requirement
        if self.downsample is not None:
            clf.coef_ = clf.coef_.astype(self.downsample)
            if self.bias:
                clf.intercept_ = clf.intercept_.astype(self.downsample)

        return clf

    @staticmethod
    def resplit(X, idxs, clf, classes):
        X_train = [X[i] for i in idxs]
        new_idxs = [[] for _ in xrange(classes)]
        for i, k in enumerate(clf.predict(stack(X_train))):
            new_idxs[k].append(idxs[i])

        return new_idxs

    def grow_mtree(self, X, y, idxs, rs):
        """
        We create an K-label discrete classifier if the idxs are less than the
        minimum binary level.  We continue by binary splitting all the indexes
        but train a single classifier instead of a tree of small classifiers.

        TODO: allow for a different classifier for multiclass to allow for
        nonlineariies.
        """
        finished = []
        idx_set = deque([idxs])
        while idx_set:
            idx = idx_set.popleft()
            if len(idx) <= self.max_leaf_size:
                finished.append(idx)
            else:
                left, right = self.split_node(X, y, idx, rs)
                if left and right:
                    idx_set.extendleft([left, right])
                else:
                    finished.append(left if left else right)

        if len(finished) == 1:
            return Leaf(self.compute_probs(y, idxs))

        # Build leafs for all the nodes
        clf = self.train_clf(X, finished, rs)
        if self.re_split:
            finished = self.resplit(X, idxs, clf, len(finished))

        leafs = [Leaf(self.compute_probs(y, idx)) for idx in finished]
        return MNode(leafs, clf)

    def grow_tree(self, X, y, idxs, rs):
        if len(idxs) <= self.max_leaf_size:
            return Leaf(self.compute_probs(y, idxs))

        if len(idxs) < self.min_binary:
            return self.grow_mtree(X, y, idxs, rs)
        
        l_idx, r_idx = self.split_node(X, y, idxs, rs)

        if not l_idx or not r_idx:
            return Leaf(self.compute_probs(y, idxs))

        # Train the classifier
        if self.verbose and len(idxs) > 1000:
            print "Training classifier"

        clf = self.train_clf(X, [l_idx, r_idx], rs)

        # Resplit the data
        if self.re_split:
            if self.verbose and len(idxs) > 1000:
                print "Pre-split:", len(l_idx), len(r_idx)

            l_idx, r_idx = self.resplit(X, idxs, clf, 2)
            if self.verbose and len(idxs) > 1000:
                print "Post-split:", len(l_idx), len(r_idx)

        if not l_idx or not r_idx:
            return Leaf(self.compute_probs(y, idxs))

        lNode = self.grow_tree(X, y, l_idx, rs)
        rNode = self.grow_tree(X, y, r_idx, rs)

        return Node(lNode, rNode, clf)

    def predict(self, X):
        probs = [{}]
        for tree in self.roots:
            while not isinstance(tree, Leaf):
                tree = tree.traverse(X)

            probs.append(tree.probs)
    
        def merge(d1, d2):
            for k, v in d2.iteritems():
                d1[k] = d1.get(k,0) + v

            return d1
        
        res = reduce(merge, probs)
        xs = sorted(res.iteritems(), key=lambda x: x[1], reverse=True)
        return OrderedDict((k, v / len(self.roots)) for k, v in xs)

    def fit(self, X, y):
        if self.n_jobs > 1:
            f = fork_call(self.grow_tree)
        else:
            f = faux_fork_call(self.grow_tree)

        procs = []
        finished = []
        counter = iter(xrange(self.n_trees))
        while len(finished) < self.n_trees:
            if len(procs) < self.n_jobs and (len(procs) + len(finished)) < self.n_trees :
                rs = np.random.RandomState(seed=self.seed + next(counter))
                procs.append(f(X, y, range(len(X)), rs))
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

class MetricNode(object):
    __slots__ = ('left', 'right')
    is_leaf = False
    
    def __init__(self, left, right):
        self.left = left
        self.right = right

class MetricLeaf(object):
    __slots__ = ('idxs')
    is_leaf = True

    def __init__(self, idxs):
        self.idxs = idxs


def metric_cluster(y, max_leaf_size=10, seed=2016):
    rs = np.random.RandomState(seed=seed)
    def _metric_cluster(idxs):
        if len(idxs) < max_leaf_size:
            return MetricLeaf(idxs)

        left, right = split_node(y, idxs, rs, 50)
        if not left or not right:
            return MetricLeaf(idxs)

        return MetricNode(_metric_cluster(left), _metric_cluster(right))

    return _metric_cluster(range(len(y)))
