import json
import multiprocessing
import sys
import math
import time
from itertools import islice, chain, repeat
from collections import Counter, OrderedDict, deque

import numpy as np

import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import FeatureHasher

MAX_LABELS = 15000000
LOGS = (1 / np.arange(2, MAX_LABELS + 2, dtype='float32'))

class Node(object):
    def __init__(self, left, right, clf):
        self.left = left
        self.right = right
        self.clf = clf

    def traverse(self, X):
        if self.clf.predict(X) == 0:
            return self.left

        return self.right

class MNode(Node):
    def __init__(self, nodes, clf):
        self.nodes = nodes
        self.clf = clf

    def traverse(self, X):
        return self.nodes[self.clf.predict(X)[0]]

class Leaf(object):
    def __init__(self, probs):
        self.probs = probs

def ndcg(order, ls):
    score = idcg = 0
    for i, l in enumerate(ls):
        idcg += 1 / math.log(2+i)
        if l in order:
            score += 1 / math.log(order[l] + 2)

    return score / idcg

def dcg(order, ls):
    """
    We only need to use DCG since we're only using it to determine which partition
    bucket the label set in
    """
    return sum(LOGS[order[l]] for l in ls if l in order)

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
            re_split=False, even_split=False, n_jobs=1, classifier='sgd',
            min_binary=1, seed=2016):
        assert classifier in ('sgd', 'liblinear')
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size
        self.max_labels_per_leaf = max_labels_per_leaf
        self.re_split = re_split
        self.even_split = even_split
        self.n_jobs = n_jobs
        self.classifier = classifier
        self.seed = seed
        self.min_binary = min_binary

    @staticmethod
    def count_labels(y, idxs):
        it = (yi for i in idxs for yi in y[i])
        return Counter(it)

    def order_labels(self, y, idxs):
        counter = self.count_labels(y, idxs)
        sLabels = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
        return {l: i for i, l in enumerate(sLabels)}

    @staticmethod
    def divide(y, idxs, lOrder, rOrder):
        newLeft, newRight = [], []
        for i in idxs:
            lNdcg = dcg(lOrder, y[i])
            rNdcg = dcg(rOrder, y[i])
            if lNdcg >= rNdcg:
                newLeft.append(i)
            else:
                newRight.append(i)

        return newLeft, newRight

    def split_node(self, X, y, idxs, rs):
        if len(idxs) > 1000:
            print "Splitting: {}".format(len(idxs))

        if self.even_split:
            ix = idxs[:]
            rs.shuffle(ix)
            left = ix[:len(ix)/2]
            right = ix[len(ix)/2:]
        else:
            left, right = [], []
            for i in idxs:
                (left if rs.rand() < 0.5 else right).append(i)

        iterations = 0
        while True:
            if len(idxs) > 1000:
                print "Iterations", iterations
            iterations += 1

            # Build ndcg for the sides
            lOrder = self.order_labels(y, left)
            rOrder = self.order_labels(y, right)

            # Divide out the sides
            left1, right1 = self.divide(y, left, lOrder, rOrder)
            left2, right2 = self.divide(y, right, lOrder, rOrder)
            if not right1 and not left2:
                # Done!
                break

            left = left1 + left2
            right = right1 + right2

        return left, right

    def compute_probs(self, y, idxs):
        counter = self.count_labels(y, idxs)
        total = float(sum(counter.itervalues()))
        it = ((k, v / total) for k, v in counter.iteritems())
        return OrderedDict(islice(it, self.max_labels_per_leaf))

    def train_clf(self, X, idxss):
        X_train = []
        y_train = []

        for i, idxs in enumerate(idxss):
            for idx in idxs:
                X_train.append(X[idx])

            y_train.extend([i] * len(idxs))

        if self.classifier == 'sgd':
            clf = SGDClassifier(loss='log', penalty='l1', n_iter=2)
        else:
            clf = LinearSVC(penalty='l1', dual=False)

        clf.fit(stack(X_train), y_train)
        clf.sparsify()
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

        # Build leafs for all the nodes
        clf = self.train_clf(X, finished)
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
        if len(idxs) > 1000:
            print "Training classifier"

        clf = self.train_clf(X, [l_idx, r_idx])

        # Resplit the data
        if self.re_split:
            l_idx, r_idx = self.resplit(X, idxs, clf, 2)

        if not l_idx or not r_idx:
            return Leaf(self.compute_probs(y, idxs))

        lNode = self.grow_tree(X, y, l_idx, rs)
        rNode = self.grow_tree(X, y, r_idx, rs)

        return Node(lNode, rNode, clf)

    def predict(self, X):
        probs = []
        for tree in self.roots:
            while not isinstance(tree, Leaf):
                tree = tree.traverse(X)

            probs.append(tree.probs)
    
        def merge(d1, d2):
            d = d1.copy()
            for k, v in d2.iteritems():
                d[k] = d.get(k,0) + v

            return d
        
        res = reduce(merge, probs, {})
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

def sliding(it, window):
    x = list(islice(it, window))
    try:
        if len(x) == window:
            while True:
                yield x
                x2 = x[1:]
                x2.append(next(it))
                x = x2

    except StopIteration:
        pass

class Quantizer(object):
    def __init__(self):
        self.fh = FeatureHasher(dtype='float64')

    def quantize(self, text):
        text = text.lower().replace(',', '')
        unigrams = text.split()
        bigrams = (' '.join(xs) for xs in sliding(iter(unigrams), 2))
        trigrams = (' '.join(xs) for xs in sliding(iter(unigrams), 3))
        
        d = {f: 1.0 for f in chain(unigrams, bigrams, trigrams)}
        return self.fh.transform([d])

def quantize(fname, quantizer):
    with file(fname) as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print "%s docs encoded" % i

            data = json.loads(line)
            X = quantizer.quantize(data['title'])
            y = data['tags']
            yield X, y

def main(train, test):
    quantizer = Quantizer()
    X_train, y_train = [], []
    for X, y in quantize(train, quantizer):
        X_train.append(X)
        y_train.append(y)

    clf = FastXML(n_trees=1, re_split=True)
    clf.fit(X_train, y_train)
    sys.exit(0)

    for X, y in quantize(test, quantizer):
        y_hat = clf.predict(X)
        y_top = list(islice(y_hat, len(y)))
        print sorted(y)
        print sorted(y_top)
        print list(set(y) & set(y_top))
        print [(yi, y_hat.get(yi,0)) for yi in sorted(y)]
        print

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
