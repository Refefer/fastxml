import json
import sys
import math
from itertools import islice, chain, repeat
from collections import Counter, OrderedDict

import numpy as np

import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher

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

class Node(object):
    def __init__(self, left, right, clf):
        self.left = left
        self.right = right
        self.clf = clf

class Leaf(object):
    def __init__(self, probs):
        self.probs = probs

def ndcg(order, ls):
    score = 0
    idcg = 0
    for i, l in enumerate(ls):
        if l in order:
            score += 1 / math.log(order[l] + 2)
        idcg += 1 / math.log(2+i)

    return score / idcg

def count_labels(y, idxs):
    it = (yi for i in idxs for yi in y[i])
    return Counter(it)

def order_labels(y, idxs):
    counter = count_labels(y, idxs)
    sLabels = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
    return {l: i for i, l in enumerate(sLabels)}

def divide(y, idxs, lOrder, rOrder):
    newLeft, newRight = [], []
    for i in idxs:
        lNdcg = ndcg(lOrder, y[i])
        rNdcg = ndcg(rOrder, y[i])
        if lNdcg >= rNdcg:
            newLeft.append(i)
        else:
            newRight.append(i)

    return newLeft, newRight

def split_node(X, y, idxs, rs):
    print "Splitting: {}", len(idxs)

    left, right = [], []
    for i in idxs:
        (left if rs.rand() < 0.5 else right).append(i)

    while True:
        # Build ndcg for the sides
        lOrder = order_labels(y, left)
        rOrder = order_labels(y, right)

        # Divide out the sides
        left1, right1 = divide(y, left, lOrder, rOrder)
        left2, right2 = divide(y, right, lOrder, rOrder)
        if not right1 and not left2:
            # Done!
            break

        left = left1 + left2
        right = right1 + right2

    return left, right

def compute_probs(y, idxs):
    counter = count_labels(y, idxs)
    total = float(sum(counter.itervalues()))
    return OrderedDict([(k, v / total) for k, v in counter.iteritems()])

def stack(X):
    stack = np.vstack if isinstance(X[0], np.ndarray) else sp.vstack
    return stack(X)

def train_clf(X, l_idx, r_idx):
    # Construct dataset
    X_train = []
    y_train = []
    for idxs, yv in ((l_idx, 0), (r_idx, 1)):
        for idx in idxs:
            X_train.append(X[idx])

        y_train.extend([yv] * len(idxs))

    clf = SGDClassifier(loss='log', penalty='l1')
    clf.fit(stack(X_train), y_train)

    return clf

def resplit(X, idxs, clf):
    X_train = [X[i] for i in idxs]
    l_idx, r_idx = [], []
    for i, k in enumerate(clf.predict(stack(X_train))):
        if k:
            r_idx.append(idxs[i])
        else:
            l_idx.append(idxs[i])

    return l_idx, r_idx

def grow_tree(X, y, idxs, max_depth, rs):
    if max_depth == 0:
        return Leaf(compute_probs(y, idxs))
    
    l_idx, r_idx = split_node(X, y, idxs, rs)

    if not l_idx or not r_idx:
        return Leaf(compute_probs(y, idxs))

    # Train the classifier
    clf = train_clf(X, l_idx, r_idx)

    # Resplit the data
    l_idx, r_idx = resplit(X, idxs, clf)

    if not l_idx or not r_idx:
        return Leaf(compute_probs(y, idxs))

    lNode = grow_tree(X, y, l_idx, max_depth - 1, rs)
    rNode = grow_tree(X, y, r_idx, max_depth - 1, rs)

    return Node(lNode, rNode, clf)

def predict(tree, X):
    while not isinstance(tree, Leaf):
        if tree.clf.predict(X) == 0:
            tree = tree.left
        else:
            tree = tree.right

    return tree.probs

class Quantizer(object):
    def __init__(self):
        self.fh = FeatureHasher(dtype='float32')

    def quantize(self, text):
        text = text.lower().replace(',', '')
        unigrams = text.split()
        bigrams = (' '.join(xs) for xs in sliding(iter(unigrams), 2))
        trigrams = (' '.join(xs) for xs in sliding(iter(unigrams), 3))
        
        d = {f: 1.0 for f in chain(unigrams, bigrams, trigrams)}
        return self.fh.transform([d])

def main(train, test):
    quantizer = Quantizer()
    X_train, y_train = [], []
    with file(train) as f:
        for line in f:
            data = json.loads(line)
            X_train.append(quantizer.quantize(data['title']))
            y_train.append(data['tags'])

    rs = np.random.RandomState(seed=2016)
    tree = grow_tree(X_train, y_train, range(len(X_train)), 10, rs)

    with file(test) as f:
        for line in f:
            data = json.loads(line)
            X = quantizer.quantize(data['title'])
            y = data['tags']
            y_hat = predict(tree, X)
            y_top = list(islice(y_hat, len(y)))
            print sorted(y)
            print sorted(y_top)
            print list(set(y) & set(y_top))
            print [(yi, y_hat.get(yi,0)) for yi in sorted(y)]
            print

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
