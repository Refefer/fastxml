#!/usr/bin/env python
import sys
import json
import os
from collections import defaultdict
import multiprocessing
from itertools import islice, chain, count
import cPickle

import argparse

from sklearn.feature_extraction import FeatureHasher
import scipy.sparse as sp

from fastxml import FastXML
from fastxml.weights import uniform, nnllog, propensity

def build_arg_parser():
    parser = argparse.ArgumentParser(description='FastXML trainer and tester',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", 
        help="Model to use for dataset file")

    parser.add_argument("input_file", 
        help="Input file to use")
    
    subparsers = parser.add_subparsers(dest="command")

    trainer = subparsers.add_parser('train', help="Set up for trainer")
    build_train_parser(trainer)

    inference = subparsers.add_parser('inference', help="Runs a model against a dataset")
    build_inference_parser(inference)

    return parser

def build_inference_parser(parser):
    parser.add_argument("--max-predict", dest="max_predict", type=int,
        default=10,
        help="Maximum number of classes to predict"
    )
    parser.add_argument("--dict", dest="dict", action="store_true",
        help="Store predict as dict"
    )
    parser.add_argument("--gamma", type=float,
        help="Overrides default gamma value for leaf classifiers"
    )
    parser.add_argument("--blend_factor", type=float,
        help="Overrides default blend factor"
    )

def build_train_parser(parser):
    parser.add_argument("--trees", dest="trees", type=int,
        default=50,
        help="Number of trees to use"
    )
    parser.add_argument("--max_leaf_size", dest="max_leaf_size", type=int,
        default=10,
        help="Maximumum number of examples allowed per leaf"
    )
    parser.add_argument("--max_labels_per_leaf", dest="max_labels_per_leaf", type=int,
        default=50,
        help="Maximum number of classes to retaion for probability distribution per leaf"
    )
    parser.add_argument("--re_split", dest="re_split", type=int,
        default=1,
        help="After fitting a classifier, re-splits the data according to fitted "\
             "classifier.  If greater than 1, it will re-fit and re-train a classifier "\
             "the data if after splitting, it all ends in a leaf.  Will retry N times."
    )
    parser.add_argument("--alpha", dest="alpha", type=float,
        default=1e-3,
        help="L1 coefficient.  Too high and it won't learn a split, too low and "\
             "it won't be sparse (larger file size, slower inference)."
    )
    parser.add_argument("--iters", dest="iters", type=int,
        default=2,
        help="Number of iterations to run over the dataset when fitting classifier"
    )
    parser.add_argument("--no_bias", dest="bias", action="store_false",
        help="Fits a bias for the classifier.  Not needed if data has E[X] = 0"
    )
    parser.add_argument("--subsample", dest="subsample", type=float,
        default=1.0,
        help="Subsample data per tree.  if less than 1, interpretted as a "\
             "percentage.  If greater than one, taken as number of data " \
             "points per tree."
    )
    parser.add_argument("--loss", dest="loss", choices=('log', 'hinge'),
        default='log',
        help="Loss to minimize."
    )
    parser.add_argument("--threads", dest="threads", type=int,
        default=multiprocessing.cpu_count(),
        help="Number of threads to use.  Will use min(threads, trees)"
    )
    parser.add_argument("--label-weight", dest="label_weight", 
        choices=('uniform', 'nnllog', 'propensity'), default='propensity',
        help="Number of threads to use.  Will use min(threads, trees)"
    )
    parser.add_argument("--leaf-classifiers", dest="leaf_class", 
        action="store_true",
        help="Whether to use and compute leaf classifiers"
    )
    parser.add_argument("--gamma", type=int, default=30,
        help="Gamma coefficient for hyper-sphere weighting"
    )
    parser.add_argument("--blend-factor", dest="blend_factor",
        type=float, default=0.5,
        help="blend * tree-probs + (1 - blend) * tail-classifiers"
    )
    parser.add_argument("--verbose", action="store_true",
        help="Verbose"
    )
    return parser

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
        self.fh = FeatureHasher(dtype='float32')

    def quantize(self, text):
        text = text.lower().replace(',', '')
        unigrams = text.split()
        bigrams = (' '.join(xs) for xs in sliding(iter(unigrams), 2))
        trigrams = (' '.join(xs) for xs in sliding(iter(unigrams), 3))
        
        d = {f: 1.0 for f in chain(unigrams, bigrams, trigrams)}
        return self.fh.transform([d])

def quantize(fname, quantizer, labels_only=False, verbose=True):
    with file(fname) as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and verbose:
                print "%s docs encoded" % i

            data = json.loads(line)
            if not data['tags'] or not data['title']:
                continue

            X = quantizer.quantize(data['title']) if not labels_only else None
            y = list(set(data['tags']))
            yield data, X, y

class Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def model(self):
        return os.path.join(self.dataset, 'model')

    @property
    def classes(self):
        return os.path.join(self.dataset, 'counts')

def train(args):
    # Quantize
    quantizer = Quantizer()
    cnt = count()
    classes = defaultdict(int)
    X_train, y_train = [], []
    for _, X, ys in quantize(args.input_file, quantizer):
        nys = []
        for y in ys:
            if y not in classes:
                classes[y] = next(cnt)

            nys.append(classes[y])
        
        X_train.append(X)
        y_train.append(nys)

    # Save the mapping
    dataset = Dataset(args.model)
    if not os.path.isdir(args.model):
        os.makedirs(args.model)

    with file(dataset.classes, 'w') as out:
        json.dump(classes.items(), out)

    # Train
    clf = FastXML(
        n_trees=args.trees,
        max_leaf_size=args.max_leaf_size,
        max_labels_per_leaf=args.max_labels_per_leaf,
        re_split=args.re_split,
        alpha=args.alpha,
        n_epochs=args.iters,
        bias=args.bias,
        subsample=args.subsample,
        loss=args.loss,
        leaf_classifiers=args.leaf_class,
        blend=args.blend_factor,
        gamma=args.gamma,
        n_jobs=args.threads,
        verbose=args.verbose
    )

    if args.label_weight == 'nnllog':
        weights = nnllog(y_train)
    elif args.label_weight == 'uniform':
        weights = uniform(y_train)
    elif args.label_weight == 'propensity':
        weights = propensity(y_train)
    else:
        raise NotImplementedError(args.label_weight)

    clf.fit(X_train, y_train, weights=weights)

    with file(dataset.model, 'w') as out:
        cPickle.dump(clf, out, cPickle.HIGHEST_PROTOCOL)

    sys.exit(0)

def inference(args):
    dataset = Dataset(args.model)

    quantizer = Quantizer()
    with file(dataset.model) as f:
        clf = cPickle.load(f)

    if args.blend_factor is not None:
        clf.blend = args.blend_factor

    if args.gamma is not None:
        clf.gamma = args.gamma

    # Load reverse map
    with file(dataset.classes) as f:
        data = json.load(f)
        classes = {v:k for k, v in data}

    for data, X, y in quantize(args.input_file, quantizer, verbose=False):
        y_hat = clf.predict(X, 'dict')[0]
        yi = islice(y_hat.iteritems(), args.max_predict)
        nvals = [[unicode(classes[k]), v] for k, v in yi]
        data['predict'] = dict(nvals) if args.dict else nvals
        print json.dumps(data)

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    if args.command == 'train':
        train(args)
    else:
        inference(args)
