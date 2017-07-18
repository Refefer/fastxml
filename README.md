FastXML / PFastXML / PFastreXML - Fast and Accurate Tree Extreme Multi-label Classifier
===

This is a fast implementation of FastXML, PFastXML, and PFastreXML based on the following papers:

 - "FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning" [Paper](https://manikvarma.github.io/pubs/prabhu14.pdf)
 - "Extreme Multi-label Loss Functions for Recommendation, Tagging, Ranking & Other Missing Label Application" [Paper](https://manikvarma.github.io/pubs/jain16.pdf)
 - "DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification" [Paper](https://arxiv.org/abs/1609.02521) [Code](https://sites.google.com/site/rohitbabbar/code/dismec)

DiSMEC makes it's appearance via an L2 penalty rather than an L1 which, when set with a high alpha and sparsity eps of 0.01-0.05, also can produce sparse linear classifiers.

It's implemented in the quasi-familiar scikit-learn clf format.

Release Notes
===
2.0
---
 - Version 2.0 is _not_ backward compatible with 1.x
 - User model.save(path) to save models instead of cPickle
 - Rewrites data storage layer
 - Uses 50% the memory, loads 30% faster, and is 40% faster to inference

Binary
===

This repo provides a simple script along with the library, fxml.py, which allows easy train / testing of simple datasets.

It takes two formats: a simple JSON format and the standard extreme multi label dataset format.

Standard Benchmark Datasets
---

As an example, to train a standalone classifier against the Delicious-200K dataset:

    fxml.py delicious.model deliciousLarge_train.txt --standard-dataset --verbose train --iters 5 --trees 20 --label-weight propensity --alpha 1e-4 --leaf-classifiers --no-remap-labels

To test:

    fxml.py delicious.model deliciousLarge_test.txt --standard-dataset inference

JSON File
---

As fxml.py is intended as an easy to understand example for setting up a FastXML classifier, the JSON format
is very simple.  It is newline delimited format.

train.json:
    
    {"title": "red dresses", "tags": ["clothing", "women", "dresses"]}
    {"title": "yellow dresses for sweet 16", "tags": ["yellow", "summer dresses", "occasionwear"]}
    ...

It can then be trained:
    
    fxml.py my_json.model train.json --verbose train --iters 5 --trees 20 --label-weight propensity --alpha 1e-4 --leaf-classifiers

Not the omission of the flags "--standard-dataset" and "--no-remap-labels".  Since the tags/classes provided are strings, fxml.py will remap them to an integer label space for training.  During inference, it will map the label index back

Simple Python Usage
===

    from fastxml import Trainer, Inferencer

    X = [Sparse or numpy arrays]
    y = [[1, 3]] # Currently requires list[list[int]]

    trainer = Trainer(n_trees=32, n_jobs=-1)

    trainer.fit(X, y)

    trainer.save(path)

    clf = Inferencer(path)

    clf.predict(X)
    # or
    clf.predict(X, fmt='dict')

    #############
    # PFastXML
    #############

    from fastxml.weights import propensity

    weights = propensity(y)
    trainer.fit(X, y, weights)
    
    ###############
    # PFastreXML
    ###############
    trainer = Trainer(n_trees=32, n_jobs=-1, leaf_classifiers=True)
    trainer.fit(X, y, weights)

TODO
===

1. Run all the standard benchmark datasets against it.

2. Refactor.  Most of the effort has been spent on speed and it needs to be cleaned up.
