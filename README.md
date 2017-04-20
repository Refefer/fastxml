FastXML / PFastXML / PFastreXML - Fast and Accurate Tree Extreme Multi-label Classifier
===

This is a fast implementation of FastXML, PFastXML, and PFastreXML based on the following papers:

 - "FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning" [Paper](https://manikvarma.github.io/pubs/prabhu14.pdf)
 - "Extreme Multi-label Loss Functions for Recommendation, Tagging, Ranking & Other Missing Label Application" [Paper](https://manikvarma.github.io/pubs/jain16.pdf)
 - "DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification" [Paper](https://arxiv.org/abs/1609.02521) [Code](https://sites.google.com/site/rohitbabbar/code/dismec)

DiSMEC makes it's appearance via an L2 penalty rather than an L1 which, when set with a high alpha and sparsity eps of 0.01-0.05, also can produce sparse linear classifiers.

It's implemented in the quasi-familiar scikit-learn clf format.

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

    from fastxml import FastXML

    X = [Sparse or numpy arrays]
    y = [[1, 3]] # Currently requires list[list[int]]

    clf = FastXML(n_trees=32, n_jobs=-1)

    clf.fit(X, y)

    clf.predict(X)
    # or
    clf.predict(X, fmt='dict')

    #############
    # PFastXML
    #############

    from fastxml.weights import propensity

    weights = propensity(y)
    clf.fit(X, y, weights)
    
    ###############
    # PFastreXML
    ###############
    clf = FastXML(n_trees=32, n_jobs=-1, leaf_classifiers=True)
    clf.fit(X, y, weights)

TODO
===

1. I'm currently estimating the leaf classifiers with the label mean and setting margin to the maximum radius of a sample in that set (ie. Hard Margin SVDD).  This works reasonably well for tail labels with few examples (~5-10 examples), however for head classes the hard margin starts to include _everything_, a problem for boosting tail labels.

 - Is there a better way to learn this?

2. Run all the standard benchmark datasets against it.

3. Refactor.  Most of the effort has been spent on speed and it needs to be cleaned up.
