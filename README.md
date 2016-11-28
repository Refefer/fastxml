(P)FastXML - Fast and Accurate Tree Extreme Multi-label Classifier
===

An implementation of "FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning" (https://manikvarma.github.io/pubs/prabhu14.pdf).  It currently implements PfastXML as well (https://manikvarma.github.io/pubs/jain16.pdf) and PfasterXML is in the works.

It's implemented in the quasi-familiar scikit-learn clf format.

Usage
===

    from fastxml import FastXML

    X = [Sparse or numpy arrays]
    y = [[1, 3]] # Currently requires list[list[int]]

    clf = FastXML(n_trees=32, n_jobs=-1)

    clf.fit(X, y)

    clf.predict(X)
    # or
    clf.predict(X, fmt='dict')

TODO
===

Quite a bit todo!  A base implementation using NDCG is fully working with Cython bindings to make the alternating minization reasonably fast - but a lot more could be done here:

1. Leaf re-ranking based on the above paper.  That should dramatically improve performance of rare labels.
2. Propensity calculator.  Right now I'm using the suggested hyperparameters from the paper which isn't ideal depending on the dataset.
<s>Abstract out Rank Metric.  This is needed to get Propensity-based ranking metrics integrated in a consistent manner.</s> This is done via the weights vector passed into Splitter
