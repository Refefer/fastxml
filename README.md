FastXML - Fast and Accurate Tree Extreme Multi-label Classifier
===

An implementation of "FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning" (https://manikvarma.github.io/pubs/prabhu14.pdf).  It's implemented in the quasi-familiar scikit-learn clf format.

Usage
===

    from fastxml import FastXML

    X = [Sparse or numpy arrays]
    y = [["red", "pants"]]

    clf = FastXML(n_trees=32, n_jobs=-1)

    clf.fit(X, y)

    clf.predict(X)

TODO
===

Quite a bit todo!  A base implementation using NDCG is fully working with Cython bindings to make the alternating minization reasonably fast - but a lot more could be done here:

1. Abstract out Rank Metric.  This is needed to get Propensity-based ranking metrics integrated in a consistent manner.  See https://manikvarma.github.io/pubs/jain16.pdf for motivation.
2. Leaf re-ranking based on the above paper.  That should dramatically improve performance of rare labels.
3. Abstract out classifier from learning algorithm.
4. General speedups:
    a. Sparse vstack for learning is incredibly expensive currently, taking ~50% of the total learning time!
    b. Speed up split_node.  Right now it's using dicts all over the place since labels aren't required to to be integers.  We can look to moving it to sparse arrays.
    c. Better parallelism.  It currently does parallelism across trees rather than intra-tree.
    d. Better memory management.  We're creating a lot of garbage using intermediate lists for splitting data

