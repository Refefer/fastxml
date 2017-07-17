import math

def precision(scores, k):
    return sum(scores[:k]) / float(k)

def dcg(scores, k=None):
    if k is not None:
        scores = scores[:k]

    return sum(rl / math.log(i + 2) for i, rl in enumerate(scores))

def ndcg(scores, k=None, eps=1e-6):
    idcgs = dcg(sorted(scores, reverse=True), k)
    if idcgs < eps:
        return 0.0

    dcgs = dcg(scores, k)

    return dcgs / idcgs

def pSdcg(scores, props, k=None):
    if k is not None:
        scores = scores[:k]

    k = 0
    for i, rl in enumerate(scores):
        p = props[i] if i < len(props) else 1
        k += rl / (p * math.log(i + 2))
    
    return k

def pSndcg(scores, props, k=None):
    dcgs = pSdcg(scores, props, k)

    denom = sum(1. / math.log(i + 2) for i in xrange(k or len(scores)))

    return dcgs / denom


