#cython: boundscheck=False, wraparound=False, cdivision=True

from collections import defaultdict
import numpy as np

cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef int MAX_LABELS = 15000000
cdef np.float32_t [:] LOGS

ctypedef pair[vector[int],vector[int]] LR_SET

def init_logs():
    global LOGS
    LOGS = 1 / np.arange(2, MAX_LABELS + 2, dtype=np.float32)

init_logs()

cdef float dcg(dict order, list ls):
    """
    We only need to use DCG since we're only using it to determine which partition
    bucket the label set in
    """
    cdef float log, s = 0
    for l in ls:
        log = order.get(l, 0.0)
        s += log

    return s

cdef object count_labels(list y, vector[int]& idxs):
    cdef long size = idxs.size()
    cdef int offset
    d = defaultdict(int)
    while size:
        size -= 1
        offset = idxs[size]
        for yi in y[offset]:
            d[yi] += 1

    return d

cdef dict order_labels(list y, float [:] weights, vector[int]& idxs):
    cdef int i, l
    counter = count_labels(y, idxs)
    sLabels = sorted(counter.keys(), key=counter.__getitem__, reverse=True)
    cdef dict d = {}
    for i in range(len(sLabels)):
        l = sLabels[i]
        w = weights[l]
        d[l] = LOGS[i] * w

    return d

cdef LR_SET divide(list y, vector[int]& idxs, dict lOrder, dict rOrder):
    cdef vector[int] newLeft, newRight
    cdef int i, idx
    cdef float lNdcg, rNdcg
    cdef LR_SET empty

    for i in range(idxs.size()):
        idx = idxs[i]
        ys = y[idx]
        lNdcg = dcg(lOrder, ys)
        rNdcg = dcg(rOrder, ys)
        if lNdcg >= rNdcg:
            newLeft.push_back(idx)
        else:
            newRight.push_back(idx)

    empty.first = newLeft
    empty.second = newRight
    return empty

cdef copy_into(vector[int]& dest, vector[int]& src1):
    for i in range(src1.size()):
        dest.push_back(src1[i])

cdef replace_vecs(vector[int]& dest, vector[int]& src1, vector[int]& src2):
    dest.swap(src1)
    copy_into(dest, src2)

def split_node(list y, np.ndarray[np.float32_t] weights, list idxs, rs, int max_iters = 50):
    cdef vector[int] left, right
    cdef LR_SET newLeft, newRight
    cdef int iters

    for i in idxs:
        if rs.rand() < 0.5:
            left.push_back(i)
        else:
            right.push_back(i)

    iters = 0
    while True:
        if iters > max_iters:
            break

        iters += 1

        # Build ndcg for the sides
        lOrder = order_labels(y, weights, left)
        rOrder = order_labels(y, weights, right)

        # Divide out the sides
        newLeft = divide(y, left, lOrder, rOrder)
        newRight = divide(y, right, lOrder, rOrder)
        if newLeft.second.empty() and newRight.first.empty():
            # Done!
            break

        replace_vecs(left, newLeft.first, newRight.first)
        replace_vecs(right, newLeft.second, newRight.second)

    return left, right
