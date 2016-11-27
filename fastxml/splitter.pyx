#cython: boundscheck=False, wraparound=False, cdivision=True

from collections import defaultdict
import numpy as np

cimport cython
cimport numpy as np
from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector
from libcpp.pair cimport pair


ctypedef pair[vector[int],vector[int]] LR_SET
ctypedef pair[int,int] I_PAIR
ctypedef vector[I_PAIR] COUNTER

cdef class Splitter:
    cdef vector[int] left, right
    cdef LR_SET newLeft, newRight

    cdef int n_labels, max_iters

    cdef COUNTER counter 
    cdef vector[float] lOrder, rOrder, weights, logs

    def __init__(self, np.ndarray[np.float32_t] ws, const int max_iters=50):

        # Initialize counters
        cdef pair[int,int] p
        p.first = p.second = 0

        cdef int n_labels = ws.shape[0]
        self.n_labels = n_labels

        # Variable for NDCG sorting
        self.counter = vector[pair[int,int]](n_labels, p)

        # ndcg cache
        self.lOrder = vector[float](n_labels, 0.0)
        self.rOrder = vector[float](n_labels, 0.0)

        self.max_iters = max_iters

        self._init_weights(ws, n_labels)

    cdef void _init_weights(self, const float [:] ws, const int size):
        cdef int i

        self.weights.reserve(size)
        self.logs.reserve(size)

        for i in range(size):
            self.weights.push_back(ws[i])
            self.logs.push_back(1 / (i + 2.0))

    @property
    def max_label(self):
        return self.n_labels

    def split_node(self, list y, list idxs, rs):
        cdef vector[int] left, right
        cdef LR_SET newLeft, newRight
        cdef int i

        # Initialize counters
        for i in idxs:
            if rs.rand() < 0.5:
                left.push_back(i)
            else:
                right.push_back(i)

        for idx in range(self.max_iters):

            # Build ndcg for the sides
            self.order_labels(y, left, self.lOrder)
            self.order_labels(y, right, self.rOrder)

            # Divide out the sides
            newLeft = self.divide(y, left)
            newRight = self.divide(y, right)
            if newLeft.second.empty() and newRight.first.empty():
                # Done!
                break

            replace_vecs(left, newLeft.first, newRight.first)
            replace_vecs(right, newLeft.second, newRight.second)

        return left, right

    cdef void count_labels(self, list y, const vector[int]& idxs):
        cdef int offset, yi, i
        cdef I_PAIR p

        for i in range(self.counter.size()):
             self.counter[i].first = i
             self.counter[i].second = 0
            
        for i in range(idxs.size()):
            offset = idxs[i]
            for yi in y[offset]:
                self.counter[yi].second += 1

        return

    cdef void order_labels(self, list y, const vector[int]& idxs, vector[float]& logs):
        cdef vector[float] rankings
        cdef int i, label
        cdef float w
        cdef pair[int,int] ord

        # Clean and copy
        self.count_labels(y, idxs)

        # Sort the results
        stdsort(self.counter.begin(), self.counter.end(), &sort_pairs)

        for i in range(self.counter.size()):
            ord = self.counter[i]
            label = ord.first
            if ord.second == 0:
                logs[label] = 0.0
            else:
                w = self.weights[label]
                logs[label] = self.logs[i] * w

        return

    cdef LR_SET divide(self, list y, const vector[int]& idxs):
        cdef vector[int] newLeft, newRight
        cdef int i, idx
        cdef float lNdcg, rNdcg, ddcg
        cdef LR_SET empty
        cdef list ys

        for i in range(idxs.size()):
            idx = idxs[i]
            ys = y[idx]
            ddcg = dcg(self.lOrder, self.rOrder, ys)
            if ddcg <= 0:
                newLeft.push_back(idx)
            else:
                newRight.push_back(idx)

        empty.first = newLeft
        empty.second = newRight
        return empty
 
@cython.profile(False)
cdef bool sort_pairs(const I_PAIR& l, const I_PAIR& r):
    return l.second > r.second

cdef void copy_into(vector[int]& dest, vector[int]& src1):
    for i in range(src1.size()):
        dest.push_back(src1[i])

cdef void replace_vecs(vector[int]& dest, vector[int]& src1, vector[int]& src2):
    dest.swap(src1)
    copy_into(dest, src2)

cdef inline float dcg(const vector[float]& ord_left, const vector[float]& ord_right, list ls):
    """
    We only need to use DCG since we're only using it to determine which partition
    bucket the label set in
    """
    cdef int l
    cdef float log_left, log_right, sl = 0, sr = 0
    for l in ls:
        sl += ord_left[l]
        sr += ord_right[l]

    return sr - sl


