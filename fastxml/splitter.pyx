#cython: boundscheck=False, wraparound=False, cdivision=True, profile=True

from collections import defaultdict
import numpy as np
import scipy.sparse as sp

cimport cython
cimport numpy as np

from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef pair[vector[int],vector[int]] LR_SET
ctypedef pair[int,int] I_PAIR
ctypedef vector[I_PAIR] COUNTER
ctypedef vector[vector[int]] YSET
ctypedef vector[pair[int,float]] SR
ctypedef vector[SR] CSR

@cython.profile(False)
cdef bool sort_pairs(const I_PAIR& l, const I_PAIR& r):
    if l.second > r.second:
        return True

    if l.second < r.second:
        return False

    return l.first > r.first

cdef void copy_into(vector[int]& dest, vector[int]& src1):
    for i in range(src1.size()):
        dest.push_back(src1[i])

cdef void replace_vecs(vector[int]& dest, vector[int]& src1, vector[int]& src2):
    dest.swap(src1)
    copy_into(dest, src2)

cdef inline float dcg(const vector[float]& ord_left, const vector[float]& ord_right, const vector[int]& ls):
    """
    We only need to use DCG since we're only using it to determine which partition
    bucket the label set in
    """
    cdef int i, l
    cdef float log_left, log_right, sl = 0, sr = 0
    for i in range(ls.size()):
        l = ls[i]
        sl += ord_left[l]
        sr += ord_right[l]

    return sr - sl

cdef class NDCGSplitter:
    cdef void order_labels(self, const vector[int]& idxs, const YSET& yset, 
            const vector[float]& weights, vector[float]& p_logs, vector[float]& logs):
        return

cdef class DenseNDCGSplitter(NDCGSplitter):
    cdef int n_labels
    cdef vector[I_PAIR] counter

    def __init__(self, const int n_labels):

        cdef pair[int,int] p
        p.first = p.second = 0

        # Variable for NDCG sorting
        self.counter = vector[I_PAIR](n_labels, p)

    cdef void count_labels(self, const vector[int]& idxs, const YSET& yset):

        cdef int offset, yi, i, label
        cdef vector[int] ys

        # Clear the counter
        for i in range(self.counter.size()):
             self.counter[i].first = i
             self.counter[i].second = 0
            
        for i in range(idxs.size()):
            offset = idxs[i]
            ys = yset[offset]
            for yi in range(ys.size()):
                label = ys[yi]
                self.counter[label].second += 1

        return

    cdef void sort_counter(self):
        # Since this is potentially very sparse, we do a single pass moving non-empty
        # pairs to the front of counter
        cdef pair[int,int] tmp
        cdef size_t i = 0, j = self.counter.size() - 1
        while i < j:
            if self.counter[i].second > 0:
                i += 1
            elif self.counter[j].second == 0:
                j -= 1
            else:
                # swap
                tmp = self.counter[i]
                self.counter[i] = self.counter[j]
                self.counter[j] = tmp
                i += 1
                j -= 1

        # Partial sort only up to i
        stdsort(self.counter.begin(), 
                self.counter.begin() + i + 1, 
                &sort_pairs)

    cdef void order_labels(self, const vector[int]& idxs, const YSET& yset, 
            const vector[float]& weights, vector[float]& p_logs, vector[float]& logs):

        cdef int i, label
        cdef float w
        cdef pair[int,int] ord

        # Clean and copy
        self.count_labels(idxs, yset)

        # Sort the results
        self.sort_counter()

        for i in range(self.counter.size()):
            ord = self.counter[i]
            label = ord.first
            if ord.second == 0:
                break
            else:
                w = weights[label]
                logs[label] = p_logs[i] * w

        for l in range(i, self.counter.size()):
            logs[self.counter[l].first] = 0.0

        return

cdef class Splitter:
    cdef vector[int] left, right
    cdef LR_SET newLeft, newRight

    cdef int n_labels, max_iters

    cdef NDCGSplitter dense

    cdef vector[float] lOrder, rOrder, weights, logs
    cdef vector[vector[int]] yset

    def __init__(self, list y, np.ndarray[np.float32_t] ws, const int max_iters=50):

        # Initialize counters
        cdef pair[int,int] p
        p.first = p.second = 0

        cdef int n_labels = ws.shape[0]
        self.n_labels = n_labels

        # Variable for NDCG sorting
        self.dense = DenseNDCGSplitter(n_labels)

        # ndcg cache
        self.lOrder = vector[float](n_labels, 0.0)
        self.rOrder = vector[float](n_labels, 0.0)

        self.max_iters = max_iters

        self._init_ys(y, n_labels)
        self._init_weights(ws, n_labels)

    cdef void _init_ys(self, list y, const int n_labels):
        cdef list ys
        cdef int yi
        cdef vector[int] y_set

        for ys in y:
            y_set = vector[int]()
            for yi in ys:
                if yi > n_labels - 1:
                    raise Exception("Y label out of bounds")

                y_set.push_back(yi)

            self.yset.push_back(y_set)

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

    def split_node(self, list idxs, rs):
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
            self.dense.order_labels(left, self.yset, self.weights, self.logs, self.lOrder)
            self.dense.order_labels(right, self.yset, self.weights, self.logs, self.rOrder)

            # Divide out the sides
            newLeft = self.divide(left)
            newRight = self.divide(right)
            if newLeft.second.empty() and newRight.first.empty():
                # Done!
                break

            replace_vecs(left, newLeft.first, newRight.first)
            replace_vecs(right, newLeft.second, newRight.second)

        return left, right


    cdef LR_SET divide(self, const vector[int]& idxs):
        cdef vector[int] newLeft, newRight
        cdef int i, idx
        cdef float lNdcg, rNdcg, ddcg
        cdef LR_SET empty
        cdef vector[int] ys

        for i in range(idxs.size()):
            idx = idxs[i]
            ys = self.yset[idx]
            ddcg = dcg(self.lOrder, self.rOrder, ys)
            if ddcg <= 0:
                newLeft.push_back(idx)
            else:
                newRight.push_back(idx)

        empty.first = newLeft
        empty.second = newRight
        return empty
 
cdef class Node:
    cdef int idx
    cdef bool is_leaf

cdef class INode(Node):
    cdef Node left
    cdef Node right

    def __init__(self, int idx, Node left, Node right):
        self.is_leaf = False
        self.idx = idx
        self.left = left
        self.right = right

cdef class Leaf(Node):
    def __init__(self, const int idx):
        self.is_leaf = True
        self.idx = idx

cdef class PTree:
    cdef CSR weights
    cdef vector[float] bias
    cdef Node root
    cdef list payloads

    def __init__(self, object tree):
        self.payloads = []
        self.root = self.build_tree(tree)

    cdef SR convert_to_dense(self, const int [:] indices, const float [:] data, const int size):
        cdef SR sparse
        cdef pair[int,float] p
        cdef int i

        for i in range(size):
            p = pair[int,float]()
            p.first = indices[i]
            p.second = data[i]
            sparse.push_back(p)

        return sparse

    cdef Node build_tree(self, object tree):
        cdef int idx
        cdef float bias
        cdef Node left, right, node
        cdef np.ndarray[np.float32_t] data
        cdef np.ndarray[np.int32_t] indices

        if not tree.is_leaf:
            idx = self.weights.size()

            # Convert sparse weights to SR
            indices = tree.w.indices
            data = tree.w.data
            self.weights.push_back(self.convert_to_dense(indices, data, data.shape[0]))

            # Convert bias to float
            bias = tree.b[0]
            self.bias.push_back(bias)

            # Build subtree
            left  = self.build_tree(tree.left)
            right = self.build_tree(tree.right)
            
            return INode(idx, left, right)
        else:
            idx = len(self.payloads)
            self.payloads.append(tree.probs)
            return Leaf(idx)
    
    cdef float dot(self, const SR& x, const SR& w, const float bias):
        cdef int xidx = 0, widx = 0, xi, wi
        cdef int x_s = x.size(), w_s = w.size()
        cdef float tally = 0.0

        while xidx < x_s and widx < w_s:
            xi = x[xidx].first 
            wi = w[widx].first
            if xi < wi:
                xidx += 1
            elif xi > wi:
                widx += 1
            else:
                tally += x[xidx].second * w[widx].second
                xidx += 1
                widx += 1

        return tally + bias

    def predict(self, np.ndarray[np.float32_t] data, np.ndarray[np.int32_t] indices):
        cdef SR x = self.convert_to_dense(indices, data, data.shape[0])
        cdef int idx = self.predict_sr(x)
        return self.payloads[idx]

    cdef int predict_sr(self, SR& x):
        cdef SR w
        cdef float b, d
        cdef INode inode

        cdef Node node = self.root
        while not node.is_leaf:
            inode = <INode>node
            w = self.weights[node.idx]
            b = self.bias[node.idx]
            d = self.dot(x, w, b)
            if d < 0:
                node = inode.left
            else:
                node = inode.right

        return node.idx

def sparsify(np.ndarray[np.float64_t, ndim=2] dense, float eps=1e-6):
    """
    More work speeding up common operations that at large N add up to real time
    """
    cdef double [:, :] npv = dense
    cdef int i, count = 0
    cdef double n
    cdef vector[int] col
    cdef vector[float] data

    for i in range(npv.shape[1]):
        n = npv[0,i]
        if n > eps: 
            count += 1
            data.push_back(<float>n)
            col.push_back(i)

    cdef np.ndarray[np.int32_t] ip = np.zeros(2, dtype=np.int32)
    cdef np.ndarray[np.int32_t] c = np.zeros(count, dtype=np.int32)
    cdef np.ndarray[np.float32_t] d = np.zeros(count, dtype=np.float32)

    cdef int [:] cv = c
    cdef float [:] dv = d
    for i in range(count):
        cv[i] = col[i]
        dv[i] = data[i]

    cv = ip
    cv[1] = count

    csr = sp.csr_matrix((1, npv.shape[1]), dtype='float32')
    csr.indptr = ip
    csr.indices = c
    csr.data = d

    return csr
