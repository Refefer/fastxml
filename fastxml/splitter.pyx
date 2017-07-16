#cython: boundscheck=False, wraparound=False, initializedcheck=False

from collections import defaultdict
import numpy as np
import scipy.sparse as sp

cimport cython
cimport numpy as np

from libc.math cimport log, abs, exp, pow
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.unordered_map cimport unordered_map
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

cdef inline void dcg(const vector[float]& ord_left, const vector[float]& ord_right, const vector[int]& ls, pair[float,float]& p):
    """
    We only need to use DCG since we're only using it to determine which partition
    bucket the label set in
    """
    cdef int i, l
    cdef float sl = 0, sr = 0
    for i in range(ls.size()):
        l = ls[i]
        sl += ord_left[l]
        sr += ord_right[l]

    p.first = sl
    p.second = sr

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

cdef class SparseNDCGSplitter(NDCGSplitter):
    cdef int n_labels
    cdef vector[I_PAIR] sorter
    cdef unordered_map[int,int] counter

    def __init__(self, const int n_labels):
        pass

    cdef void count_labels(self, const vector[int]& idxs, const YSET& yset):
        cdef int offset, yi, i, label
        cdef vector[int] ys

        self.counter.clear()
        for i in range(idxs.size()):
            offset = idxs[i]
            ys = yset[offset]
            for yi in range(ys.size()):
                label = ys[yi]
                inc(self.counter[label])

    cdef void sort_counter(self):

        # Copy it into a new vector
        cdef unordered_map[int,int].iterator b = self.counter.begin()
        cdef unordered_map[int,int].iterator e = self.counter.end()

        self.sorter.clear()

        while b != e:
            self.sorter.push_back(deref(b))
            inc(b)

        stdsort(self.sorter.begin(), self.sorter.end(), &sort_pairs)

    cdef void fill(self, vector[float]& k):
        cdef int size = k.size()
        for i in range(size):
            k[i] = 0.0

    cdef void order_labels(self, const vector[int]& idxs, const YSET& yset, 
            const vector[float]& weights, vector[float]& p_logs, vector[float]& logs):
        cdef int i, label
        cdef float w
        cdef pair[int,int] ord

        # Clean and copy
        self.count_labels(idxs, yset)

        # No access to std::fill, so write it yourself
        self.fill(logs)

        # Sort the results
        self.sort_counter()

        for i in range(self.sorter.size()):
            ord = self.sorter[i]
            label = ord.first
            w = weights[label]
            logs[label] = p_logs[i] * w

        return

cdef class Splitter:
    cdef vector[int] left, right
    cdef LR_SET newLeft, newRight

    cdef int n_labels, max_iters
    cdef float sparse_multiple

    cdef NDCGSplitter dense
    cdef NDCGSplitter sparse

    cdef vector[float] lOrder, rOrder, weights, logs
    cdef vector[vector[int]] yset

    def __init__(self, list y, 
            np.ndarray[np.float32_t] ws, 
            const float sparse_multiple,
            const int max_iters=50):

        # Initialize counters
        cdef pair[int,int] p
        p.first = p.second = 0

        cdef int n_labels = ws.shape[0]
        self.n_labels = n_labels

        # Variable for NDCG sorting
        self.sparse_multiple = sparse_multiple
        self.dense = DenseNDCGSplitter(n_labels)
        self.sparse = SparseNDCGSplitter(n_labels)

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

        prev = 0.0
        for i in range(size):
            self.weights.push_back(ws[i])
            self.logs.push_back(1 / (i + 2.0) )

    @property
    def max_label(self):
        return self.n_labels

    cdef bool use_sparse(self, const float ratio):
        """
        Sparse and Dense use different methods for computing ndcg scores:

        Dense writes a pair vector for label,count and then sorts that vector.  This
        can be very expensive if the total number of labels is high but the expected
        number of labels is low.  A big part of this cost comes from zeroing out the
        counts array every pass.

        Sparse uses a hashmap to keep the counts.  Its speed up comes from not
        having to preallocate the count vector or sort the entire vector set.

        """
        cdef int k = <int>(ratio  * self.n_labels)
        cdef float klogk = k * log(k) / log(2)
        return (klogk + self.n_labels) > (self.sparse_multiple * klogk)

    cdef void resevoir_split(self, list idxs, vector[int]& left, vector[int]& right, object rs):
        """
        We use sampling to guarantee both left and right sides have exactly half the
        items, with the P(left|X) == 0.5
        """
        cdef int i = 0
        cdef int idx
        cdef int size = len(idxs)
        cdef int half = size / 2

        if half < 2:
            for i in range(len(idxs)):
                left.push_back(idxs[i])
            return

        # Initialize counters
        for i in range(half):
            left.push_back(idxs[i])

        for i in range(half, size):
            idx = rs.randint(0, i)
            if idx < half:
                right.push_back(left[idx])
                left[idx] = idxs[i]
            else:
                right.push_back(idxs[i])

    def split_node(self, list idxs, rs):
        cdef vector[int] left, right
        cdef LR_SET newLeft, newRight
        cdef NDCGSplitter splitter
        cdef int i

        # Initialize counters
        self.resevoir_split(idxs, left, right, rs)

        cdef float ratio = (left.size() + right.size()) / <float>self.yset.size()
        if self.use_sparse(ratio):
            splitter = self.sparse
        else:
            splitter = self.dense

        for idx in range(self.max_iters):

            # Build ndcg for the sides
            splitter.order_labels(left, self.yset, self.weights, self.logs, self.lOrder)
            splitter.order_labels(right, self.yset, self.weights, self.logs, self.rOrder)

            # Divide out the sides
            newLeft = self.divide(left, True)
            newRight = self.divide(right, False)
            if newLeft.second.empty() and newRight.first.empty():
                # Done!
                break

            replace_vecs(left, newLeft.first, newRight.first)
            replace_vecs(right, newLeft.second, newRight.second)

        return left, right

    cdef LR_SET divide(self, const vector[int]& idxs, const bool is_left):
        cdef vector[int] newLeft, newRight
        cdef int i, idx
        cdef float lNdcg, rNdcg
        cdef LR_SET empty
        cdef vector[int] ys
        cdef pair[float,float] dcg_out

        for i in range(idxs.size()):
            idx = idxs[i]
            ys = self.yset[idx]
            dcg(self.lOrder, self.rOrder, ys, dcg_out)
            if dcg_out.first > dcg_out.second:
                newLeft.push_back(idx)
            elif dcg_out.first < dcg_out.second:
                newRight.push_back(idx)
            elif is_left:
                newLeft.push_back(idx)
            else:
                newRight.push_back(idx)

            lNdcg += dcg_out.first
            rNdcg += dcg_out.second

        empty.first = newLeft
        empty.second = newRight
        return empty
 
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
        if abs(n) > eps: 
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

cdef double radius2(const int [:] xi, const double [:] xd, 
                    const int [:] ui, const double [:] ud, 
                    const int s1, const int s2):
    """
    Computes the Sum((Xi - Ux) ** 2)
    """

    cdef int xidx = 0, uidx = 0
    cdef int xcol, ucol
    cdef double tally = 0.0, diff

    while xidx < s1 and uidx < s2:
        xcol = xi[xidx]
        ucol = ui[uidx]
        if xcol < ucol:
            tally += pow(xd[xidx], 2) 
            xidx += 1
        elif xcol > ucol:
            tally += pow(ud[uidx], 2)
            uidx += 1
        else:
            diff = (xd[xidx] - ud[uidx])
            tally += pow(diff, 2)
            xidx += 1
            uidx += 1

    # Get the remainder
    while xidx < s1 or uidx < s2:
        if xidx < s1:
            tally += xd[xidx] * xd[xidx]
            xidx += 1
        else:
            tally += ud[uidx] * ud[uidx]
            uidx += 1

    return tally

def radius(np.ndarray[np.double_t] Xid, np.ndarray[np.int32_t] Xii, 
           np.ndarray[np.double_t] uid, np.ndarray[np.int32_t] uii):

    return radius2(Xii, Xid, uii, uid, Xii.shape[0], uii.shape[0])

def sparse_mean_64(list xs, np.ndarray[np.double_t] ret):
    cdef int i, k
    cdef int [:] indices
    cdef double [:] data
    cdef double [:] r = ret
    for i in range(len(xs)):
        x = xs[i]
        indices = x.indices
        data = x.data
        for k in range(data.shape[0]):
            r[indices[k]] += data[k]

    cdef int size = len(xs)
    for i in range(r.shape[0]):
        r[i] /= size

