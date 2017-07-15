#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
import scipy.sparse as sp

cimport cython
cimport numpy as np

from libc.math cimport log, abs, exp, pow
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef pair[int,float] DP
ctypedef vector[DP] SR
ctypedef vector[SR] CSR
ctypedef vector[vector[float]] DENSE

cdef object sr_to_sparse(const SR& sr, const int size):
    cdef int count = sr.size()

    cdef np.ndarray[np.int32_t] ip = np.zeros(2, dtype=np.int32)
    cdef np.ndarray[np.int32_t] c = np.zeros(count, dtype=np.int32)
    cdef np.ndarray[np.float32_t] d = np.zeros(count, dtype=np.float32)

    cdef int [:] cv = c
    cdef float [:] dv = d
    cdef pair[int,float] p 
    for i in range(count):
        p = sr[i]
        cv[i] = p.first
        dv[i] = p.second

    cv = ip
    cv[1] = count

    csr = sp.csr_matrix((1, size), dtype='float32')
    csr.indptr = ip
    csr.indices = c
    csr.data = d

    return csr

cdef float dot(const SR& x, const SR& w, const float bias):
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

cdef SR convert_to_sr(const int [:] indices, const float [:] data, const int size):
    cdef SR sparse
    cdef pair[int,float] p
    cdef int i

    for i in range(size):
        p = pair[int,float]()
        p.first = indices[i]
        p.second = data[i]
        sparse.push_back(p)

    return sparse

cdef SR sparse_sr_mean(const vector[SR*] probs, SR& averaged):
    cdef unordered_map[int,float] summer
    cdef SR* sr
    cdef int i,k

    # Copy srs into vector
    for i in range(probs.size()):
        sr = probs[i]
        for k in range(deref(sr).size()):
            summer[deref(sr)[k].first] += deref(sr)[k].second

    # Copy it into a new vector, averaging the values
    cdef unordered_map[int,float].iterator b = summer.begin()
    cdef unordered_map[int,float].iterator e = summer.end()

    cdef DP val
    while b != e:
        val = deref(b)
        val.second = val.second / probs.size()
        averaged.push_back(val)
        inc(b)

    stdsort(averaged.begin(), averaged.end())

    return averaged

cdef load_sparse(str fname, CSR& csr):
    cdef SR row
    cdef DP p
    with open(fname) as f:
        for line in f:
            row = vector[DP]()
            pieces = line.strip().split()
            for piece in pieces:
                index, weight = piece.split(':')
                p.first = int(index)
                p.second = float(weight)
                row.push_back(p)

            csr.push_back(row)

cdef load_dense_f32(str fname, DENSE& dense):
    cdef vector[float] row
    with open(fname) as f:
        for line in f:
            row = vector[float]()
            pieces = line.strip().split()
            for weight in pieces:
                row.push_back(float(weight))

            dense.push_back(row)

cdef load_dense_int(str fname, vector[vector[int]]& dense):
    cdef vector[int] row
    with open(fname) as f:
        for line in f:
            row = vector[int]()
            pieces = line.strip().split()
            for weight in pieces:
                row.push_back(int(weight))

            dense.push_back(row)


cdef class IForest:
    cdef list trees
    cdef int n_labels
    
    def __init__(self, str dname, int trees, int n_labels):
        self.n_labels = n_labels
        self.trees = []
        cdef int i
        for i in range(trees):
            self.trees.append(ITree(dname, i))
    
    cdef SR* _predict(self, const SR& sr, ITree t):
        return t.predict_payload(sr)
    
    def predict(self, np.ndarray[np.float32_t] data, np.ndarray[np.int32_t] indices):
        cdef SR sr = convert_to_sr(indices, data, data.shape[0])
        cdef vector[SR*] prob_set
        cdef SR payload
        cdef object sparse
        cdef SR* res
        for t in self.trees:
            res = self._predict(sr, t)
            prob_set.push_back(res)

        sparse_sr_mean(prob_set, payload)
        sparse = sr_to_sparse(payload, self.n_labels)

        return sparse

cdef class ITree:
    cdef int rootIdx
    cdef vector[vector[int]] tree
    cdef CSR payloads

    cdef CSR W

    cdef vector[float] bias

    def __init__(self, str dname, int tree_idx) :

        p = dname.rstrip('/') + '/tree.%s' % tree_idx

        # Load Sparse into W points
        load_sparse(p + '.weights', self.W)

        # Load bias
        cdef DENSE tmp
        load_dense_f32(p + '.bias', tmp)
        self.bias.swap(tmp[0])

        # Load Tree
        load_dense_int(p + '.tree', self.tree)

        # Load Payloads
        load_sparse(p + '.probs', self.payloads)

        self.rootIdx = self.tree.size() - 1

    cdef SR* predict_payload(self, const SR& sr):
        cdef int idx = self.predict_sr(sr)
        return &self.payloads[idx]

    cdef inline int index(self, const vector[int]& node):
        return node[0]

    cdef inline int left(self, const vector[int]& node):
        return node[1]

    cdef inline int right(self, const vector[int]& node):
        return node[2]

    cdef inline bool is_leaf(self, const vector[int]& node):
        return node[3] == 1

    cdef int predict_sr(self, const SR& data):
        cdef unsigned int index, nIndex
        cdef vector[int] node
        cdef float d
        cdef SR* W

        node = self.tree[self.rootIdx]
        while not self.is_leaf(node):
            index = self.index(node)
            W = &self.W[index]
            d = dot(data, deref(W), self.bias[index])
            if d < 0:
                nIndex = self.left(node)
            else:
                nIndex = self.right(node)

            node = self.tree[nIndex]

        return self.index(node)

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

def compute_leafs(float gamma, np.ndarray[np.double_t] Xid, np.ndarray[np.int32_t] Xii, 
        np.ndarray[np.int32_t] indices, object sparse, np.ndarray[np.float32_t] radius):
    cdef int i, start, end, index
    cdef float r, ur, rad
    cdef object m 
    cdef vector[float] ret

    cdef int [:] m_indptr = sparse.indptr
    cdef int [:] m_indices = sparse.indices, mi_indices
    cdef double [:] m_data = sparse.data, mi_data

    for i in range(indices.shape[0]):
        index = indices[i]
        ur = radius[index]

        start = m_indptr[index]
        end   = m_indptr[index+1]

        mi_indices = m_indices[start:end]
        mi_data    = m_data[start:end]

        rad = radius2(Xii, Xid, mi_indices, mi_data, Xii.shape[0], mi_indices.shape[0])
        k = exp(gamma  * (rad - ur)) 
        ret.push_back(1. / (1. + k))

    return ret

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

def sparse_mean_32(list xs, np.ndarray[np.float32_t] ret):
    cdef int i, k
    cdef int [:] indices
    cdef float [:] data
    cdef float [:] r = ret
    for i in range(len(xs)):
        x = xs[i]
        indices = x.indices
        data = x.data
        for k in range(data.shape[0]):
            r[indices[k]] += data[k]

    cdef int size = len(xs)
    for i in range(r.shape[0]):
        r[i] /= size

