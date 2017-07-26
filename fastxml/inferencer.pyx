#cython: boundscheck=False, wraparound=False

import numpy as np
import scipy.sparse as sp
import struct

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

cdef object read_row(object f, str type):
    d = f.read(struct.calcsize('I'))
    if not d: 
        return None

    # Get size of row, and unpack entire pair set
    num, = struct.unpack("I", d)
    d2 = f.read(num * struct.calcsize(type))

    return struct.unpack(type * num, d2)

cdef void load_sparse(str fname, CSR& csr):
    cdef SR row
    cdef DP p
    cdef int i
    with open(fname, 'rb') as f:
        while True:
            values = read_row(f, 'If')
            if values is None:
                break

            row = vector[DP]()
            for i in range(0, len(values), 2):
                p.first = values[i]
                p.second = values[i+1]
                row.push_back(p)

            csr.push_back(row)

cdef load_dense_f32(str fname, DENSE& dense):
    cdef vector[float] row
    cdef int i
    with open(fname, 'rb') as f:
        while True:
            values = read_row(f, 'f')
            if values is None:
                break

            row = vector[float]()

            # Get size of row, and unpack floats
            for i in range(0, len(values)):
                row.push_back(values[i])

            dense.push_back(row)

cdef load_dense_int(str fname, vector[vector[int]]& dense):
    cdef vector[int] row
    cdef int i
    with open(fname, 'rb') as f:

        while True:
            values = read_row(f, 'I')
            if values is None:
                break

            row = vector[int]()

            for i in range(0, len(values)):
                row.push_back(values[i])

            dense.push_back(row)

cdef class Blender:
    cdef IForest forest
    cdef LeafComputer lc 

    def __init__(self, IForest forest, LeafComputer lc):
        self.forest = forest
        self.lc = lc

    def predict(self, np.ndarray[np.float32_t] data, 
                      np.ndarray[np.int32_t] indices, 
                      const float blend,
                      const float gamma,
                      const bool keep_probs=False):
        cdef SR sr = convert_to_sr(indices, data, data.shape[0])
        cdef SR tree_probs

        # Get tree probs
        self.forest._predict(sr, tree_probs)

        # If blend == 1.0, we're done
        if blend == 1.0:
            return sr_to_sparse(tree_probs, self.forest.n_labels)

        cdef SR leaf_probs
        cdef vector[int] labels
        cdef int i

        # Build the indices
        for i in range(tree_probs.size()):
            labels.push_back(tree_probs[i].first)

        # Compute leaf classifier
        self.lc.predict(sr, labels, gamma, leaf_probs)

        cdef SR res
        self._blend(tree_probs, leaf_probs, blend, keep_probs, res)
        return sr_to_sparse(res, self.forest.n_labels)

    cdef void _blend(self, const SR& tree_probs, 
                           const SR& leaf_probs, 
                           const float blend, 
                           const bool keep_probs, 
                           SR& out):
        cdef int i
        cdef DP tp, lp, t
        for i in range(tree_probs.size()):
            tp = tree_probs[i]
            lp = leaf_probs[i]
            if keep_probs:
                tp.second *= blend
                lp.second *= (1 - blend)
            else:
                tp.second = log(tp.second) * blend
                lp.second = log(lp.second) * (1 - blend)

            t.first = tp.first
            t.second = tp.second + lp.second
            out.push_back(t)

cdef class IForestBlender:
    cdef IForest forest

    def __init__(self, IForest forest):
        self.forest = forest

    def predict(self, np.ndarray[np.float32_t] data, 
                      np.ndarray[np.int32_t] indices, 
                      const float blend,
                      const float gamma,
                      const bool keep_probs=False):

        return self.forest.predict(data, indices)

cdef class IForest:
    cdef list trees
    cdef int n_labels
    
    def __init__(self, str dname, int trees, int n_labels):
        self.n_labels = n_labels
        self.trees = []
        cdef int i
        for i in range(trees):
            self.trees.append(ITree(dname, i))
    
    cdef SR* _predict_tree(self, const SR& sr, ITree t):
        return t.predict_payload(sr)

    cdef void _predict(self, const SR& sr, SR& payload):
        cdef vector[SR*] prob_set
        cdef SR* res
        for t in self.trees:
            res = self._predict_tree(sr, t)
            prob_set.push_back(res)

        sparse_sr_mean(prob_set, payload)
    
    def predict(self, np.ndarray[np.float32_t] data, np.ndarray[np.int32_t] indices):
        cdef SR sr = convert_to_sr(indices, data, data.shape[0])
        cdef SR payload

        self._predict(sr, payload)
        return sr_to_sparse(payload, self.n_labels)

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

cdef class LeafComputer:
    cdef vector[float] norms
    cdef vector[float] radii
    cdef CSR means

    def __init__(self, str dname):
        p = dname.rstrip('/') + '/lc'

        # Load norms
        cdef DENSE tmp
        load_dense_f32(p + '.norms', tmp)
        self.norms.swap(tmp[0])

        # Load bias
        cdef DENSE tmp2
        load_dense_f32(p + '.radii', tmp2)
        self.radii.swap(tmp2[0])
        
        # Load means
        load_sparse(p + '.means', self.means)

    cdef void predict(self, const SR& X, const vector[int] ys, const float gamma, SR& out):
        cdef SR normed
        cdef int yi, i
        cdef DP p
        cdef float dist
        cdef SR* mean

        # Norm the vector
        norm(self.norms, X, normed)

        # Loop over each class, determining the leaf classifier vlaues
        for i in range(ys.size()):
            yi = ys[i]
            mean = &self.means[yi]
            dist = radius_sr(normed, deref(mean))
            k = exp(gamma  * (dist - self.radii[yi])) 
            p.first = i
            p.second = 1. / (1. + k)
            out.push_back(p)

cdef void norm(const vector[float]& norms, const SR& X, SR& normed):
    cdef int i
    cdef float l2 = 0
    cdef DP p

    # Column norm and compute l2 norm
    for i in range(X.size()):
        p = X[i]
        p.second /= norms[p.first]
        l2 += p.second * p.second
        normed.push_back(p)
    
    # Divide out the l2 norm
    l2 = pow(l2, .5)
    for i in range(normed.size()):
        normed[i].second = normed[i].second / l2


cdef float radius_sr(const SR& xi, const SR& ui):
    """
    Computes the Sum((Xi - Ux) ** 2)
    """

    cdef int xidx = 0, uidx = 0
    cdef int s1 = xi.size(), s2 = ui.size()
    cdef double tally = 0.0, diff
    cdef DP xp, up

    while xidx < s1 and uidx < s2:
        xp = xi[xidx]
        up = ui[uidx]
        if xp.first < up.first:
            tally += pow(xp.second, 2) 
            xidx += 1
        elif xp.first > up.first:
            tally += pow(up.second, 2)
            uidx += 1
        else:
            diff = xp.second - up.second
            tally += pow(diff, 2)
            xidx += 1
            uidx += 1

    # Get the remainder
    while xidx < s1 or uidx < s2:
        if xidx < s1:
            tally += pow(xi[xidx].second, 2)
            xidx += 1
        else:
            tally += pow(ui[uidx].second, 2)
            uidx += 1

    return tally

