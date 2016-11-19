from collections import Counter

cdef int MAX_LABELS = 15000000
cdef list LOGS = [1.0 / i for i in xrange(2, MAX_LABELS + 2)]

cdef float dcg(dict order, list ls):
    """
    We only need to use DCG since we're only using it to determine which partition
    bucket the label set in
    """
    cdef float s = 0
    for l in ls:
        if l in order:
            s += LOGS[order[l]]

    return s

def count_labels(y, idxs):
    it = (yi for i in idxs for yi in y[i])
    return Counter(it)

def order_labels(y, idxs):
    counter = count_labels(y, idxs)
    sLabels = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
    return {l: i for i, l in enumerate(sLabels)}

cdef tuple divide(list y, list idxs, dict lOrder, dict rOrder):
    cdef list newLeft, newRight
    cdef float lNdcg, rNdcg
    newLeft, newRight = [], []
    for i in idxs:
        lNdcg = dcg(lOrder, y[i])
        rNdcg = dcg(rOrder, y[i])
        if lNdcg >= rNdcg:
            newLeft.append(i)
        else:
            newRight.append(i)

    return newLeft, newRight

def split_node(list y, list idxs, rs, even_split):
    cdef list left, left1, left2, right, right1, right2

    if even_split:
        ix = idxs[:]
        rs.shuffle(ix)
        left = ix[:len(ix)/2]
        right = ix[len(ix)/2:]
    else:
        left, right = [], []
        for i in idxs:
            (left if rs.rand() < 0.5 else right).append(i)

    while True:

        # Build ndcg for the sides
        lOrder = order_labels(y, left)
        rOrder = order_labels(y, right)

        # Divide out the sides
        left1, right1 = divide(y, left, lOrder, rOrder)
        left2, right2 = divide(y, right, lOrder, rOrder)
        if not right1 and not left2:
            # Done!
            break

        left = left1 + left2
        right = right1 + right2

    return left, right
