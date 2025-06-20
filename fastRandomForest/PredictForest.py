import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _predict_forest_njit(feat, thr, left, right, pred, data_test, out, num_classes):
    """
    Monolithic JIT for RandomForest.predict:
      - feat[t,i] is the feature idx for node i of tree t (-1 for leaf)
      - thr[t,i] is the threshold
      - left[t,i], right[t,i] are child-node indices
      - pred[t,i] is only valid when feat[t,i] == -1
      - X_test: shape (n_test, n_features)
      - out: preallocated int32 array of shape (n_test,)
      - num_classes: number of classes
    """

    n_trees, n_nodes = feat.shape
    n_test, n_features = data_test.shape

    for j in prange(n_test):
        counts = np.zeros(num_classes, np.int32)
        for t in range(n_trees):
            node = 0
            # descend until leaf
            while feat[t, node] != -1:
                # branch
                f = feat[t, node]
                if data_test[j, f] < thr[t, node]:
                    node = left[t, node]
                else:
                    node = right[t, node]
            # at leaf, increment its class vote
            counts[pred[t, node]] += 1

        # find the majority class
        best = 0
        best_count = counts[0]
        for c in range(1, num_classes):
            if counts[c] > best_count:
                best = c
                best_count = counts[c]
        out[j] = best
