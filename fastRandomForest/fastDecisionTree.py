import numpy as np
from fastSplit import fast_best_split
from numba import njit

@njit(nogil=True)
def _predict_one_njit(x, feat, thr, left, right, pred):
    node = 0
    # feature < 0 marks a leaf
    while feat[node] >= 0:
        if x[feat[node]] < thr[node]:
            node = left[node]
        else:
            node = right[node]
    return pred[node]


class FastDecisionTree:
    """
    A binary decision tree classifier using Gini impurity
    and an optimized, Numba-accelerated split routine.
    """

    def __init__(
        self,
        max_depth: int = None,
        max_features: int = None,
        min_samples_leaf: int = 1,
    ):
        self.max_depth = max_depth or np.inf
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.num_samples, self.num_features = data.shape
        self.num_classes = int(labels.max() + 1)
        self.tree = self._grow_tree(data, labels, depth=0)

        self._node_count = 0
        self._count_nodes(self.tree)
        N = self._node_count

        self._feat = np.empty(N, dtype=np.int32)
        self._thr = np.empty(N, dtype=np.float32)
        self._left = np.empty(N, dtype=np.int32)
        self._right = np.empty(N, dtype=np.int32)
        self._pred = np.empty(N, dtype=np.int32)

        # fill them in preorder
        self._node_count = 0
        self._export(self.tree)

    def predict(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            out[i] = _predict_one_njit(
                data[i], self._feat, self._thr, self._left, self._right, self._pred
            )
        return out

    # Build tree recursively
    def _grow_tree(self, data: np.ndarray, labels: np.ndarray, depth: int) -> dict:
        num_entries = labels.size
        counts = np.bincount(labels, minlength=self.num_classes)
        prediction = int(np.argmax(counts))
        probability = counts / num_entries
        gini = 1.0 - np.sum(probability * probability)

        # Terminal Condition
        leaf_node = {
            "feature": -1,
            "threshold": 0.0,
            "left": None,
            "right": None,
            "prediction": prediction,
        }
        if (
            depth >= self.max_depth
            or num_entries <= self.min_samples_leaf
            or gini == 0.0
        ):
            return leaf_node

        feature, threshold = fast_best_split(
            data, labels, self.num_classes, self.max_features
        )
        if feature < 0:
            return leaf_node

        mask_left = data[:, feature] < threshold
        data_left, labels_left = data[mask_left], labels[mask_left]
        data_right, labels_right = data[~mask_left], labels[~mask_left]

        if (
            labels_left.size < self.min_samples_leaf
            or labels_right.size < self.min_samples_leaf
        ):
            return leaf_node

        left_node = self._grow_tree(data_left, labels_left, depth + 1)
        right_node = self._grow_tree(data_right, labels_right, depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_node,
            "right": right_node,
            "prediction": prediction,
        }

    def _count_nodes(self, node):
        self._node_count += 1
        if node["feature"] >= 0:
            self._count_nodes(node["left"])
            self._count_nodes(node["right"])

    def _export(self, node):
        idx = self._node_count
        self._node_count += 1

        self._feat[idx] = node["feature"]
        self._thr[idx] = node["threshold"]
        self._pred[idx] = node["prediction"]
        if node["feature"] < 0:
            # leaf: point children at self to stop
            self._left[idx] = idx
            self._right[idx] = idx
        else:
            # reserve space for children
            left_idx = self._node_count
            self._left[idx] = left_idx
            self._export(node["left"])

            right_idx = self._node_count
            self._right[idx] = right_idx
            self._export(node["right"])
