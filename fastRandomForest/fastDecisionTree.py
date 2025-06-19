import numpy as np
from fastSplit import fast_best_split


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
        self.num_classes = len(set(labels))
        self.tree = self._grow_tree(data, labels, depth=0)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(input) for input in data], dtype=np.int32)

    def _grow_tree(self, data: np.ndarray, labels: np.ndarray, depth: int) -> dict:
        num_entries = labels.size
        counts = np.bincount(labels, minlength=self.num_classes)
        prediction = int(np.argmax(counts))
        probability = counts / num_entries
        gini = 1.0 - np.sum(probability * probability)

        # Terminal Condition
        leaf_node = {
            "feature": None,
            "threshold": None,
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
        if feature is None:
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

    def _predict_one(self, input: np.ndarray) -> int:
        node = self.tree
        while node["feature"] is not None:
            if input[node["feature"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]
