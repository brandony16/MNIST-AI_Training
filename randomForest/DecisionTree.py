import numpy as np
from typing import Optional
from BestSplit import best_split


class DecisionTree:
    """
    Class for a single tree of a random forest
    """

    def __init__(
        self,
        max_depth: int = None,
        max_features: int = None,
        min_samples_split: int = 2,
    ):
        self.max_depth = max_depth or np.inf
        self.max_features = max_features
        self.min_samples_split = min_samples_split

    # Fits the tree to the data
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]

        self.tree_ = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X], dtype=int)

    def _best_split(self, X, y):
        return best_split(X, y, self.n_classes_, self.max_features)

    # Recursively builds the decision tree
    def _grow_tree(self, X, y, depth=0):
        # Terminal break if pure or too small or too deep
        if (
            len(y) < self.min_samples_split
            or depth >= self.max_depth
            or np.all(y == y[0])
        ):
            counts = np.bincount(y, minlength=self.n_classes_)
            pred = int(np.argmax(counts))
            gini = 1.0 - np.sum((counts / len(y)) ** 2)
            return Node(gini, len(y), counts, pred)

        # Parent counts & impurity
        parent_counts = np.bincount(y, minlength=self.n_classes_)
        parent_gini = 1.0 - np.sum((parent_counts / len(y)) ** 2)
        idx, thr = self._best_split(X, y)

        # If no split found, make leaf
        if idx is None:
            pred = int(np.argmax(parent_counts))
            return Node(parent_gini, len(y), parent_counts, pred)

        # Make new node and recursively grow tree
        mask_left = X[:, idx] < thr
        X_left, y_left = X[mask_left], y[mask_left]
        X_right, y_right = X[~mask_left], y[~mask_left]

        node = Node(parent_gini, len(y), parent_counts, int(np.argmax(parent_counts)))
        node.feature_index = idx
        node.threshold = thr

        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    # Predicts for a simple input
    def _predict(self, entry):
        node = self.tree_
        while node.left is not None:
            if entry[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


class Node:
    """
    Basic node class that stores relevant data for each node of a decision tree
    """

    __slots__ = (
        "gini",
        "num_samples",
        "num_samples_per_class",
        "predicted_class",
        "feature_index",
        "threshold",
        "left",
        "right",
    )

    def __init__(
        self,
        gini: float,
        num_samples: int,
        num_samples_per_class: np.ndarray,
        predicted_class: int,
    ):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
