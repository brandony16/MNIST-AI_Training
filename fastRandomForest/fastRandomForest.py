import numpy as np
from fastDecisionTree import FastDecisionTree
from joblib import Parallel, delayed
from typing import Optional


class FastRandomForest:
    """
    A fast Random Forest classifier using multiple Numba-accelerated DecisionTree
    learners, bootstrap sampling, and parallel training/prediction.
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
    ):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

        trees_per_job = np.ceil(num_trees / n_jobs)
        self.trees_per_job = trees_per_job

        # Will be populated after fit()
        self.trees_ = []  # list of DecisionTree
        self.classes_ = None  # array of unique class labels

    def fit(self, data: np.ndarray, labels: np.ndarray):
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        num_samples, num_features = data.shape
        self.classes_ = np.unique(labels)

        rng = np.random.RandomState(self.random_state)

        if self.bootstrap:
            all_indices = [
                rng.randint(0, num_samples, size=num_samples)
                for _ in range(self.num_trees)
            ]
        else:
            all_indices = [np.arange(num_samples) for _ in range(self.num_trees)]

        seeds = list(range(self.num_trees))

        chunks = [
            all_indices[i : i + self.trees_per_job]
            for i in range(0, len(all_indices), self.trees_per_job)
        ]
        seed_chunks = [
            seeds[i : i + self.trees_per_job]
            for i in range(0, self.num_trees, self.trees_per_job)
        ]

        self.trees = Parallel(n_jobs=self.n_jobs, verbose=0, backend="threading")(
            delayed(self._fit_batch)(data, labels, idx_chunk, seed_chunk)
            for idx_chunk, seed_chunk in zip(chunks, seed_chunks)
        )

        return self

    def _fit_batch(
        self, data: np.ndarray, labels: np.ndarray, index_arrays: list, seeds: list
    ):
        trees = []
        for idxs, seed in zip(index_arrays, seeds):
            # reseed so feature‐subsampling is reproducible
            np.random.seed((self.random_state or 0) + seed)
            data_s, labels_s = data[idxs], labels[idxs]
            tree = FastDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(data_s, labels_s)
            trees.append(tree)
        return trees

    def _fit_tree(self, data: np.ndarray, labels: np.ndarray, sample_indices, seed):
        np.random.seed(
            self.random_state + seed if self.random_state is not None else None
        )

        data_sample = data[sample_indices]
        labels_sample = labels[sample_indices]

        tree = FastDecisionTree(
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
        )
        tree.fit(data_sample, labels_sample)

        return tree

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X by majority vote over all trees.
        """
        data = np.asarray(data, dtype=np.float32)
        num_samples = data.shape[0]

        predictions = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(tree.predict)(data) for tree in self.trees
        )
        predictions = np.vstack(predictions)

        # Majority Vote
        final = np.empty(num_samples, dtype=np.int32)
        for i in range(num_samples):
            counts = np.bincount(predictions[:, i], minlength=len(self.classes_))
            final[i] = counts.argmax()

        return final
