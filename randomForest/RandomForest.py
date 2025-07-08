import numpy as np
from DecisionTree import DecisionTree
from joblib import Parallel, delayed
from typing import Optional
from PredictForest import _predict_forest_njit
import multiprocessing


class RandomForest:
    """
    A fast Random Forest classifier using multiple Numba-accelerated DecisionTree
    learners, bootstrap sampling, and parallel training/prediction.
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        min_samples_leaf: int = 2,
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
    ):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.random_state = random_state

        trees_per_job = (num_trees + n_jobs - 1) // n_jobs
        self.trees_per_job = trees_per_job

        # Will be populated after fit()
        self.trees_ = []  # list of DecisionTree
        self.classes_ = None  # array of unique class labels

    def fit(self, data: np.ndarray, labels: np.ndarray):
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        num_samples = data.shape[0]
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

        tree_batches = Parallel(n_jobs=self.n_jobs, verbose=0, backend="threading")(
            delayed(self._fit_batch)(data, labels, idx_chunk, seed_chunk)
            for idx_chunk, seed_chunk in zip(chunks, seed_chunks)
        )

        self.trees_ = [tree for batch in tree_batches for tree in batch]

        node_counts = [tree._feat.shape[0] for tree in self.trees_]
        max_nodes = max(node_counts)

        # 3) allocate forest‐level arrays, padding each tree to max_nodes
        n_trees = len(self.trees_)
        self.feat_arr = np.full((n_trees, max_nodes), -1, dtype=np.int32)
        self.thr_arr = np.zeros((n_trees, max_nodes), dtype=np.float32)
        self.left_arr = np.zeros((n_trees, max_nodes), dtype=np.int32)
        self.right_arr = np.zeros((n_trees, max_nodes), dtype=np.int32)
        self.pred_arr = np.zeros((n_trees, max_nodes), dtype=np.int32)

        for t, tree in enumerate(self.trees_):
            n = tree._feat.shape[0]
            # copy into the first n slots; the rest stay as “leaf” (-1 feature)
            self.feat_arr[t, :n] = tree._feat
            self.thr_arr[t, :n] = tree._thr
            self.left_arr[t, :n] = tree._left
            self.right_arr[t, :n] = tree._right
            self.pred_arr[t, :n] = tree._pred

        return self

    def _fit_batch(
        self, data: np.ndarray, labels: np.ndarray, index_arrays: list, seeds: list
    ):
        trees = []
        for idxs, seed in zip(index_arrays, seeds):
            # reseed so feature‐subsampling is reproducible
            np.random.seed((self.random_state or 0) + seed)
            data_s, labels_s = data[idxs], labels[idxs]
            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(data_s, labels_s)
            trees.append(tree)
        return trees

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X by majority vote over all trees.
        """
        data = np.asarray(data, dtype=np.float32)
        n_test = data.shape[0]
        y_out = np.empty(n_test, dtype=np.int32)
        num_classes = int(self.pred_arr.max()) + 1

        _predict_forest_njit(
            self.feat_arr,
            self.thr_arr,
            self.left_arr,
            self.right_arr,
            self.pred_arr,
            data,
            y_out,
            num_classes,
        )

        return y_out


def resolve_n_jobs(n_jobs):
    """
    Turns n_jobs into a positive integer:
    - If n_jobs > 0: use that many jobs.
    - If n_jobs == 0: treat as 1.
    - If n_jobs < 0: use (n_cores + 1 + n_jobs), e.g. -1 → all cores, -2 → all but one.
    """
    n_cpus = multiprocessing.cpu_count()  # total logical cores
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, n_cpus + 1 + n_jobs)
    return n_jobs
