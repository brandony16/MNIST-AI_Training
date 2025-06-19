import numpy as np
from DecisionTree import DecisionTree
from sklearn.utils import resample
from scipy import stats
from joblib import Parallel, delayed


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, n_jobs=-1, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_tree)(X, y) for _ in range(self.n_trees)
        )

    def _fit_tree(self, X, y):
        print("fitting tree")
        X_sample, y_sample = resample(X, y)
        tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        tree_predictions = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict)(X) for tree in self.trees
            )
        )
        
        # Find the mode of the predictions of the trees
        return np.squeeze(stats.mode(tree_predictions, axis=0)[0])
