import numpy as np
from DecisionTreeModule import DecisionTree
from sklearn.utils import resample
from scipy import stats

class RandomForest:
  def __init__(self, n_trees=100, max_depth=None):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.trees = []

  def fit(self, X, y):
    self.trees = []
    for _ in range(self.n_trees):
      X_sample, y_sample = resample(X, y)
      tree = DecisionTree(max_depth=self.max_depth)
      tree.fit(X_sample, y_sample)
      self.trees.append(tree)

  def predict(self, X):
    tree_predictions = np.array([tree.predict(X) for tree in self.trees])
    # Find the mode of the predictions of the trees
    return np.squeeze(stats.mode(tree_predictions, axis=0)[0])
