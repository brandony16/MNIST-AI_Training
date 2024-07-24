import numpy as np
from collections import Counter

class DecisionTree:
  def __init__(self, max_depth = None, max_features=None):
    self.max_depth = max_depth
    self.max_features = max_features
  
  # X is the data, y is the labels
  # Fits the tree to the data
  def fit(self, X, y):
    # Number of classes. 10 bc 0-9
    self.n_classes_ = len(set(y))

    # Number of inputs. 784 bc 28x28 images
    self.n_features_ = X.shape[1]

    self.tree_ = self._grow_tree(X,y)
  
  def predict(self, X):
    # Calls predict for each input and returns predicted labels
    return[self._predict(inputs) for inputs in X]
  
  # This method finds the best feature and threshold to split the data to minimize the Gini impurity.  
  def _best_split(self, X, y):
    samples, features = X.shape
    if samples <= 1:
      return None, None

    # List of counts of each class in the current node
    num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
    # Calculates the gini impurity for the current node
    best_gini = 1.0 - sum((num/samples) ** 2 for num in num_parent)
    best_idx, best_thr = None, None

    feature_indices = np.random.choice(features, self.max_features, replace=False) if self.max_features else range(features)

    for idx in feature_indices:
      # Sort the samples by the current feature values. Thresholds are the sorted feature values, and classes are the corresponding class labels.
      thresholds, classes = zip(*sorted(zip(X[:, idx] , y)))

      num_left = [0] * self.n_classes_
      num_right = num_parent.copy()

      # Loop over possible split points and update class counts
      for i in range(1, samples):
        c = classes[i - 1]
        num_left[c] += 1
        num_right[c] -= 1

        # Calculate gini for both sides of split then calculate weighted impurity
        gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
        gini_right = 1.0 - sum((num_right[x] / (samples - i)) ** 2 for x in range(self.n_classes_))
        gini = (i * gini_left + (samples - i) * gini_right) / samples

        # Skip repeating thresholds
        if thresholds[i] == thresholds[i - 1]:
          continue
        if gini < best_gini:
          best_gini = gini
          best_idx = idx
          best_thr = (thresholds[i] + thresholds[i - 1]) / 2

    return best_idx, best_thr

  # Recursively builds the decision tree
  def _grow_tree(self, X, y, depth=0):
    num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
    # Class with most samples in current node
    predicted_class = np.argmax(num_samples_per_class)
    # Initializes a node
    node = Node(
      gini = 1.0 - sum((num/len(y)) ** 2 for num in num_samples_per_class),
      num_samples = len(y),
      num_samples_per_class = num_samples_per_class,
      predicted_class = predicted_class
    )

    if depth < self.max_depth:
      idx, thr = self._best_split(X,y)
      if idx is not None:
        indices_left = X[:, idx] < thr
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        node.feature_index = idx
        node.threshold = thr

        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
    return node
  
  def _predict(self, inputs):
    node = self.tree_
    while node.left:
      if inputs[node.feature_index] < node.threshold:
        node = node.left
      else:
        node = node.right
    return node.predicted_class
  
class Node:
  def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
    self.gini = gini
    self.num_samples = num_samples
    self.num_samples_per_class = num_samples_per_class
    self.predicted_class = predicted_class
    self.feature_index = 0
    self.threshold = 0
    self.left = None
    self.right = None