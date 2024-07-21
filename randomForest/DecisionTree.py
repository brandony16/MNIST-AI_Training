import numpy as np
from collections import Counter

class DecisionTree:
  def __init__(self, max_depth = None):
    self.max_depth = max_depth
  
  # X is the data, y is the labels
  def fit(self, X, y):
    # Number of classes. 10 bc 0-9
    self.n_classes_ = len(set(y))

    # Number of inputs. 784 bc 28x28 images
    self.n_features_ = X.shape[1]

    self.tree_ = self._grow_tree(X,y)
  
  def predict(self, X):
    # Calls predict for each input and returns label
    return[self._predict(inputs) for inputs in X]
  
  # This method finds the best feature and threshold to split the data to minimize the Gini impurity.  
  def _best_split(self, X, y):
    samples, features = X.shape
    if samples <= 1:
      return None, None

      
  
class Node:
  def __init__(self, gini):
    self.gini = gini