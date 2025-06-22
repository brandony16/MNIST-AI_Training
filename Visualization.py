import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

def plot_curves(dataframe):
    # Loss
    plt.figure()
    plt.plot(dataframe["epoch"], dataframe["train_loss"], label="Train Loss")
    plt.plot(dataframe["epoch"], dataframe["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(dataframe["epoch"], dataframe["train_acc"], label="Train Acc")
    plt.plot(dataframe["epoch"], dataframe["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.show()


def show_confusion_matrix(labels, predictions):
    confMat = confusion_matrix(labels, predictions, normalize="true")
    plt.figure()
    plt.imshow(confMat, interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(confMat.shape[1]))
    plt.yticks(np.arange(confMat.shape[0]))
    plt.show()

def show_metrics(labels, predictions):
  precision = precision_score(labels, predictions, average=None)
  recall = recall_score(labels, predictions, average=None)
  f1 = f1_score(labels, predictions, average=None)

  metrics_df = pd.DataFrame({
    'precision': precision,
    'recall':    recall,
    'f1_score':  f1
  }, index=[str(i) for i in range(len(precision))])

  metrics_df.plot.bar()
  plt.xlabel("Class")
  plt.ylabel("Score")
  plt.title("Precision / Recall / F1 by Class")
  plt.show()