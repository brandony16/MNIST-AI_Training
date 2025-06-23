import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def build_plot_curves_fig(dataframe):
    # Loss
    plt.figure()
    plt.plot(dataframe["epoch"], dataframe["train_loss"], label="Train Loss")
    plt.plot(dataframe["epoch"], dataframe["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    # Accuracy
    plt.figure()
    plt.plot(dataframe["epoch"], dataframe["train_acc"], label="Train Acc")
    plt.plot(dataframe["epoch"], dataframe["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()


def build_confusion_matrix_fig(labels, predictions):
    confMat = confusion_matrix(labels, predictions, normalize="true")
    plt.figure()
    plt.imshow(confMat, interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(confMat.shape[1]))
    plt.yticks(np.arange(confMat.shape[0]))


def build_metrics_fig(labels, predictions):
    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)
    f1 = f1_score(labels, predictions, average=None)

    metrics_df = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1_score": f1},
        index=[str(i) for i in range(len(precision))],
    )

    metrics_df.plot.bar()
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 by Class")


def show_all_metrics(labels, predictions, dataframe, class_map):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Loss
    if (dataframe["train_loss"].notna().any()) or (
        dataframe["test_loss"].notna().any()
    ):
        axes[0, 0].plot(dataframe["epoch"], dataframe["train_loss"], label="Train Loss")
        axes[0, 0].plot(dataframe["epoch"], dataframe["test_loss"], label="Test Loss")
        axes[0, 0].set(title="Loss vs. Epoch", xlabel="Epoch", ylabel="Loss")
        axes[0, 0].legend()
    else:
        axes[0, 0].set(title="N/A for Selected Model")

    # Accuracy
    axes[0, 1].plot(dataframe["epoch"], dataframe["train_acc"], label="Train Acc")
    axes[0, 1].plot(dataframe["epoch"], dataframe["test_acc"], label="Test Acc")
    axes[0, 1].set(title="Accuracy vs. Epoch", xlabel="Epoch", ylabel="Accuracy")
    axes[0, 1].legend()

    class_names = [class_map[i] for i in range(len(class_map))]

    # Norm Confusion Matrix
    confMat = confusion_matrix(labels, predictions, normalize="true")
    axes[1, 0].imshow(confMat, interpolation="nearest", vmin=0, vmax=1)
    axes[1, 0].set(
        title="Confusion Matrix Heatmap",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=range(len(class_names)),
        xticklabels=class_names,
        yticks=range(len(class_names)),
        yticklabels=class_names,
    )
    plt.setp(axes[1, 0].get_xticklabels(), rotation=90)

    # Precision, Recall, & F1 per class
    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)
    f1 = f1_score(labels, predictions, average=None)

    metrics_df = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1_score": f1},
        index=[str(i) for i in range(len(precision))],
    )

    metrics_df.plot.bar(ax=axes[1, 1])
    axes[1, 1].set(
        title="Precision/Recall/F1 by Class",
        xlabel="Class",
        ylabel="Score",
        xticks=range(len(class_names)),
        xticklabels=class_names,
    )

    fig.tight_layout()
    plt.show()
