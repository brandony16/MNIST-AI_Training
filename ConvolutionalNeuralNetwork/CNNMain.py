import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import math
import numpy as np
import pandas as pd
import cupy as cp

from DatasetFunctions.LoadData import load_and_preprocess_data
from Sequential import Sequential
from SGD import use_optimizer
from Visualization import show_all_metrics
from OneCycle import OneCycleScheduler
from ConvolutionalNeuralNetwork.Architectures import (
    MNIST_PARAMETERS,
    CIFAR_PARAMETERS,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logging.info(f"{name} took {end - start:.3f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN on MNIST or CIFAR"
    )
    parser.add_argument("--dataset", default="MNIST", help="Which dataset to load")
    parser.add_argument(
        "--valid-split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Subset size of training data (-1=all)",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="SGD",
        help="optimizer to use (Adam or SGD)",
    )

    return parser.parse_args()


def augment_batch(
    X: cp.ndarray,
    max_shift: int = 2,
    flip_prob: float = 0.5,
):
    """
    Augments a batch of images with shifts and flips for less overfitting.

    Returns: The augmented data
    """
    N, C, _, _ = X.shape
    X_aug = X.copy()

    # shifts and flips only
    shifts = cp.random.randint(-max_shift, max_shift + 1, (N, 2))
    for i in range(N):
        dy, dx = shifts[i]
        X_aug[i] = cp.roll(cp.roll(X_aug[i], int(dy), 1), int(dx), 2)

    if C > 1:
        mask = cp.random.rand(N) < flip_prob
        X_aug[mask] = X_aug[mask, :, :, ::-1]  # Flip width wise

    return X_aug


def load_data(dataset: str, valid_split: float, sample_size: int):
    """
    Loads the data for a model to train on.

    Returns:
        X_train: the training split of data.
        y_train: the labels for the training data (one-hot encoded).
        X_test: the test split of data.
        y_test: the labels for the testing data (one-hot encoded).
        class_names: an object that converts the numerical predictions to the actual names of the classes (Ex: 4 -> Dog).
    """
    logging.info("Loading and preprocessing data")
    X_train, y_train, X_test, y_test, _, _, class_names = load_and_preprocess_data(
        validation_split=valid_split, one_hot=True, use_dataset=dataset
    )

    # Make cupy arrays and get sample size
    X_train = X_train.astype(cp.float32)[:sample_size]
    y_train = y_train[:sample_size]
    X_test = X_test.astype(cp.float32)

    return X_train, y_train, X_test, y_test, class_names


def build_model(dataset: str):
    """
    Builds a model using a Sequential class.

    Returns: the model.
    """
    logging.info("Building model")

    if dataset == "CIFAR":
        params = CIFAR_PARAMETERS
    else:
        params = MNIST_PARAMETERS
    model = Sequential([ctor() for ctor in params])

    return model


def compute_metrics(model, X, y_onehot, batch_size=1024):
    """
    Computes loss and accuracy for a model given a batch of data and its labels
    """
    model.eval()
    labels = np.argmax(y_onehot, axis=1)

    preds, loss = model.predict(X, y_onehot, batch_size)

    preds = cp.asnumpy(preds)
    acc = np.mean(preds == labels)

    return loss, acc, labels, preds


def train_and_evaluate(
    args, model: Sequential, optimizer, X_train, y_train, X_test, y_test
):
    """
    Trains a model and stores performance metrics on test and train loss and accuracy.

    Returns the history statistics as a list of objects.
    """
    history = []
    lr = args.lr

    steps_per_epoch = math.ceil(len(X_train) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    scheduler = OneCycleScheduler(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.02,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            model.train_mode()
            model.train(
                optimizer,
                X_train,
                y_train,
                batch_size=args.batch_size,
                augment_fn=augment_batch,
                scheduler=scheduler,
            )

        model.eval()
        # Random subset of 5000 for each to speed up evaluation.
        test_idxs = np.random.choice(len(X_test), size=5000, replace=False)
        train_idxs = np.random.choice(len(X_train), size=5000, replace=False)
        train_loss, train_acc, _, _ = compute_metrics(
            model, X_train[train_idxs], y_train[train_idxs]
        )
        test_loss, test_acc, _, _ = compute_metrics(
            model, X_test[test_idxs], y_test[test_idxs]
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        logging.info(
            f"Epoch {epoch}: test_loss={test_loss:.2f}, test_acc={test_acc*100:.2f}%"
        )
        logging.info(
            f"Epoch {epoch}: train_loss={train_loss:.2f}, train_acc={train_acc*100:.2f}%"
        )

    return history


def evaluate_final(model, X_test, y_test, class_names, history):
    """
    Does the final evaluation of a model on the test data.
    Also generates and displays the graphs of its performance over the epochs.
    """
    model.eval()

    logging.info("Final evaluation on test set")
    test_loss, test_acc, labels, preds = compute_metrics(model, X_test, y_test)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

    df_history = pd.DataFrame(history)
    show_all_metrics(labels, preds, df_history, class_names)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """
    Builds, trains, and evaluates a CNN model.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    with timer("Total run"):
        X_train, y_train, X_test, y_test, class_names = load_data(
            args.dataset, args.valid_split, args.sample_size
        )

        model = build_model(args.dataset)
        optimizer = use_optimizer(model.parameters(), type=args.opt, lr=args.lr)

        history = train_and_evaluate(
            args, model, optimizer, X_train, y_train, X_test, y_test
        )

    evaluate_final(model, X_test, y_test, class_names, history)


if __name__ == "__main__":
    main()
