import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import cupy as cp

from DatasetFunctions.LoadData import load_cnn_data
from Sequential import Sequential
from SGD import use_optimizer
from Visualization import show_all_metrics
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
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN on MNIST or CIFAR")
    parser.add_argument("--dataset", default="MNIST", help="Which dataset to load")
    parser.add_argument(
        "--valid-split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument(
        "--batch-size", type=int, default=96, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--lr-drop-every", type=int, default=5, help="Epoch interval to drop LR"
    )
    parser.add_argument(
        "--lr-drop-factor", type=float, default=0.3, help="Factor to multiply LR by"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Subset size of training data (-1=all)",
    )
    parser.add_argument(
        "--model-name", default=None, help="Name for saving/loading weights"
    )
    parser.add_argument(
        "--out-history", default="training_history.json", help="Where to dump history"
    )
    return parser.parse_args()


def load_data(dataset: str, valid_split: float, sample_size: int):
    logging.info("Loading and preprocessing data")
    X_train, y_train, X_test, y_test, _, _, class_names = load_cnn_data(
        validation_split=valid_split, one_hot=True, use_dataset=dataset
    )

    # Make cupy arrays and get sample size
    X_train = X_train.astype(cp.float32)[:sample_size]
    y_train = y_train[:sample_size]
    X_test = X_test.astype(cp.float32)

    return X_train, y_train, X_test, y_test, class_names


def build_model(model_name: str, dataset: str):
    logging.info("Building model")

    if dataset == "CIFAR":
        params = CIFAR_PARAMETERS
    else:
        params = MNIST_PARAMETERS
    model = Sequential([ctor() for ctor in params])

    # check for saved weights
    if model_name:
        weights_path = Path(f"{model_name}.npz")
        if weights_path.exists():
            logging.info(f"Loading weights from {weights_path}")
            model.load(model_name, *params)
    return model


def adjust_learning_rate(
    epoch: int, base_lr: float, drop_every: int, drop_factor: float
) -> float:
    """Reduce LR by drop_factor at each drop_every interval (excluding epoch 0)."""
    if epoch != 0 and epoch % drop_every == 0:
        return base_lr * drop_factor
    return base_lr


def compute_metrics(model, X, y_onehot, batch_size=256):
    """Returns (loss, accuracy)."""
    num_samples = X.shape[0]
    total_loss = 0.0
    labels = np.argmax(y_onehot, axis=1)

    for i in range(0, num_samples, batch_size):
        X_batch = X[i : i + batch_size]
        y_batch = y_onehot[i : i + batch_size]

        # Forward pass
        batch_loss = model.forward(X_batch, y_batch)
        total_loss += float(batch_loss) * X_batch.shape[0]  # weighted sum

    # Predict is already batched in Sequential 
    preds = cp.asnumpy(model.predict(X))
    acc = np.mean(preds == labels)

    avg_loss = total_loss / num_samples
    return avg_loss, acc, labels, preds


def train_and_evaluate(args, model, optimizer, X_train, y_train, X_test, y_test):
    history = []
    lr = args.lr

    for epoch in range(args.epochs + 1):
        lr = adjust_learning_rate(epoch, lr, args.lr_drop_every, args.lr_drop_factor)
        optimizer.set_lr(lr)

        if epoch > 0:
            model.train(optimizer, X_train, y_train, batch_size=args.batch_size)

        train_loss, train_acc, _, _ = compute_metrics(model, X_train, y_train)
        test_loss, test_acc, _, _ = compute_metrics(model, X_test, y_test)

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

        logging.info(f"Epoch {epoch}: \ntest_acc={test_acc*100:.2f}%, \ntrain_acc={train_acc*100:.2f}%")

        if args.model_name:
            model.save(args.model_name)

    return history


def evaluate_final(model, X_test, y_test, class_names, history):
    logging.info("Final evaluation on test set")
    test_loss, test_acc, labels, preds = compute_metrics(model, X_test, y_test)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

    df_history = pd.DataFrame(history)
    show_all_metrics(labels, preds, df_history, class_names)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    with timer("Total run"):
        X_train, y_train, X_test, y_test, class_names = load_data(
            args.dataset, args.valid_split, args.sample_size
        )
        X_train = X_train[:args.sample_size]
        y_train = y_train[:args.sample_size]
        X_test = X_test[:args.sample_size]
        y_test = y_test[:args.sample_size]

        model = build_model(args.model_name, args.dataset)
        optimizer = use_optimizer(model.parameters(), type="Adam", lr=args.lr)

        history = train_and_evaluate(
            args, model, optimizer, X_train, y_train, X_test, y_test
        )

        # Save history to JSON
        with open(args.out_history, "w") as f:
            json.dump(history, f, indent=4)

    evaluate_final(model, X_test, y_test, class_names, history)


if __name__ == "__main__":
    main()
