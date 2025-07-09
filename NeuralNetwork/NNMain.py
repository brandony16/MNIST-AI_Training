import numpy as np
import cupy as cp
import pandas as pd
import time
import logging
import time
import argparse
from contextlib import contextmanager

from NeuralNetwork.NNModel import NeuralNetwork
from DatasetFunctions.LoadData import load_and_preprocess_data
from Visualization import show_all_metrics
from Optimizer import use_optimizer


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logging.info(f"{name} took {end - start:.3f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a NN on MNIST or CIFAR"
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
        "--lr-drop-every", type=int, default=10, help="Epoch interval to drop LR"
    )
    parser.add_argument(
        "--lr-drop-factor", type=float, default=0.1, help="Factor to multiply LR by"
    )
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
    parser.add_argument(
        "--activ",
        type=str,
        default="ReLU",
        help="Activation function to use",
    )

    return parser.parse_args()


def augment_batch(
    X: cp.ndarray,
    max_shift: int = 2,
    flip_prob: float = 0.5,
    brightness_jitter: float = 0.1,
):
    """
    Augments a batch of images with shifts and flips for less overfitting.

    Returns: The augmented data
    """
    N, C, _, _ = X.shape
    X_aug = X.copy()

    # Random shifts by up to max_shift pixels
    shifts = cp.random.randint(-max_shift, max_shift + 1, size=(N, 2))
    for i in range(N):
        dy, dx = int(shifts[i, 0]), int(shifts[i, 1])
        X_aug[i] = cp.roll(cp.roll(X_aug[i], dy, axis=1), dx, axis=2)

    # Random horizontal flips (only if more than 1 channel)
    if C > 1:
        flip_mask = cp.random.rand(N) < flip_prob
        # flip W axis (axis=3)
        X_aug[flip_mask] = X_aug[flip_mask, :, :, ::-1]

        # Brightness jitter
        # scale factor per image in [1-brightness_jitter, 1+brightness_jitter]
        deltas = (
            1
            + (cp.random.rand(N, 1, 1, 1, dtype=cp.float32) * 2 - 1) * brightness_jitter
        )
        X_aug = X_aug * deltas
        cp.clip(X_aug, 0.0, 1.0, out=X_aug)

    return X_aug


def load_data(dataset: str, valid_split: float, sample_size: int):
    """
    Loads the data for a model to train on.

    Returns:
        X_train: the training split of data.
        y_train: the labels for the training data (one-hot encoded).
        X_test: the test split of data.
        y_test: the labels for the testing data (one-hot encoded).
        input: the number of inputs for the network
        output: the number of classes the data can fall into
        class_names: an object that converts the numerical predictions to the actual names of the classes (Ex: 4 -> Dog).
    """
    logging.info("Loading and preprocessing data")
    X_train, y_train, X_test, y_test, input, output, class_names = load_and_preprocess_data(
        validation_split=valid_split, one_hot=True, use_dataset=dataset, flatten=False
    )

    # Make cupy arrays and get sample size
    X_train = X_train.astype(cp.float32)[:sample_size]
    y_train = y_train[:sample_size]
    X_test = X_test.astype(cp.float32)

    return X_train, y_train, X_test, y_test, input, output, class_names


def compute_metrics(model, X, y_onehot, batch_size=1024):
    """
    Computes loss and accuracy for a model given a batch of data and its labels
    """
    model.eval_mode()
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


def adjust_learning_rate(
    epoch: int, base_lr: float, drop_every: int, drop_factor: float
) -> float:
    """Reduce LR by drop_factor at each drop_every interval (excluding epoch 0)."""
    if epoch != 0 and epoch % drop_every == 0:
        return base_lr * drop_factor
    return base_lr


def train_and_evaluate(args, model, optimizer, X_train, y_train, X_test, y_test):
    """
    Trains a model and stores performance metrics on test and train loss and accuracy.

    Returns the history statistics as a list of objects.
    """
    history = []
    lr = args.lr

    for epoch in range(args.epochs + 1):
        lr = adjust_learning_rate(epoch, lr, args.lr_drop_every, args.lr_drop_factor)
        optimizer.set_lr(lr)

        if epoch > 0:
            model.train_mode()
            model.train(
                optimizer,
                X_train,
                y_train,
                batch_size=args.batch_size,
                augment_fn=augment_batch,
            )

        model.eval_mode()
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
    model.eval_mode()

    logging.info("Final evaluation on test set")
    test_loss, test_acc, labels, preds = compute_metrics(model, X_test, y_test)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

    df_history = pd.DataFrame(history)
    show_all_metrics(labels, preds, df_history, class_names)


def main():
    """
    Builds, trains, and evaluates a NN model.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    with timer("Total run"):
        X_train, y_train, X_test, y_test, input, output, class_names = load_data(
            args.dataset, args.valid_split, args.sample_size
        )

        layer_sizes = [input, 4096, 2048, 1024, 512, 128, output]
        model = NeuralNetwork(
            layer_sizes=layer_sizes,
            activation=args.activ,
        )
        optimizer = use_optimizer(model.parameters(), type=args.opt, lr=args.lr)

        history = train_and_evaluate(
            args, model, optimizer, X_train, y_train, X_test, y_test
        )

    evaluate_final(model, X_test, y_test, class_names, history)


if __name__ == "__main__":
    main()
