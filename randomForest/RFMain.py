from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from DatasetFunctions.LoadData import load_and_preprocess_data
from Visualization import show_final_metrics
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a NN on MNIST or CIFAR"
    )
    parser.add_argument("--dataset", default="MNIST", help="Which dataset to load")
    parser.add_argument(
        "--valid-split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument("--trees", type=int, default=100, help="Number of trees")
    parser.add_argument(
        "--max-depth", type=int, default=20, help="Max depth a tree can go"
    )
    parser.add_argument(
        "--max-feat",
        type=int,
        default=28,
        help="Max features to consider at each split",
    )
    parser.add_argument(
        "--min-samp-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=8,
        help="Number of jobs to run in parallel. Affects speed, not accuracy.",
    )

    return parser.parse_args()


def main():
    print("Starting RF Progam")

    args = parse_args()

    # Get relevant split and class name data
    X_train, y_train, X_test, y_test, _, _, class_names = load_and_preprocess_data(
        validation_split=args.valid_split,
        one_hot=False,
        use_dataset=args.dataset,
        flatten=True,
    )

    sample_size = -1
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]

    # Initialize and train the RandomForestClassifier
    start = time.perf_counter()
    forest = RandomForest(
        num_trees=args.trees,
        max_depth=args.max_depth,
        max_features=args.max_feat,
        min_samples_leaf=args.min_samp_leaf,
        n_jobs=args.njobs,
        random_state=42,
    )

    print("Building Forest")
    forest.fit(X_train_sample, y_train_sample)

    # Make predictions and evaluate the model
    y_pred = forest.predict(X_test)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy, {accuracy * 100:.2f}%")

    show_final_metrics(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()
