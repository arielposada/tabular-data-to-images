import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from tabular_data_to_images.common.dataset import augment_data, load_dataset, split_data
from tabular_data_to_images.common.train_utils import set_seed
from tabular_data_to_images.baselines.model import get_model

METRIC_NAMES = ["f1_score", "accuracy", "precision", "recall"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_single_seed(dataset_type, model_type, augment_strategy, seed):
    set_seed(seed)

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False, random_state=seed
    )
    X_train, y_train = augment_data(X_train, y_train, X_test, augment_strategy)

    model = get_model(model_type, num_classes, dataset_type=dataset_type, seed=seed)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    return {
        "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        ),
    }


def train_model(
    dataset_type="psa",
    model_type="xgboost",
    augment_strategy="original",
    seeds=DEFAULT_SEEDS,
):
    print(f"Training {model_type} on {dataset_type} over {len(seeds)} seeds {seeds}...")

    per_metric = {name: [] for name in METRIC_NAMES}
    for seed in seeds:
        metrics = run_single_seed(dataset_type, model_type, augment_strategy, seed)
        for name in METRIC_NAMES:
            per_metric[name].append(metrics[name])
        print(
            f"[{model_type} | seed {seed}] "
            + " ".join(f"{name}={metrics[name] * 100:.2f}%" for name in METRIC_NAMES)
        )

    summary = {
        name: (float(np.mean(values)), float(np.std(values)))
        for name, values in per_metric.items()
    }

    print(f"=== {model_type} on {dataset_type} ({len(seeds)} seeds) ===")
    for name in METRIC_NAMES:
        mean, std = summary[name]
        print(f"  {name}: {mean * 100:.2f}% +/- {std * 100:.2f}%")

    return summary


def print_summary_table(results):
    header = f"{'model':<20}" + "".join(f"{name:>18}" for name in METRIC_NAMES)
    print(header)
    for model_type, summary in results.items():
        row = f"{model_type:<20}"
        for name in METRIC_NAMES:
            mean, std = summary[name]
            row += f"{mean * 100:6.2f}+/-{std * 100:4.2f}%".rjust(18)
        print(row)


if __name__ == "__main__":
    model_types = [
        "xgboost",
        "random_forest",
        "refined",
        "deepinsight",
        "tac",
        "igtd",
        "lstm",
        "transformer",
        "svm",
        "logistic_regression",
    ]

    augment_strategy = None
    results = {}
    for model_type in model_types:
        results[model_type] = train_model("ddos", model_type, augment_strategy)

    print(f"\n================ Summary ({augment_strategy}) ================")
    print_summary_table(results)
