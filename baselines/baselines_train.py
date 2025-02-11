import sys
from pathlib import Path
import os

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from joblib import dump
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import StandardScaler

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from common.dataset import load_dataset, split_data
from common.train_utils import get_feature_selector


def get_classifier(classifier="xgboost"):
    if classifier == "xgboost":
        return XGBClassifier()
    elif classifier == "logistic_regression":
        return LogisticRegression()
    elif classifier == "tabpfn":
        return TabPFNClassifier(device="cuda")
    elif classifier == "auto_tabpfn":
        return AutoTabPFNClassifier(max_time=1800, device="cuda")
    elif classifier == "auto_tabpfn3600":
        return AutoTabPFNClassifier(max_time=3600, device="cuda")
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")


def save_results(base_name, pipeline, results, n_features, plt):
    if os.path.isdir(f"./models/{base_name}") == False:
        os.makedirs(f"./models/{base_name}")

    with open(f"./models/{base_name}/{base_name}_results.txt", "w") as f:
        res = f"Results for {base_name} model\n"
        res += str(results)
        res += "\n"
        res += f"Number of selected features: {n_features}"
        res += "\n results: "

        values = ",".join([str(x) for x in results.values()])
        res += f"\n{values}"
        f.write(str(res))
    plt.savefig(f"./models/{base_name}/{base_name}_confusion_matrix.png")
    plt.close()

def save_model(base_name, pipeline):
    if os.path.isdir(f"./models/{base_name}") == False:
        os.makedirs(f"./models/{base_name}")
    dump(pipeline, f"./models/{base_name}/{base_name}_pipeline.joblib")

def train_model(dataset_type="ddos", classifier="xgboost", feat_selector="rfecv"):
    print(f"Running {feat_selector}-based feature selection using {classifier}.")
    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False
    )

    if "tabpfn" in classifier and "none" in feat_selector:
        print("TabPFN requires feature selection. Skipping training.")
        return

    model = get_classifier(classifier)
    if (
        "tabpfn" in classifier or "deepinsight" in classifier
    ) and feat_selector == "rfecv":
        featselect_classifier = XGBClassifier()
    else:
        featselect_classifier = model
    feature_selector = get_feature_selector(featselect_classifier, name=feat_selector)

    pipelinelst = []
    if feat_selector == "pca":
        scaler = StandardScaler()
        pipelinelst.append(("scaler", scaler))
    pipeline = Pipeline(
        [*pipelinelst, ("selector", feature_selector), ("classifier", model)]
    )

    print("Training ...")
    pipeline.fit(X_train, y_train)
    selector = pipeline.named_steps["selector"]

    n_features = X_train.shape[-1]
    if feat_selector == "rfecv":
        n_features = selector.n_features_
    elif feat_selector == "pca":
        n_features = selector.n_components_
    print(f"Number of selected features: {n_features}")

    base_name = f"{classifier}_{feat_selector}"
    save_model(base_name, pipeline)

    print(f"Evaluating {classifier} model on test set...")
    y_test_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    print(f"\n=== Test Set Performance === {classifier} + {feat_selector}")
    print(f"F1-score:      {test_f1 * 100:.2f}%")
    print(f"Accuracy:      {test_accuracy * 100:.2f}%")
    print(f"Precision:     {test_precision * 100:.2f}%")
    print(f"Recall:        {test_recall * 100:.2f}%")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, display_labels=label_encoder.classes_
    )
    plt.title("Confusion Matrix on Test Set")
    results = {
        "f1_score": round(test_f1, 4),
        "accuracy": round(test_accuracy, 4),
        "precision": round(test_precision, 4),
        "recall": round(test_recall, 4),
    }
    save_results(base_name, pipeline, results, n_features, plt)

if __name__ == "__main__":
    feature_selector = ["mutual_info", "rfecv", "none", "pca"]
    model = ["auto_tabpfn3600"]  # "auto_tabpfn"
    for fs in feature_selector:
        for m in model:
            print(f"RUNNING MODEL {m} AND {fs} FEATURE EXTRACT")
            train_model("ddos", m, fs)
