import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from common.dataset import load_dataset, split_data
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from tabpfn_extensions import interpretability, TabPFNClassifier


# Unable to run - Require more RAM in my local machine
def select_features(dataset_type="ddos"):
    print(f"Selecting features for TabPFN model on {dataset_type} dataset")

    X, y = load_dataset(dataset_type)

    clf = TabPFNClassifier(ignore_pretraining_limits=True)

    # Feature selection
    sfs = interpretability.feature_selection.feature_selection(
        estimator=clf,
        X=X,
        y=y,
        n_features_to_select=50,
    )

    print(f"Selected features: {sfs}")


if __name__ == "__main__":
    select_features("ddos")
