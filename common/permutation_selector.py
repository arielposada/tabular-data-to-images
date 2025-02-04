import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import f1_score
import tabpfn
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.model_selection import StratifiedKFold
import torch


class TabPFNPermutationEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator=None,
        explainer_sample_size=50,
        scoring=None,
        n_repeats=5,
        random_state=42,
    ):
        """
        Parameters:
          base_estimator: Underlying classifier (defaults to TabPFNClassifier).
          explainer_sample_size: Number of training samples to use when computing permutation importances.
          scoring: Scoring function to evaluate feature importance (if None, estimator.score is used).
          n_repeats: Number of times to permute a feature.
          random_state: Random state for reproducibility.
        """
        self.base_estimator = (
            base_estimator if base_estimator is not None else tabpfn.TabPFNClassifier()
        )
        self.explainer_sample_size = explainer_sample_size
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y):
        # Fit the underlying TabPFNClassifier.
        self.base_estimator.fit(X, y)
        # Set the classes_ attribute so that RFECV and scoring functions can access it.
        if hasattr(self.base_estimator, "classes_"):
            self.classes_ = self.base_estimator.classes_
        else:
            self.classes_ = np.unique(y)

        with torch.no_grad():
            n_samples = X.shape[0]
            sample_size = min(self.explainer_sample_size, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Compute permutation importances on the selected subset.
            result = permutation_importance(
                self.base_estimator,
                X_sample,
                y_sample,
                scoring=self.scoring,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            # result.importances_mean is an array of shape (n_features,) with the average decrease in score.
            self.feature_importances_ = result.importances_mean
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average="weighted")


class PermutationRFECVSelector(BaseEstimator, TransformerMixin):
    """
    Custom feature selector that first applies a coarse univariate selection (using SelectKBest)
    to reduce the feature space to a manageable size, then uses RFECV with the
    TabPFNPermutationEstimator (which computes permutation importances) to further select features.
    """

    def __init__(
        self,
        base_estimator=None,
        coarse_k=200,
        step=0.1,
        cv=None,
        scoring="f1_weighted",
        explainer_sample_size=50,
        n_repeats=5,
        random_state=42,
    ):
        """
        Parameters:
          base_estimator: The base classifier to be wrapped in the RFECV stage (e.g., a TabPFNClassifier).
          coarse_k: Number of features to keep after coarse selection.
          step: Fraction (or number) of features to remove at each RFECV iteration.
          cv: Cross-validation splitter (default is StratifiedKFold with 3 splits).
          scoring: Scoring metric for both RFECV and permutation_importance.
          explainer_sample_size: Number of samples used for computing permutation importances.
          n_repeats: Number of times to permute each feature.
          random_state: Random state for reproducibility.
        """
        self.base_estimator = base_estimator
        self.coarse_k = coarse_k
        self.step = step
        self.cv = (
            cv
            if cv is not None
            else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        )
        self.scoring = scoring
        self.explainer_sample_size = explainer_sample_size
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y):
        # Stage 1: Coarse Selection using SelectKBest.
        self.coarse_selector_ = SelectKBest(score_func=f_classif, k=self.coarse_k)
        X_coarse = self.coarse_selector_.fit_transform(X, y)

        # Stage 2: Fine Selection via RFECV using the permutation importance estimator.
        self.rfecv_ = RFECV(
            estimator=TabPFNPermutationEstimator(
                base_estimator=self.base_estimator,
                explainer_sample_size=self.explainer_sample_size,
                scoring=self.scoring,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            ),
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            verbose=2,
        )
        self.rfecv_.fit(X_coarse, y)
        return self

    def transform(self, X):
        # Apply the coarse selector.
        X_coarse = self.coarse_selector_.transform(X)
        # Then, apply the RFECV feature mask.
        X_fine = self.rfecv_.transform(X_coarse)
        return X_fine

    def get_support(self, indices=False):
        # Combine the support masks from the coarse and RFECV stages.
        coarse_mask = self.coarse_selector_.get_support()
        fine_mask = self.rfecv_.get_support()
        full_mask = np.zeros_like(coarse_mask, dtype=bool)
        coarse_indices = np.where(coarse_mask)[0]
        full_mask[coarse_indices] = fine_mask
        if indices:
            return np.where(full_mask)[0]
        return full_mask
