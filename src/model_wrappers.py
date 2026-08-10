"""Small model wrappers used by the training pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class ProbabilityAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """A fast soft-voting style ensemble with a sklearn-like interface."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        if not self.estimators:
            raise ValueError("estimators must contain at least one classifier")
        names = [name for name, _ in self.estimators]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("estimator names must be non-empty and unique")
        classes = np.unique(np.asarray(y))
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError("ProbabilityAveragingEnsemble requires binary labels 0 and 1")
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if len(weights) != len(self.estimators):
                raise ValueError("weights must have the same length as estimators")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("weights must be finite, non-negative, and have a positive sum")

        fitted_estimators = []
        for _, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            fitted_classes = getattr(fitted, "classes_", None)
            if fitted_classes is None or not np.array_equal(fitted_classes, classes):
                raise ValueError("all estimators must expose binary classes in the order [0, 1]")
            fitted_estimators.append(fitted)
        self.classes_ = classes
        self.fitted_estimators_ = fitted_estimators
        self.n_features_in_ = int(X.shape[1])
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_estimators_")
        probabilities = []
        expected_shape = (len(X), len(self.classes_))
        for model in self.fitted_estimators_:
            model_probabilities = np.asarray(model.predict_proba(X), dtype=float)
            if model_probabilities.shape != expected_shape:
                raise ValueError(
                    "an estimator returned probabilities with an unexpected shape: "
                    f"{model_probabilities.shape}, expected {expected_shape}"
                )
            probabilities.append(model_probabilities)
        probs = np.stack(probabilities)
        if not np.all(np.isfinite(probs)):
            raise ValueError("an estimator returned non-finite probabilities")
        if np.any((probs < 0) | (probs > 1)):
            raise ValueError("an estimator returned probabilities outside [0, 1]")
        if not np.allclose(probs.sum(axis=2), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("an estimator returned probabilities that do not sum to 1")
        if self.weights is None:
            return probs.mean(axis=0)
        weights = np.array(self.weights, dtype=float)
        weights = weights / weights.sum()
        return np.average(probs, axis=0, weights=weights)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
