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

        self.classes_ = classes
        self.fitted_estimators_ = []
        for _, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.fitted_estimators_.append(fitted)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_estimators_")
        probs = np.array([model.predict_proba(X) for model in self.fitted_estimators_])
        if not np.all(np.isfinite(probs)):
            raise ValueError("an estimator returned non-finite probabilities")
        if self.weights is None:
            return probs.mean(axis=0)
        weights = np.array(self.weights, dtype=float)
        weights = weights / weights.sum()
        return np.average(probs, axis=0, weights=weights)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
