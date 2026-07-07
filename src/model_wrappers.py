"""Small model wrappers used by the training pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class ProbabilityAveragingEnsemble(BaseEstimator, ClassifierMixin):
    """A fast soft-voting style ensemble with a sklearn-like interface."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators or []
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fitted_estimators_ = []
        for _, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.fitted_estimators_.append(fitted)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fitted_estimators_")
        probs = np.array([model.predict_proba(X) for model in self.fitted_estimators_])
        if self.weights is None:
            return probs.mean(axis=0)
        weights = np.array(self.weights, dtype=float)
        weights = weights / weights.sum()
        return np.average(probs, axis=0, weights=weights)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
