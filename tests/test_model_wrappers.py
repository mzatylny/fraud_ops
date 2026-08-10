import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.model_wrappers import ProbabilityAveragingEnsemble


class InvalidProbabilityEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.tile([1.1, -0.1], (len(X), 1))


def sample_data():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_probability_ensemble_produces_normalised_probabilities():
    X, y = sample_data()
    model = ProbabilityAveragingEnsemble(
        estimators=[("one", LogisticRegression()), ("two", LogisticRegression())],
        weights=[0.6, 0.4],
    ).fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "estimators,weights",
    [
        ([], None),
        ([('same', LogisticRegression()), ('same', LogisticRegression())], None),
        ([('one', LogisticRegression())], [0.5, 0.5]),
        ([('one', LogisticRegression())], [-1.0]),
    ],
)
def test_probability_ensemble_validates_configuration(estimators, weights):
    X, y = sample_data()
    with pytest.raises(ValueError):
        ProbabilityAveragingEnsemble(estimators=estimators, weights=weights).fit(X, y)


def test_probability_ensemble_rejects_invalid_estimator_output():
    X, y = sample_data()
    model = ProbabilityAveragingEnsemble(
        estimators=[("invalid", InvalidProbabilityEstimator())]
    ).fit(X, y)
    with pytest.raises(ValueError, match="outside"):
        model.predict_proba(X)
