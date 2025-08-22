import numpy as np
from sklearn.base import BaseEstimator, clone
from .stepwise_zero_inflated import StepwiseZeroInflated

class RandomFeatureSelector(BaseEstimator):
    def __init__(self, estimator: StepwiseZeroInflated, n_estimators=10, max_features=0.8, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.models_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.models_ = []
        self.feature_subsets_ = []

        self.best_estimator_ = None
        self.best_features_ = None

        for _ in range(self.n_estimators):
            n_selected = int(np.ceil(self.max_features * n_features))
            feature_idx = rng.choice(n_features, n_selected, replace=False)
            X_subset = X[:, feature_idx] if isinstance(X, np.ndarray) else X.iloc[:, feature_idx]

            model: StepwiseZeroInflated = clone(self.estimator)
            model.fit(X_subset, y)

            self.models_.append(model)
            self.feature_subsets_.append(feature_idx)

        return self

    @property
    def models(self):
      return self.models_

    def predict(self, X):
      pass

    def predict_proba(self, X):
      pass