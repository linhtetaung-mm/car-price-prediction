"""Four-class softmax regression. All optimization is implemented using NumPy."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from a3.metrics import ClassificationMetrics


class LogisticRegression(ClassifierMixin, ClassificationMetrics, BaseEstimator):
    """Minimize mean cross entropy + l2 * sum(W**2); do not penalize bias.

    Softmax generalizes the binary sigmoid to four mutually exclusive classes.
    l2=0 disables ridge. The brief uses summed cross entropy; dividing its entire
    objective by m gives our convention with l2 = lambda_in_brief / m.
    """
    def __init__(self, lr=0.1, num_epochs=600, l2=0.0, tol=1e-8):
        self.lr = lr
        self.num_epochs = num_epochs
        self.l2 = l2
        self.tol = tol

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be a non-empty finite two-dimensional numeric array")
        return X

    @staticmethod
    def _log_softmax(logits):
        # Subtract the largest logit before exponentiating to prevent overflow.
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    def _loss_gradient(self, X, y, weights, bias):
        log_probs = self._log_softmax(X @ weights + bias)
        loss = -log_probs[np.arange(len(y)), y].mean() + self.l2 * np.sum(weights ** 2)
        error = np.exp(log_probs)
        error[np.arange(len(y)), y] -= 1
        error /= len(y)
        # d/dW [lambda * W^2] = 2 * lambda * W. Bias is unpenalized.
        return float(loss), X.T @ error + 2 * self.l2 * weights, error.sum(axis=0)

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.shape != (len(X),) or not np.isin(y, np.arange(4)).all():
            raise ValueError("y must have one integer class in {0,1,2,3} per row")
        if not np.isfinite(self.lr) or self.lr <= 0 or not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("lr must be positive and l2 non-negative, both finite")
        if not isinstance(self.num_epochs, (int, np.integer)) or self.num_epochs < 1:
            raise ValueError("num_epochs must be a positive integer")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and non-negative")
        y = y.astype(int)
        self.classes_ = np.arange(4)
        self.n_features_in_ = X.shape[1]
        self.weights_ = np.zeros((self.n_features_in_, 4))
        self.bias_ = np.zeros(4)
        self.loss_history_ = []
        for epoch in range(self.num_epochs):
            loss, dw, db = self._loss_gradient(X, y, self.weights_, self.bias_)
            if not np.isfinite(loss):
                raise FloatingPointError("Training diverged; reduce the learning rate")
            self.loss_history_.append(loss)
            if epoch and abs(self.loss_history_[-2] - loss) <= self.tol:
                break
            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db
        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "weights_")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features; received {X.shape[1]}")
        return np.exp(self._log_softmax(X @ self.weights_ + self.bias_))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
