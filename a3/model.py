"""Tasks 1 and 2: multinomial logistic regression with optional ridge penalty."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from a3.metrics import ClassificationMetrics


class LogisticRegression(ClassifierMixin, ClassificationMetrics, BaseEstimator):
    """Predict one of the four price classes using batch gradient descent.

    lr : float
        Learning rate for the weight and bias updates.
    num_epochs : int
        Maximum number of passes through the training data.
    l2 : float
        Ridge strength. Use 0 for no regularization.
    tol : float
        Stop when the change in training loss is this small.

    Loss = mean cross entropy + l2 * sum(weights**2).
    The bias is stored separately, so it is not included in the penalty.
    The brief uses a summed loss; its lambda would be m * l2 here.
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
        """Calculate the loss and the gradients for one batch."""
        number_of_samples = len(y)
        log_probs = self._log_softmax(X @ weights + bias)

        # Select the probability of the actual class for each training row.
        actual_log_probs = log_probs[np.arange(number_of_samples), y]
        cross_entropy = -actual_log_probs.mean()
        ridge_penalty = self.l2 * np.sum(weights ** 2)
        loss = cross_entropy + ridge_penalty

        # Subtracting 1 at the actual class is the same as probabilities - one_hot_y.
        error = np.exp(log_probs)
        error[np.arange(number_of_samples), y] -= 1
        error /= number_of_samples

        weight_gradient = X.T @ error + 2 * self.l2 * weights
        bias_gradient = error.sum(axis=0)
        return float(loss), weight_gradient, bias_gradient

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

        # Each epoch uses all the training rows for one gradient update.
        for epoch in range(self.num_epochs):
            loss, weight_gradient, bias_gradient = self._loss_gradient(
                X, y, self.weights_, self.bias_
            )
            if not np.isfinite(loss):
                raise FloatingPointError("Training diverged; reduce the learning rate")
            self.loss_history_.append(loss)
            if epoch and abs(self.loss_history_[-2] - loss) <= self.tol:
                break
            self.weights_ -= self.lr * weight_gradient
            self.bias_ -= self.lr * bias_gradient
        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "weights_")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features; received {X.shape[1]}")
        return np.exp(self._log_softmax(X @ self.weights_ + self.bias_))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
