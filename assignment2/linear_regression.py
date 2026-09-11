"""From-scratch linear regression used by Assignment 2."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """Linear regression trained with gradient descent.

    Supports ordinary, L1 (lasso), and L2 (ridge) objectives; batch,
    mini-batch, and stochastic updates; zero or Xavier initialization; and
    optional classical momentum. The intercept is never regularized.
    """

    def __init__(
        self,
        regularization="normal",
        learning_rate=0.01,
        method="batch",
        num_epochs=12,
        batch_size=128,
        initialization="zeros",
        use_momentum=False,
        momentum=0.9,
        regularization_strength=0.001,
        gradient_clip=10.0,
        random_state=42,
    ):
        if regularization not in {"normal", "lasso", "ridge"}:
            raise ValueError("regularization must be normal, lasso, or ridge")
        if method not in {"batch", "mini-batch", "stochastic"}:
            raise ValueError("method must be batch, mini-batch, or stochastic")
        if initialization not in {"zeros", "xavier"}:
            raise ValueError("initialization must be zeros or xavier")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

        self.regularization = regularization
        self.learning_rate = learning_rate
        self.method = method
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.initialization = initialization
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.regularization_strength = regularization_strength
        self.gradient_clip = gradient_clip
        self.random_state = random_state

    @staticmethod
    def mse(y_true, y_pred):
        """Return mean squared error."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2(y_true, y_pred):
        """Compute R-squared = 1 - SSE/SST from scratch."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - sse / sst) if sst > 0 else 0.0

    def _initialize(self, n_inputs, rng):
        if self.initialization == "zeros":
            return np.zeros(n_inputs + 1)
        # Xavier uniform initialization, using the number of input features.
        limit = 1.0 / np.sqrt(n_inputs)
        return rng.uniform(-limit, limit, size=n_inputs + 1)

    def _regularization_gradient(self):
        gradient = np.zeros_like(self.theta_)
        if self.regularization == "lasso":
            gradient[1:] = self.regularization_strength * np.sign(self.theta_[1:])
        elif self.regularization == "ridge":
            gradient[1:] = 2 * self.regularization_strength * self.theta_[1:]
        return gradient

    def _batch_indices(self, n_samples, rng):
        order = rng.permutation(n_samples)
        if self.method == "batch":
            return [order]
        if self.method == "stochastic":
            return [order[i:i + 1] for i in range(n_samples)]
        return [order[i:i + self.batch_size] for i in range(0, n_samples, self.batch_size)]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        rng = np.random.default_rng(self.random_state)
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        self.theta_ = self._initialize(X.shape[1], rng)
        velocity = np.zeros_like(self.theta_)
        self.loss_history_ = []

        for _ in range(self.num_epochs):
            for indices in self._batch_indices(X.shape[0], rng):
                xb, yb = X_bias[indices], y[indices]
                error = xb @ self.theta_ - yb
                gradient = (xb.T @ error) / len(indices)
                gradient += self._regularization_gradient()
                # A few extreme cars create very large stochastic gradients.
                # Norm clipping keeps the required learning-rate grid numerically stable.
                gradient_norm = np.linalg.norm(gradient)
                if self.gradient_clip and gradient_norm > self.gradient_clip:
                    gradient *= self.gradient_clip / gradient_norm
                if self.use_momentum:
                    velocity = self.momentum * velocity + self.learning_rate * gradient
                    self.theta_ -= velocity
                else:
                    self.theta_ -= self.learning_rate * gradient
            self.loss_history_.append(self.mse(y, X_bias @ self.theta_))

        self.intercept_ = float(self.theta_[0])
        self.coef_ = self.theta_[1:].copy()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def plot_feature_importance(self, feature_names, top_n=20, ax=None):
        """Plot largest absolute coefficients for standardized input features."""
        names = np.asarray(feature_names, dtype=object)
        if len(names) != len(self.coef_):
            raise ValueError("feature_names length must match the fitted coefficients")
        order = np.argsort(np.abs(self.coef_))[-top_n:]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 7))
        colors = np.where(self.coef_[order] >= 0, "#0f766e", "#dc2626")
        ax.barh(names[order], self.coef_[order], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (standardized feature scale)")
        ax.set_title(f"Top {min(top_n, len(names))} coefficient importances")
        return ax
