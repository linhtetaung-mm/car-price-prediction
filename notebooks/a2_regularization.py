"""
Task 1: Implementation
Based on: https://github.com/chaklam-silpasuwanchai/Python-for-Machine-Learning/blob/main/01%20-%20Supervised/01%20-%20Regression/03%20-%20Bias-Variance%20Tradeoff%20and%20Regularization.ipynb
a. Add r-squared function.
b. Modify the class such that it allows the user to choose between zeros initialization or xavier.
c. Implement "momentum" so that users can choose whether to use momentum or not.
d. Implement a function inside the class that can plot the feature importance based on coefficients.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

class LinearRegression:
    """
    Parameters added for Task 1
    initialization : {"zeros", "xavier"}
        Select zero weights or Xavier-uniform random weights.
    use_momentum : bool
        Whether the update should include the previous gradient step.
    momentum : float
        Momentum coefficient. When enabled, it must be strictly between 0 and 1.
    random_state : int
        Makes Xavier initialization and shuffling reproducible.
    """

    def __init__(
        self,
        regularization,
        lr=0.001,
        method="batch",
        num_epochs=500,
        batch_size=50,
        cv=None,
        initialization="zeros",
        use_momentum=False,
        momentum=0.9,
        random_state=42,
        tol=1e-8, # tolerance: training stops when the validation MSE changes by no more than 0.00000001
        verbose=True,
    ):
        if method not in {"batch", "mini", "stochastic"}:
            raise ValueError("method must be 'batch', 'mini', or 'stochastic'")
        if initialization not in {"zeros", "xavier"}:
            raise ValueError("initialization must be 'zeros' or 'xavier'")
        if use_momentum and not 0 < momentum < 1:
            raise ValueError("momentum must be strictly between 0 and 1")

        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method
        self.cv = cv or KFold(n_splits=5, shuffle=True, random_state=random_state)
        self.regularization = regularization
        self.initialization = initialization
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

    def mse(self, ytrue, ypred):
        # Return mean squared error.
        ytrue = np.asarray(ytrue)
        ypred = np.asarray(ypred)
        return float(np.mean((ypred - ytrue) ** 2))

    def r2(self, ytrue, ypred):
        # Return R-squared, computed as 1 - SSE / SST.
        ytrue = np.asarray(ytrue)
        ypred = np.asarray(ypred)
        sse = np.sum((ytrue - ypred) ** 2)
        sst = np.sum((ytrue - np.mean(ytrue)) ** 2)
        if np.isclose(sst, 0):
            return 0.0
        return float(1 - sse / sst)

    def _initialize_weights(self, number_of_weights):
        """Initialize theta using the option selected by the user."""
        if self.initialization == "zeros":
            return np.zeros(number_of_weights)

        # Xavier-uniform: draw each weight from [-1/sqrt(m), 1/sqrt(m)].
        # Here m is the number of model inputs/weights, including the bias.
        limit = 1.0 / np.sqrt(number_of_weights)
        return self._rng.uniform(-limit, limit, size=number_of_weights)

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self._rng = np.random.default_rng(self.random_state)
        self.kfold_scores = []
        self.kfold_r2_scores = []

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx].copy()
            y_cross_train = y_train[train_idx].copy()
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            self.theta = self._initialize_weights(X_cross_train.shape[1])
            self.prev_step = np.zeros_like(self.theta)
            val_loss_old = np.inf

            for _ in range(self.num_epochs):
                permutation = self._rng.permutation(X_cross_train.shape[0])
                X_cross_train = X_cross_train[permutation]
                y_cross_train = y_cross_train[permutation]

                if self.method == "mini":
                    for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                        X_batch = X_cross_train[batch_idx : batch_idx + self.batch_size]
                        y_batch = y_cross_train[batch_idx : batch_idx + self.batch_size]
                        self._train(X_batch, y_batch)
                elif self.method == "stochastic":
                    for row_idx in range(X_cross_train.shape[0]):
                        self._train(
                            X_cross_train[row_idx : row_idx + 1],
                            y_cross_train[row_idx : row_idx + 1],
                        )
                else:
                    self._train(X_cross_train, y_cross_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)
                if abs(val_loss_old - val_loss_new) <= self.tol:
                    break
                val_loss_old = val_loss_new

            val_r2 = self.r2(y_cross_val, self.predict(X_cross_val))
            self.kfold_scores.append(val_loss_new)
            self.kfold_r2_scores.append(val_r2)
            if self.verbose:
                print(f"Fold {fold + 1}: MSE={val_loss_new:.6f}, R2={val_r2:.6f}")

        return self

    def _train(self, X, y):
        yhat = self.predict(X)
        m = X.shape[0]
        regularization_gradient = self.regularization.derivation(self.theta).copy()
        # The first element is the bias/intercept and should not be penalized.
        regularization_gradient[0] = 0
        grad = (1 / m) * X.T @ (yhat - y) + regularization_gradient

        # Momentum follows the pseudocode supplied in the assignment.
        step = self.lr * grad
        if self.use_momentum:
            self.theta = self.theta - step + self.momentum * self.prev_step
        else:
            self.theta = self.theta - step
        self.prev_step = step
        return self.mse(y, yhat)

    def predict(self, X):
        return np.asarray(X) @ self.theta

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]

    def plot_feature_importance(self, feature_names, top_n=None, ax=None):
        """
        Plot signed coefficients, ordered by their absolute magnitudes.

        Coefficient magnitudes are comparable only when the input features use
        the same units or have been scaled before fitting.
        """
        if not hasattr(self, "theta"):
            raise ValueError("Fit the model before plotting feature importance")

        feature_names = np.asarray(feature_names, dtype=object)
        coefficients = self._coef()
        if len(feature_names) != len(coefficients):
            raise ValueError("feature_names must match the number of non-bias coefficients")

        count = len(coefficients) if top_n is None else min(top_n, len(coefficients))
        order = np.argsort(np.abs(coefficients))[-count:]
        colors = np.where(coefficients[order] >= 0, "#15803d", "#dc2626")

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feature_names[order], coefficients[order], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient value (features must be scaled)")
        ax.set_title("Feature importance based on coefficient magnitude")
        return ax


class NoRegularization:
    # No penalty; useful when ordinary linear regression is required.

    def __call__(self, theta):
        return 0.0

    def derivation(self, theta):
        return np.zeros_like(theta)


class Lasso:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class Ridge:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class Elastic:
    def __init__(self, l, l_ratio):
        self.l = l
        self.l_ratio = l_ratio

    def __call__(self, theta):
        l1 = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2 = (1 - self.l_ratio) * self.l * np.sum(np.square(theta))
        return l1 + l2

    def derivation(self, theta):
        l1 = self.l * self.l_ratio * np.sign(theta)
        l2 = 2 * self.l * (1 - self.l_ratio) * theta
        return l1 + l2

class LassoRegression(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0.01, **kwargs):
        super().__init__(Lasso(l), lr=lr, method=method, **kwargs)


class RidgeRegression(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0.01, **kwargs):
        super().__init__(Ridge(l), lr=lr, method=method, **kwargs)


class ElasticRegression(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0.01, l_ratio=0.5, **kwargs):
        super().__init__(Elastic(l, l_ratio), lr=lr, method=method, **kwargs)

