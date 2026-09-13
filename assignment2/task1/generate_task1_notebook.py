"""Generate the self-contained, step-by-step Task 1 notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
class_source = (HERE / "linear_regression_task1.py").read_text()
source_url = (
    "https://github.com/chaklam-silpasuwanchai/Python-for-Machine-Learning/"
    "blob/main/01%20-%20Supervised/01%20-%20Regression/"
    "03%20-%20Bias-Variance%20Tradeoff%20and%20Regularization.ipynb"
)

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Assignment 2 — Task 1\n\n"
        "This notebook modifies the instructor's [original regularization notebook]("
        + source_url
        + ") while keeping its main interface: the first column of `X` is the intercept, "
        "`method='mini'` selects mini-batch gradient descent, and `_coef()`/`_bias()` expose "
        "the learned parameters.\n\n"
        "The requested additions are:\n\n"
        "1. an R² function;\n"
        "2. selectable zero or Xavier initialization;\n"
        "3. optional momentum; and\n"
        "4. a coefficient-based feature-importance plot."
    ),
    nbf.v4.new_markdown_cell(
        "## Step 1 — Imports and the modified class\n\n"
        "The cell below is self-contained. Xavier initialization samples every weight from "
        "$[-1/\\sqrt{m}, 1/\\sqrt{m}]$, where $m$ is the number of input weights. Momentum "
        "uses the update supplied in the assignment: `theta = theta - step + momentum * prev_step`."
    ),
    nbf.v4.new_code_cell(class_source),
    nbf.v4.new_markdown_cell(
        "## Step 2 — Prepare scaled example data\n\n"
        "The diabetes features are standardized before training. This is essential when "
        "coefficient magnitude is used as feature importance. An intercept column of ones is "
        "then added, matching the original notebook. The target is also standardized to make "
        "gradient-descent behavior easy to compare."
    ),
    nbf.v4.new_code_cell(
        "from sklearn.datasets import load_diabetes\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "diabetes = load_diabetes()\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    diabetes.data, diabetes.target, test_size=0.30, random_state=42\n"
        ")\n\n"
        "x_scaler = StandardScaler()\n"
        "X_train = x_scaler.fit_transform(X_train)\n"
        "X_test = x_scaler.transform(X_test)\n"
        "X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])\n"
        "X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])\n\n"
        "y_scaler = StandardScaler()\n"
        "y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()\n"
        "y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()\n"
        "print('Training shape:', X_train.shape)\n"
        "print('First column is intercept:', np.all(X_train[:, 0] == 1))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 3 — Verify zero and Xavier initialization\n\n"
        "Zero initialization must contain only zeros. Xavier weights must stay inside the "
        "calculated range. A fixed random seed makes the demonstration reproducible."
    ),
    nbf.v4.new_code_cell(
        "zero_model = LinearRegression(NoRegularization(), initialization='zeros', verbose=False)\n"
        "zero_model._rng = np.random.default_rng(zero_model.random_state)\n"
        "zero_weights = zero_model._initialize_weights(X_train.shape[1])\n\n"
        "xavier_model = LinearRegression(NoRegularization(), initialization='xavier', verbose=False)\n"
        "xavier_model._rng = np.random.default_rng(xavier_model.random_state)\n"
        "xavier_weights = xavier_model._initialize_weights(X_train.shape[1])\n"
        "limit = 1 / np.sqrt(X_train.shape[1])\n\n"
        "print('Zero weights:', zero_weights)\n"
        "print(f'Xavier range: [{-limit:.4f}, {limit:.4f}]')\n"
        "print('Xavier weights:', np.round(xavier_weights, 4))\n"
        "assert np.all(zero_weights == 0)\n"
        "assert np.all((xavier_weights >= -limit) & (xavier_weights <= limit))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 4 — Train with optional momentum\n\n"
        "Both models use identical data, seed, learning rate, and Xavier initialization. The "
        "only changed setting is `use_momentum`."
    ),
    nbf.v4.new_code_cell(
        "without_momentum = RidgeRegression(\n"
        "    method='batch', lr=0.01, l=0.001, num_epochs=300,\n"
        "    initialization='xavier', use_momentum=False, random_state=42, verbose=False\n"
        ")\n"
        "with_momentum = RidgeRegression(\n"
        "    method='batch', lr=0.01, l=0.001, num_epochs=300,\n"
        "    initialization='xavier', use_momentum=True, momentum=0.9,\n"
        "    random_state=42, verbose=False\n"
        ")\n\n"
        "without_momentum.fit(X_train, y_train)\n"
        "with_momentum.fit(X_train, y_train)\n"
        "print('Mean CV MSE without momentum:', np.mean(without_momentum.kfold_scores))\n"
        "print('Mean CV MSE with momentum:   ', np.mean(with_momentum.kfold_scores))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 5 — Evaluate with MSE and the new R² function\n\n"
        "R² is implemented from scratch as $1 - SSE/SST$. The assertion confirms that it agrees "
        "with scikit-learn's reference implementation."
    ),
    nbf.v4.new_code_cell(
        "from sklearn.metrics import r2_score\n\n"
        "best_demo_model = without_momentum\n"
        "y_pred = best_demo_model.predict(X_test)\n"
        "test_mse = best_demo_model.mse(y_test, y_pred)\n"
        "test_r2 = best_demo_model.r2(y_test, y_pred)\n"
        "print(f'Test MSE: {test_mse:.6f}')\n"
        "print(f'Test R2:  {test_r2:.6f}')\n"
        "assert np.isclose(test_r2, r2_score(y_test, y_pred))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 6 — Plot coefficient-based feature importance\n\n"
        "Green bars are positive coefficients and red bars are negative coefficients. Larger "
        "absolute values indicate stronger linear influence only because all input features were "
        "standardized before fitting."
    ),
    nbf.v4.new_code_cell(
        "ax = best_demo_model.plot_feature_importance(diabetes.feature_names)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 7 — How to use the class in later tasks\n\n"
        "```python\n"
        "model = RidgeRegression(\n"
        "    method='mini', lr=0.01, l=0.01, bs=50,\n"
        "    initialization='xavier',\n"
        "    use_momentum=True, momentum=0.9\n"
        ")\n"
        "model.fit(X_train, y_train)\n"
        "predictions = model.predict(X_test)\n"
        "print(model.mse(y_test, predictions))\n"
        "print(model.r2(y_test, predictions))\n"
        "model.plot_feature_importance(feature_names)\n"
        "```\n\n"
        "Use `initialization='zeros'` to select zero initialization and "
        "`use_momentum=False` to disable momentum."
    ),
]

nbf.write(nb, HERE / "task1_modified_linear_regression.ipynb")
print(HERE / "task1_modified_linear_regression.ipynb")
