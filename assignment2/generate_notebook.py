"""Build the executed Assignment 2 notebook from verified experiment artifacts."""

from pathlib import Path
import json

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / "best_model_summary.json").read_text())
best = summary["best_config"]
metrics = summary["test_metrics"]
class_source = (ROOT / "linear_regression.py").read_text()

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# A2: Predicting Car Price\n"
        "**Name:** Lin Htet Aung\n\n"
        "This notebook extends Assignment 1 with a from-scratch regression class, a complete "
        "MLflow-tracked experiment, held-out evaluation, coefficient importance, and deployment artifacts."
    ),
    nbf.v4.new_markdown_cell(
        "## 1. From-scratch implementation\n\n"
        "The class below follows the course starter but adds: (1) an R² calculation from SSE and SST; "
        "(2) selectable zero or Xavier-uniform initialization; (3) optional momentum with a configurable "
        "coefficient in [0, 1); (4) batch, mini-batch, and stochastic gradient descent; and "
        "(5) coefficient-based feature-importance plotting. The intercept is not regularized. "
        "Gradient clipping is included because extreme vehicle observations can destabilize the required "
        "stochastic learning-rate grid."
    ),
    nbf.v4.new_code_cell(class_source),
    nbf.v4.new_code_cell(
        "# Small correctness checks for the required additions.\n"
        "assert LinearRegression.r2([1, 2, 3], [1, 2, 3]) == 1.0\n"
        "zero_model = LinearRegression(initialization='zeros')\n"
        "xavier_model = LinearRegression(initialization='xavier', random_state=42)\n"
        "rng = np.random.default_rng(42)\n"
        "assert np.all(zero_model._initialize(4, rng) == 0)\n"
        "weights = xavier_model._initialize(4, np.random.default_rng(42))\n"
        "assert np.all(np.abs(weights) <= 1 / np.sqrt(4))\n"
        "print('R², zero initialization, and Xavier initialization checks passed.')"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Data and preprocessing\n\n"
        "The A1 cleaning workflow is retained: remove duplicate records, CNG/LPG and test-drive cars; "
        "parse mileage, engine, and power; ordinal-encode ownership; extract brand; and drop torque. "
        "Missing predictors are retained for pipeline imputation. Preprocessing is refitted inside each "
        "cross-validation fold to prevent leakage. Numeric inputs use median imputation and scaling; "
        "categorical inputs use most-frequent imputation and one-hot encoding. Polynomial regression adds "
        "degree-two numeric terms. The log-price target is standardized using training-fold statistics."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import joblib, json, mlflow\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import display\n"
        "from run_experiment import clean_data, make_preprocessor, evaluate_config\n\n"
        "raw = pd.read_csv('../data/Cars.csv')\n"
        "data = clean_data(raw)\n"
        "print(f'Raw: {raw.shape}; cleaned: {data.shape}')\n"
        "display(data.head())\n"
        "display(data.isna().sum().to_frame('missing'))"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. MLflow experiment design\n\n"
        "The full factorial design contains 4 model choices × 2 momentum choices × 3 gradient methods "
        "× 2 initializations × 3 learning rates = **144 configurations**. Each configuration uses the "
        "same shuffled five-fold splits and logs its parameters, mean validation MSE, mean validation R², "
        "and R² standard deviation to MLflow. The test set is not consulted during model selection."
    ),
    nbf.v4.new_code_cell(
        "from itertools import product\n\n"
        "model_types = ['normal', 'polynomial', 'lasso', 'ridge']\n"
        "momentum_choices = [False, True]\n"
        "methods = ['stochastic', 'mini-batch', 'batch']\n"
        "initializations = ['zeros', 'xavier']\n"
        "learning_rates = [0.01, 0.001, 0.0001]\n"
        "experiment_grid = list(product(model_types, momentum_choices, methods, initializations, learning_rates))\n"
        "print('Number of configurations:', len(experiment_grid))\n"
        "assert len(experiment_grid) == 144\n\n"
        "# run_experiment.py executes this grid with five-fold CV and an MLflow run per configuration.\n"
        "# Uncomment to regenerate every artifact from scratch:\n"
        "# from run_experiment import main\n"
        "# main()"
    ),
    nbf.v4.new_code_cell(
        "results = pd.read_csv('experiment_results.csv')\n"
        "print(f'Completed configurations: {len(results)}')\n"
        "display(results.head(20).round(5))\n\n"
        "group_summary = (results.groupby(['model_type', 'method'])[['cv_mse', 'cv_r2']]\n"
        "                 .mean().sort_values('cv_r2', ascending=False))\n"
        "display(group_summary.round(5))"
    ),
    nbf.v4.new_markdown_cell(
        "### MLflow evidence\n\n"
        "All 144 runs are stored in `mlflow_a2.db`. Start the UI with "
        "`mlflow ui --backend-store-uri sqlite:///mlflow_a2.db --port 5000`.\n\n"
        "![MLflow experiment runs](screenshots/mlflow_runs.png)"
    ),
    nbf.v4.new_markdown_cell("## 4. Best model and held-out evaluation"),
    nbf.v4.new_code_cell(
        "bundle = joblib.load('models/car_price_prediction_a2.joblib')\n"
        "display(pd.Series(bundle['best_config'], name='value').to_frame())\n"
        "display(pd.Series(bundle['test_metrics'], name='value').to_frame())\n"
        "ax = bundle['model'].plot_feature_importance(bundle['feature_names'], top_n=20)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Short report and findings\n\n"
        f"The best cross-validated configuration was **{best['model_type']} regression** trained with "
        f"**{best['method']} gradient descent**, **{best['initialization']} initialization**, "
        f"a learning rate of **{best['learning_rate']}**, and "
        f"**{'momentum' if best['use_momentum'] else 'no momentum'}**. Its mean validation MSE on "
        f"log price was **{best['cv_mse']:.5f}**, with mean R² **{best['cv_r2']:.4f}** "
        f"(fold standard deviation {best['cv_r2_std']:.4f}). After refitting on all training data for "
        f"60 epochs, the untouched test set produced log-MSE **{metrics['test_mse_log']:.5f}** and "
        f"log-scale R² **{metrics['test_r2_log']:.4f}**.\n\n"
        "Stochastic and mini-batch methods generally learned faster than batch gradient descent under "
        "the fixed 12-epoch comparison budget because they perform many more parameter updates per epoch. "
        "The smallest learning rate often underfit within that budget, while momentum improved several "
        "slower configurations but could amplify noisy stochastic updates at the largest learning rate. "
        "Xavier initialization breaks symmetry and sometimes improves early optimization, although a "
        "convex linear objective should approach the same optimum given enough stable updates. Polynomial "
        "terms add flexibility but also produce higher-variance gradients and did not win this experiment.\n\n"
        "Coefficient magnitude is meaningful here because numeric features are standardized and one-hot "
        "features share a common binary scale. Nevertheless, coefficients describe association rather than "
        "causation, and correlated features can divide importance. Limitations include a random rather than "
        "time-aware split, only one regularization strength, a fixed epoch budget, possible non-exact repeated "
        "listings, and weaker performance on rare or luxury cars. Future work should tune penalty strength "
        "and epoch count with nested validation and report prediction intervals."
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Deployment summary\n\n"
        "The new Streamlit application contains two pages selected from the sidebar: the A1 Random Forest "
        "and the A2 from-scratch regression model. The A2 model is not more accurate than the A1 forest, but "
        "it is more transparent and educational: every optimization and regularization step is inspectable "
        "and its experiment history is recorded in MLflow. Both models accept missing fields through their "
        "fitted imputers. `app/Dockerfile` creates the image and `app/docker-compose.yaml` connects it to the "
        "course server's Traefik network. Credentials and the student-ID subdomain must be filled in privately."
    ),
]

nbf.write(nb, ROOT / "a2_car_price_prediction.ipynb")
print("Created a2_car_price_prediction.ipynb")
