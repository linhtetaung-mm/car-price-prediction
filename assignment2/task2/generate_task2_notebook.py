"""Generate the self-contained Task 2 report notebook from verified artifacts."""

from pathlib import Path
import base64
import json

import nbformat as nbf


HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "task2_best_model_summary.json").read_text())
best = summary["best_config"]
metrics = summary["test_metrics"]


def image_markdown(title, filename):
    """Return a markdown cell with an embedded screenshot attachment."""
    image_path = HERE / "screenshots" / filename
    cell = nbf.v4.new_markdown_cell(f"### {title}\n\n![{title}](attachment:{filename})")
    cell["attachments"] = {
        filename: {"image/png": base64.b64encode(image_path.read_bytes()).decode("ascii")}
    }
    return cell


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Assignment 2 — Task 2: Car Price Model Comparison\n\n"
        "**Name:** Lin Htet Aung\n\n"
        "This notebook extends Assignment 1's cleaned car data, train/test split, log-price target, "
        "and missing-value handling. It uses the from-scratch regression implementation developed "
        "for Task 1. The untouched test set is used only after cross-validation selects the best model."
    ),
    nbf.v4.new_markdown_cell(
        "## Step 1 — Load and clean the Assignment 1 data\n\n"
        "The attached A1 sequence is reproduced exactly: remove CNG/LPG; ordinal-encode ownership; "
        "parse mileage, engine, and power; extract brand; remove torque and test-drive cars; delete rows "
        "missing mileage, engine, power, or seats; and finally remove duplicates after transformation."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import os\n"
        "os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'\n"
        "import joblib\n"
        "import mlflow\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import display\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "from run_task2_mlflow import (\n"
        "    clean_data, build_grid, NUMERIC, CATEGORICAL, RANDOM_STATE\n"
        ")\n\n"
        "raw = pd.read_csv('../../data/Cars.csv')\n"
        "data = clean_data(raw)\n"
        "X = data[NUMERIC + CATEGORICAL]\n"
        "y = np.log(data['selling_price'].astype(float))\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.20, random_state=RANDOM_STATE\n"
        ")\n"
        "print('Raw shape:', raw.shape)\n"
        "print('Cleaned shape:', data.shape)\n"
        "print('Training rows:', len(X_train), '| Test rows:', len(X_test))\n"
        "display(data.head())\n"
        "display(data[NUMERIC + CATEGORICAL].isna().sum().to_frame('missing'))"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 2 — Define the 144 configurations\n\n"
        "The full factorial comparison is 4 model types × 2 momentum choices × 3 gradient methods "
        "× 2 initialization methods × 3 learning rates = **144 configurations**. Each configuration "
        "uses the same shuffled five-fold splits. Preprocessing is fitted independently inside each "
        "training fold to prevent information leakage."
    ),
    nbf.v4.new_code_cell(
        "configurations = build_grid()\n"
        "print('Configurations:', len(configurations))\n"
        "display(pd.DataFrame(configurations).head(12))\n"
        "assert len(configurations) == 4 * 2 * 3 * 2 * 3 == 144"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 3 — Run and verify the MLflow experiment\n\n"
        "`run_task2_mlflow.py` opens one MLflow run **before** evaluating each configuration. It logs "
        "all parameters, every fold's MSE/R², the mean cross-validation metrics, and their standard "
        "deviations. The code below uses the completed local database. Set `RUN_EXPERIMENT=True` only "
        "when all 144 configurations need to be regenerated."
    ),
    nbf.v4.new_code_cell(
        "RUN_EXPERIMENT = False\n"
        "if RUN_EXPERIMENT:\n"
        "    from run_task2_mlflow import main\n"
        "    main()\n"
        "else:\n"
        "    print('Using the completed MLflow experiment artifacts.')\n\n"
        "tracking_database = Path('mlflow_task2_a1.db').resolve()\n"
        "mlflow.set_tracking_uri(f'sqlite:///{tracking_database}')\n"
        "experiment = mlflow.get_experiment_by_name('A2 Task 2 - Based on A1 Car Price Notebook')\n"
        "logged_runs = mlflow.search_runs([experiment.experiment_id], max_results=200)\n"
        "print('MLflow experiment:', experiment.name)\n"
        "print('Finished MLflow runs:', len(logged_runs))\n"
        "assert len(logged_runs) == 144\n"
        "assert set(logged_runs['status']) == {'FINISHED'}"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 4 — Final comparison tables\n\n"
        "The first table shows the strongest configurations. The factor-level table summarizes the "
        "average effect of each choice. The final table contains all 144 comparisons and is also saved "
        "as `task2_experiment_results.csv`. Higher R² and lower MSE are better."
    ),
    nbf.v4.new_code_cell(
        "results = pd.read_csv('task2_experiment_results.csv')\n"
        "assert len(results) == 144\n"
        "assert not results.duplicated(\n"
        "    ['model_type', 'use_momentum', 'method', 'initialization', 'learning_rate']\n"
        ").any()\n"
        "print('Top 15 configurations:')\n"
        "display(results.head(15).round(6))\n\n"
        "factor_tables = []\n"
        "for factor in ['model_type', 'use_momentum', 'method', 'initialization', 'learning_rate']:\n"
        "    table = results.groupby(factor)[['cv_mse', 'cv_r2']].mean().reset_index()\n"
        "    table.insert(0, 'factor', factor)\n"
        "    table = table.rename(columns={factor: 'choice'})\n"
        "    factor_tables.append(table)\n"
        "factor_summary = pd.concat(factor_tables, ignore_index=True)\n"
        "print('Average comparison by experimental factor:')\n"
        "display(factor_summary.round(6))"
    ),
    nbf.v4.new_code_cell(
        "with pd.option_context('display.max_rows', 144, 'display.max_columns', None):\n"
        "    display(results.round(6))"
    ),
    nbf.v4.new_markdown_cell("## Step 5 — Captured MLflow evidence"),
    image_markdown("MLflow table of completed configuration runs", "mlflow_all_runs.png"),
    image_markdown("MLflow parameters and metrics for the best run", "mlflow_best_run.png"),
    nbf.v4.new_markdown_cell(
        "## Step 6 — Evaluate the selected model on the untouched test set\n\n"
        "The configuration with the highest mean validation R² (using MSE as the tie-breaker) is "
        "refitted on all training rows for 60 epochs. The primary metrics remain on log selling price, "
        "matching Assignment 1. Rupee-scale metrics are also included for interpretation."
    ),
    nbf.v4.new_code_cell(
        "bundle = joblib.load('models/best_task2_model.joblib')\n"
        "print('Best configuration:')\n"
        "display(pd.Series(bundle['best_config'], name='value').to_frame())\n"
        "print('Held-out test metrics:')\n"
        "display(pd.Series(bundle['test_metrics'], name='value').to_frame())"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 7 — Plot feature importance\n\n"
        "The Task 1 function ranks features by absolute coefficient magnitude while retaining each "
        "coefficient's sign. Numeric inputs are standardized, so their magnitudes are comparable. "
        "One-hot variables share a binary scale, but correlated variables may divide importance."
    ),
    nbf.v4.new_code_cell(
        "ax = bundle['model'].plot_feature_importance(\n"
        "    bundle['feature_names'], top_n=20\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Step 8 — Short report and findings\n\n"
        f"The best individual configuration was **{best['model_type']} regression** using "
        f"**{best['method']} gradient descent**, **{best['initialization']} initialization**, "
        f"a learning rate of **{best['learning_rate']}**, and "
        f"**{'momentum' if best['use_momentum'] else 'no momentum'}**. Its five-fold mean log-MSE "
        f"was **{best['cv_mse']:.5f}** and mean R² was **{best['cv_r2']:.4f}**. The fold-to-fold "
        f"R² standard deviation was **{best['cv_r2_std']:.4f}**, indicating reasonably consistent "
        "validation performance.\n\n"
        f"After refitting, the untouched test set produced log-MSE **{metrics['test_mse_log']:.5f}** "
        f"and log-scale R² **{metrics['test_r2_log']:.4f}**. On the original rupee scale, MSE was "
        f"**{metrics['test_mse_price']:,.0f}**, RMSE was **₹{metrics['test_rmse_price']:,.0f}**, and "
        f"R² was **{metrics['test_r2_price']:.4f}**.\n\n"
        "Lasso had the strongest average R² among the four model families, but polynomial regression "
        "produced the best individual configuration. For non-polynomial models, stochastic "
        "gradient descent was strongest on average because it made many updates within the fixed 12-epoch "
        "budget. Polynomial models were sensitive to the largest stochastic learning rate; this instability "
        "reduced the overall stochastic average, making mini-batch the most robust method across all 144 runs. "
        "The learning rate 0.001 was strongest on average, zero initialization outperformed Xavier within the "
        "short budget, and momentum helped the winning configuration but did not improve every combination.\n\n"
        "These results should not be interpreted as universal rankings. Batch gradient descent receives far "
        "fewer updates per epoch than stochastic learning, so a fixed epoch budget favors the latter. Only one "
        "regularization strength and polynomial degree were tested. Coefficients describe association rather "
        "than causation, and predictions may be less reliable for rare brands, luxury cars, and observations "
        "outside the training distribution."
    ),
]

output = HERE / "task2_car_price_mlflow.ipynb"
nbf.write(nb, output)
print(output)
