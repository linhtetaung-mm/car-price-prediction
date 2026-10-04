# Car Price Prediction — Assignments 1, 2 and 3

## Assignment 3 — Four-class price prediction

**Name:** Lin Htet Aung

**Student ID:** st127132

For A3, I use the same car dataset and cleaning steps from my previous assignments. Instead of predicting an exact price, the model predicts one of four price categories.

The [notebook](notebooks/a3_car_price_prediction.ipynb) follows the same order as A1/A2: load the data, clean it, inspect the target, train the model and discuss the results. The application is in [app/](app/).

### What I did

1. Kept the data cleaning from A1/A2 and converted selling price into four classes.
2. Implemented multinomial logistic regression with NumPy and added an optional ridge penalty.
3. Calculated accuracy, precision, recall, F1, macro averages and weighted averages, then checked them against sklearn.
4. Compared 12 configurations with five-fold cross-validation and recorded them in local MLflow.
5. Saved the best pipeline, added a Streamlit app and prepared model tests and a GitHub deployment workflow.

### Price labels

| Class | Selling price (INR) |
|---|---|
| 0 | 0 < price ≤ 300,000 |
| 1 | 300,000 < price ≤ 600,000 |
| 2 | 600,000 < price ≤ 1,000,000 |
| 3 | price > 1,000,000 |

I use fixed rupee boundaries so each class has a clear meaning. There are 6,607 rows after cleaning: 5,285 for training and 1,322 for testing. The split uses stratification and random state 42. As in A2, preprocessing is fitted separately inside each training fold.

### Results and observations

I selected the model using mean **CV macro F1** because the four classes have different numbers of samples. The best settings are learning rate `0.3` and L2 `0.0001`, using up to 600 batch-gradient epochs.

| Metric | Result |
|---|---:|
| Mean CV macro F1 | 0.7696 |
| CV macro F1 standard deviation | 0.0139 |
| Test accuracy | 0.7610 |
| Test macro F1 | 0.7360 |
| Test weighted F1 | 0.7608 |
| Majority-class baseline accuracy | 0.3805 |

The best unregularized CV macro F1 is 0.76955, almost identical to ridge's 0.76959. This difference is much smaller than fold variation and does not establish a meaningful regularization advantage. The fixed epoch budget also affects learning-rate comparisons.

Class 2 is weakest by F1 (0.6494); premium-class recall is 0.6064. The model has linear boundaries in the encoded feature space and does not explicitly model the ordering of price bands. Its probabilities have not been calibrated. These classification results are not directly comparable to A1/A2 regression R² or RMSE.

**Support** means the number of actual observations of a class in the evaluated data. Weighted averages use support fractions that sum to one. The extra division by four in the brief's weighted-average example is omitted to match scikit-learn. Undefined metric divisions return zero.

The objective is `mean(cross entropy) + l2 * sum(W²)`, with gradient `X.T @ (P - Y) / m + 2*l2*W`. Bias is not penalized. Set `l2=0` to disable the penalty. Because the brief uses summed loss, its numerical lambda corresponds to `m * l2` here. The notebook explains the equations and works through the metrics on a small example. The implementation is in `a3/model.py` and `a3/metrics.py`; the training loop is in `a3/train.py`, similar to the separate Python files used for A2.

### Run locally

Use Python 3.14 with the pinned packages matching the saved model:

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements-a3.txt
python -m unittest discover -s tests -v
streamlit run app/streamlit_app.py
```

Open the notebook with a Python interpreter using this environment. Its saved outputs are included. `RUN_EXPERIMENTS = False` uses the saved experiment results; change it to `True` to retrain, or run:

```bash
python -m a3.train
mlflow ui --backend-store-uri sqlite:///data/a3_mlflow.db
```

The local SQLite database and MLflow artifact folders are machine-local outputs and are ignored by Git. After a fresh clone, rerun training before publishing. The trained application artifact, results CSV, JSON summary, and executed notebook are included in the submission files. The database also contains earlier runs. The results CSV identifies the 12 candidates from the latest completed experiment.

### Upload to the course MLflow server later

Use the **updated HTTPS endpoint**, `https://mlflow.ml.brain.cs.ait.ac.th`, with username `student` and the password supplied by the course. The experiment is `st127132-a3`; the model is `st127132-a3-model`. Do not put the password into committed files.

The notebook's remote cell sets `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`, prompting for the password with `getpass`. Set `PUBLISH_REMOTE = True` to upload. Alternatively, set both variables in your shell environment and run:

```bash
python -m a3.publish
```

This uploads the 12 saved local runs with parameters, fold metrics and models, resumes interrupted uploads, registers the selected model, and sets it to **Staging**. It touches only this student's named experiment/model and never logs the dataset. It verifies the registered model can be loaded before saving `data/a3_remote_receipt.json` as evidence. The receipt is authoritative for remote completion; the training summary describes the original local run.

[MLflow stages are deprecated](https://mlflow.org/docs/latest/ml/model-registry/workflow/), but the assignment explicitly asks for Staging, so this code keeps that requirement. A server that no longer supports stages will raise an error rather than claim success. Model files are saved with explicit cloudpickle serialization for compatibility with the custom NumPy estimator and the course registry.

### Deployment and CI/CD

The [workflow](.github/workflows/a3-ci-cd.yml) runs model tests on every push and pull request. On a successful default-branch run, it builds `app/Dockerfile`, pushes a commit-tagged image to `lha007/car-price-a3`, and deploys to `~/car-price-a3` on the course VM. The deployment is gated by the test job, container health, and public HTTPS health checks.

Create the GitHub environment **course-vm** and configure these secrets:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account with push access to `lha007/car-price-a3` |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `A3_SSH_PRIVATE_KEY` | Deployment key authorized for `st127132` on the VM |
| `A3_SSH_KNOWN_HOSTS` | Verified SSH host-key entry for `ml.brain.cs.ait.ac.th` |

The Docker Hub repository must be public for the current unauthenticated VM pull, or you must configure registry authentication on the VM. GitHub-hosted runners must also be able to reach the VM; a campus-only VM needs a reachable self-hosted runner. Verify host keys through a trusted course source before saving them. See [app/README.md](app/README.md) for Docker and manual deployment commands.

Planned A3 URL: **https://st127132.ml.brain.cs.ait.ac.th/a3/**. The separate `/a3` router allows the existing A1/A2 app to remain at `/`.

**Current status (4 October 2026):** A3 has been manually deployed on mlbrain and passed its container health check and six model tests. The shared web ports 80/443 are unavailable, so the public HTTPS address still refuses connections. Use the SSH tunnel described in [app/README.md](app/README.md) to open the deployed app at `http://localhost:18503/a3/`. MLflow upload/Staging registration and verification of GitHub Actions deployment are still pending.

### A3 file guide

| File | Purpose |
|---|---|
| `notebooks/a3_car_price_prediction.ipynb` | Executed explanation, comparisons, figures and remote-upload cell |
| `a3/model.py`, `a3/metrics.py` | From-scratch model and metrics |
| `a3/data.py` | A1/A2 preprocessing and fixed price classes |
| `a3/train.py`, `a3/publish.py` | Reproducible experiment and resumable remote upload |
| `models/best_a3_model.joblib` | Selected fitted pipeline |
| `data/a3_experiment_results.csv` | All 12 completed configurations |
| `data/a3_best_model_summary.json` | Selection, test report, split counts and provenance |
| `data/a3_confusion_matrix.csv` | Held-out actual-by-predicted class counts |
| `tests/test_a3.py` | Required input/output tests and mathematical checks |
| `app/` | A3 web application, Dockerfile and Compose configuration |
| `.github/workflows/a3-ci-cd.yml` | Tests before image publication and deployment |


### References used during revision

I reviewed [SaniahKayenat's A3 project](https://github.com/SaniahKayenat/AT82.03-Machine-Learning-Assignment-3), [annasus-10's car-price project](https://github.com/annasus-10/Car_Price_Prediction), and [KHH-AKA-Lucifer's car-price project](https://github.com/KHH-AKA-Lucifer/car-price-prediction) for comparison. The first covers classification; the other two mainly cover A1 regression. No code or report text was copied from those repositories. The notebook style and cleaning steps follow my own A1/A2 work, and the model follows the course softmax example linked in the notebook.

---

## Assignments 1 and 2


This project predicts the selling price of a used car in Indian rupees. It contains my Assignment 1 Random Forest model, my Assignment 2 regression model implemented from scratch, the MLflow experiment results, and a two-page Streamlit website for comparing both models.

Live course-server deployment: <https://st127132.ml.brain.cs.ait.ac.th>

Docker Hub image: `lha007/car-price-a2:latest`

## What the website does

The Streamlit application has two pages:

- **Old model — A1 Random Forest:** the more accurate model on the held-out test data.
- **New model — A2 Regression:** the model selected from the Assignment 2 cross-validation experiment. Its main benefit is that the training process and coefficients are easier to explain.

The user enters the car information and clicks **Predict price**. Every field may be skipped. Numeric fields can be left blank, while categorical fields have a **Not sure** option. The preprocessing pipelines fill missing numeric values with medians and missing categorical values with the most frequent training value.

Predictions are estimates, not guaranteed market valuations. They are most reliable for cars that are similar to the training data.

## Final project structure

```text
.
├── app.py
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── README.md
├── REPORT.md
├── data/
│   ├── Cars.csv
│   ├── a2_best_model_summary.json
│   ├── a2_experiment_results.csv
│   └── a2_mlflow.db
├── docs/
│   ├── a2_mlflow_best_run.png
│   └── a2_mlflow_runs.png
├── models/
│   ├── car_price_prediction_a1.joblib
│   └── best_a2_model.joblib
└── notebooks/
    ├── a1_car_price_prediction.ipynb
    ├── a2_car_price_prediction.ipynb
    ├── a2_mlflow.py
    ├── a2_regularization.py
    └── original_regularization.ipynb
```

Important deployment files:

- `app.py` is the Streamlit entry point.
- `requirements.txt` contains only the packages needed by the deployed application.
- `models/` contains both fitted model artifacts.
- `notebooks/a2_regularization.py` defines the custom class required to load the A2 model.
- `Dockerfile` builds the container image.
- `docker-compose.yaml` deploys that image to the course server through Traefik.

## Assignment 2 tasks

### Task 1 — Linear regression class

The course `LinearRegression` class was extended with:

1. an R-squared function calculated as `1 - SSE / SST`;
2. a choice between zero and Xavier weight initialization;
3. optional momentum with a configurable momentum value; and
4. a function that plots feature importance from coefficient magnitudes.

The implementation is in `notebooks/a2_regularization.py`.

Coefficient magnitude is meaningful only when the input features use comparable scales. The numeric and polynomial numeric features are standardized in this project, but one-hot encoded categories are binary, so the graph should still be interpreted carefully. It shows model associations rather than causal effects.

### Task 2 — Cross-validation and MLflow

The experiment compares:

- normal, polynomial, lasso, and ridge regression;
- training with and without momentum;
- stochastic, mini-batch, and batch gradient descent;
- zero and Xavier initialization; and
- learning rates of `0.01`, `0.001`, and `0.0001`.

This produces 144 configurations:

```text
4 model choices × 2 momentum choices × 3 gradient methods
× 2 initializations × 3 learning rates = 144 configurations
```

Each configuration uses the same five shuffled cross-validation folds. Parameters, fold results, mean MSE, and mean R-squared are tracked in MLflow. Preprocessing is fitted independently inside each fold to avoid validation-data leakage.

### Task 3 — Deployment

The two-page website lets a user make a prediction with the A1 model or compare the A1 and A2 estimates for the same car. It can be deployed directly on Streamlit Community Cloud or as a Docker container.

## Data preparation

The Assignment 2 experiment follows the Assignment 1 cleaning procedure:

1. Remove CNG and LPG rows because their mileage values use different units.
2. Convert ownership labels to ordered numeric values.
3. Extract numeric mileage, engine, and maximum-power values.
4. Use the first word of the vehicle name as its brand.
5. Remove the unused torque column.
6. Remove test-drive cars.
7. Remove the small group of incomplete training rows.
8. Remove duplicate rows after transformation.

The cleaned data contain 6,607 rows. The model inputs are:

- numeric: year, kilometres driven, owner, mileage, engine, maximum power, and seats;
- categorical: brand, fuel, seller type, and transmission; and
- target: `log(selling_price)`.

The log target reduces the strong right skew in selling prices. The website converts predictions back to rupees with the exponential function.

## Verified Assignment 2 results

The best configuration was selected using the highest mean cross-validation R-squared, with MSE as the tie-breaker.

| Parameter | Best value |
|---|---|
| Model | Polynomial regression |
| Regularization | None (`normal`) |
| Momentum | Disabled |
| Gradient method | Stochastic |
| Initialization | Zeros |
| Learning rate | 0.001 |
| Mean CV MSE, log scale | 0.060138 |
| Mean CV R-squared, log scale | 0.893693 |
| CV R-squared standard deviation | 0.004543 |

The best result from each model family was:

| Model family | Best CV R-squared | CV MSE |
|---|---:|---:|
| Polynomial | 0.893693 | 0.060138 |
| Normal | 0.880279 | 0.067852 |
| Ridge | 0.880120 | 0.067943 |
| Lasso | 0.879717 | 0.068170 |

After refitting the winning configuration on the full training portion, its held-out test results were:

| Evaluation scale | MSE | RMSE | R-squared |
|---|---:|---:|---:|
| Log selling price | 0.056082 | 0.236817 | 0.896979 |
| Original price in rupees | 24,824,350,541.55 | 157,557.45 | 0.883156 |

The Task 2 model is trained on log price, so **0.896979** is its log-scale test R-squared. After transforming predictions back to rupees, its original-price R-squared is **0.883156**. These values are different because exponentiation is nonlinear.

For a fair website comparison, both models are shown on the original rupee scale:

| Model | Test R-squared | Test RMSE | Test MAE |
|---|---:|---:|---:|
| A1 Random Forest | 0.929745 | ₹122,173 | ₹71,126 |
| A2 polynomial regression | 0.883156 | ₹157,557 | ₹87,679 |

The Random Forest is therefore more accurate on this test set. The A2 model is still useful because it demonstrates the required optimization methods and provides interpretable coefficients.

## Run the website locally

Python 3.14 was used for the final application. From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

On Windows PowerShell, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

MLflow does not need to be running for the website. The application loads the two saved `joblib` files directly.

## Deploy on Streamlit Community Cloud

Streamlit Community Cloud runs `app.py` directly and does not use the Dockerfile. The repository must therefore include all of the following committed files:

```text
app.py
requirements.txt
models/car_price_prediction_a1.joblib
models/best_a2_model.joblib
notebooks/a2_regularization.py
```

Deployment steps:

1. Commit these files and push the repository to GitHub.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Select **Create app** and choose the repository and branch.
4. Set the main file path to `app.py`.
5. Open **Advanced settings** and choose Python 3.14, matching the environment used for this project.
6. No secrets or environment variables are required.
7. Select **Deploy** and wait for dependency installation and model loading to finish.

The initial start can take longer because the A1 model file is about 39 MB. Later pushes to the selected branch update the same deployed application and keep the same `streamlit.app` URL.

If a deployment fails, open **Manage app** and inspect the logs. The most important checks are that all model files were pushed to GitHub and that the dependency versions in `requirements.txt` installed successfully.

Official deployment references:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)

## Run the MLflow experiment

The saved results are already available, so this experiment does not need to run before the notebook or website. To intentionally repeat it, first install the experiment-only package:

```bash
source venv/bin/activate
pip install mlflow==3.16.0
python notebooks/a2_mlflow.py
```

The script creates or updates:

- `data/a2_mlflow.db` — the MLflow tracking database;
- `data/a2_experiment_results.csv` — the ranked 144-configuration table;
- `data/a2_best_model_summary.json` — the winning configuration and test metrics; and
- `models/best_a2_model.joblib` — the fitted A2 bundle.

MLflow appends runs, while the CSV is overwritten with the newest experiment table. Repeating the script can therefore produce more than 144 runs in MLflow.

To inspect the saved experiment:

```bash
venv/bin/mlflow ui \
  --backend-store-uri sqlite:////Users/krown/AIT/ML/Assignments/A1_Car_Prediction/data/a2_mlflow.db \
  --port 5000
```

Open <http://127.0.0.1:5000>, select the A2 experiment, display `cv_r2` and `cv_mse`, and sort `cv_r2` from highest to lowest. The best run should match:

```text
model_type       polynomial
regularization   normal
use_momentum     False
method           stochastic
initialization   zeros
learning_rate    0.001
cv_r2            0.893693
cv_mse           0.060138
```

## Run the notebooks

Install Jupyter in the same environment if it is not already available:

```bash
source venv/bin/activate
pip install notebook
jupyter notebook
```

Then open `notebooks/a1_car_price_prediction.ipynb` or `notebooks/a2_car_price_prediction.ipynb`.

## Build and run with Docker

From the repository root:

```bash
docker build -t car-price-a2:local .
docker run --rm -p 8501:8501 car-price-a2:local
```

Open <http://localhost:8501>. The health endpoint is <http://localhost:8501/_stcore/health>.

To publish the image for the `linux/amd64` course server:

```bash
docker login
docker buildx build \
  --platform linux/amd64 \
  -t lha007/car-price-a2:latest \
  --push .
```

## Deploy the Docker image on `ml-brain`

Copy the Compose file from the local Mac terminal:

```bash
ssh -i ~/.ssh/id_ed25519 st127132@ml.brain.cs.ait.ac.th \
  'mkdir -p ~/car-price-a2'

scp -i ~/.ssh/id_ed25519 \
  /Users/krown/AIT/ML/Assignments/A1_Car_Prediction/docker-compose.yaml \
  st127132@ml.brain.cs.ait.ac.th:~/car-price-a2/
```

Then connect to the server and deploy:

```bash
ssh -i ~/.ssh/id_ed25519 st127132@ml.brain.cs.ait.ac.th
cd ~/car-price-a2
docker compose config
docker compose pull
docker compose up -d
docker compose ps
docker compose logs --tail=100
```

The final Compose file uses the course server's external `web` network and the `letsencrypt` certificate resolver.

After changing the code or a model, publish the same image tag again and refresh the service:

```bash
docker compose pull
docker compose up -d --force-recreate
docker compose ps
```

The public course-server URL remains unchanged.

## Troubleshooting

### `ModuleNotFoundError: No module named 'a2_regularization'`

The A2 `joblib` file was saved with a custom class. Keep `notebooks/a2_regularization.py` in the GitHub repository. The root `app.py` adds this directory to Python's module path before loading the artifact.

### The model files are missing on Streamlit Cloud

Confirm that both `.joblib` files are committed and visible on GitHub. A file that exists only on the local computer is not available to the cloud application.

### MLflow and the CSV appear in a different order

The CSV is sorted by cross-validation R-squared, while MLflow normally sorts by creation time. Sort the MLflow table by `cv_r2` descending and use the run whose metrics match the values in this README.

### The Docker container stays in `health: starting`

Wait approximately 30–60 seconds, then check:

```bash
docker compose ps
docker compose logs --tail=100
```

### The website gives unrealistic output

Check the input units: kilometres driven, mileage in km/l, engine size in CC, and maximum power in bhp. Predictions far outside the training distribution are less reliable.

## Limitations

- Rare brands and expensive cars have less support in the training data.
- A random train-test split does not measure future market changes.
- Missing-value imputation is convenient but adds uncertainty.
- Coefficient magnitude is only an approximate feature-importance measure.
- The A2 experiment tests the assignment grid, not every possible hyperparameter.

## Submission checklist

- [ ] Both notebooks open correctly and the A2 notebook contains the final report.
- [ ] The A2 notebook displays the 144-row comparison table.
- [ ] Log-scale and original-price metrics are clearly labelled.
- [ ] The coefficient feature-importance graph is included.
- [ ] MLflow screenshots match `data/a2_experiment_results.csv`.
- [ ] Both `.joblib` model files are committed.
- [ ] No private key, password, or Docker token is committed.
- [ ] The local Streamlit application opens and both forms predict successfully.
- [ ] The Streamlit Community Cloud URL opens successfully.
- [ ] The Docker image is available as `lha007/car-price-a2:latest`.

## Author

- Student ID: `st127132`
- Docker Hub: `lha007`
