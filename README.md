# Car Price Prediction

A simple Streamlit application that predicts a used car's selling price with the fitted machine-learning pipeline from `notebooks/a1_car_price_prediction.ipynb`.

## Run locally

Create and activate a virtual environment, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501> in a browser.

## Run with Docker

From the project root:

```bash
docker build -t car-price-predictor .
docker run --rm -p 8501:8501 car-price-predictor
```

Then open <http://localhost:8501>.

## Deploy the Docker version on Render

1. Push the repository to GitHub.
2. In the Render dashboard, select **New > Web Service** and connect the GitHub repository.
3. Select the `main` branch and choose the **Docker** runtime. Render will use the root `Dockerfile`.
4. Select the desired instance plan, set the health check path to `/_stcore/health`, and create the service.

Render provides a public `onrender.com` URL. With auto-deploy enabled, each later push to `main` rebuilds the image and updates the service without changing its URL.

## Project files

- `app.py`: Streamlit prediction interface
- `models/car_price_prediction_a1.joblib`: fitted preprocessing and Random Forest pipeline
- `notebooks/a1_car_price_prediction.ipynb`: data preparation, EDA, model training, and evaluation
- `REPORT.md`: concise description of the machine-learning methods and results
- `Dockerfile`: reproducible container definition

The predicted value is in Indian rupees (₹). It is an estimate and is most reliable for inputs similar to the training data.

# Assignment 2: Predicting Car Price

This repository contains my Assignment 2 submission for **AT82.03 Machine Learning**. It continues the car-price work from Assignment 1 by implementing linear regression from scratch, comparing 144 training configurations with MLflow, and deploying a two-model Streamlit application with Docker.

The website contains both models:

- **A1 Random Forest:** the older model with better predictive accuracy.
- **A2 polynomial regression:** the new from-scratch model selected with five-fold cross-validation. It is easier to inspect because its coefficients and training options are available.

Public deployment: <https://st127132.ml.brain.cs.ait.ac.th>

Docker Hub image: `lha007/car-price-a2:latest`

## Assignment objectives

### Task 1: Implementation

The course `LinearRegression` class was extended with:

1. an R-squared function calculated as `1 - SSE / SST`;
2. a choice between zero and Xavier weight initialization;
3. optional momentum with a configurable momentum value; and
4. a function that plots coefficient-based feature importance.

The implementation is in `notebooks/a2_regularization.py`. The main notebook imports this class instead of using scikit-learn's linear-regression estimator for the A2 experiment.

### Task 2: Experiment

The experiment compares:

- model: ordinary (`normal`), polynomial, lasso, or ridge;
- momentum: enabled or disabled;
- gradient method: stochastic, mini-batch, or batch;
- initialization: zeros or Xavier; and
- learning rate: `0.01`, `0.001`, or `0.0001`.

The full factorial design contains:

```text
4 models × 2 momentum options × 3 gradient methods
× 2 initialization methods × 3 learning rates = 144 configurations
```

Each configuration is evaluated using the same five shuffled folds. All parameters, fold metrics, and aggregate cross-validation metrics are recorded in MLflow.

### Task 3: Deployment

The Streamlit application contains separate old-model and new-model pages. Both pages accept the same car information, allow missing fields, and display a predicted price in Indian rupees. The A2 page also compares the A1 and A2 predictions for the same input and explains the difference between predictive accuracy and model transparency.

The application is packaged as a `linux/amd64` Docker image and deployed to the course `ml-brain` server through Traefik.

## Current project files

These are the authoritative files for the final A2 work:

```text
.
├── README.md
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
├── notebooks/
│   ├── a1_car_price_prediction.ipynb
│   ├── a2_car_price_prediction.ipynb
│   ├── a2_mlflow.py
│   └── a2_regularization.py
└── assignment2/
    ├── requirements.txt
    └── app/
        ├── app.py
        ├── Dockerfile
        ├── docker-compose.yaml
        └── requirements.txt
```

Some older generated copies remain under `assignment2/task1/` and `assignment2/task2/`. The final notebook and report use the files in the root-level `notebooks/`, `data/`, and `models/` directories shown above. In particular, MLflow screenshots must use `data/a2_mlflow.db`, not an older database under `assignment2/task2/`.

## Data preparation

The A2 experiment repeats the Assignment 1 cleaning process:

1. Remove CNG and LPG rows because their mileage values use different units.
2. Convert ownership categories into ordered numeric values.
3. Extract numeric values from mileage, engine, and maximum-power strings.
4. Use the first word of the vehicle name as its brand.
5. Remove the unused torque column.
6. Remove test-drive cars.
7. Remove the small set of incomplete rows used in the A1 cleaning procedure.
8. Remove duplicate rows after the transformations.

After cleaning, the dataset contains **6,607 rows**. The predictors are:

- numeric: year, kilometres driven, owner, mileage, engine, maximum power, and seats;
- categorical: brand, fuel, seller type, and transmission; and
- target: `log(selling_price)`.

The logarithm is used because selling price is strongly right-skewed. Predictions are converted back to rupees with the exponential function for the application.

## Preprocessing and leakage prevention

The cleaned data are divided into 80% training data and 20% held-out test data using `random_state=42`.

Inside each cross-validation fold:

- numeric values are median-imputed and standardized;
- categorical values are most-frequent-imputed and one-hot encoded;
- polynomial regression adds degree-two numeric terms; and
- the log target is standardized using only the fold's training portion.

The preprocessor is fitted separately inside every training fold. This prevents information from the validation fold from entering preprocessing. The held-out test set is used only after the best configuration has been selected.

## Experiment settings

The main settings in `notebooks/a2_mlflow.py` are:

| Setting | Value |
|---|---:|
| Train-test split | 80/20 |
| Random state | 42 |
| Cross-validation folds | 5 |
| Epochs during comparison | 12 |
| Epochs for final refit | 60 |
| Mini-batch size | 128 |
| Lasso/ridge strength | 0.001 |
| Momentum value when enabled | 0.9 |
| Gradient clipping norm | 10.0 |

Gradient clipping is used to keep the required stochastic and polynomial configurations numerically stable when testing the assignment's learning-rate grid.

## Verified results

The best configuration was selected by the highest mean cross-validation R-squared, using MSE as the tie-breaker:

| Parameter | Selected value |
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
| Ordinary (`normal`) | 0.880279 | 0.067852 |
| Ridge | 0.880120 | 0.067943 |
| Lasso | 0.879717 | 0.068170 |

After the winning configuration was fitted using the full training portion, the held-out test results were:

| Evaluation scale | MSE | RMSE | R-squared |
|---|---:|---:|---:|
| Log selling price | 0.056082 | 0.236817 | 0.896979 |
| Original price in rupees | 24,824,350,541.55 | 157,557.45 | 0.883156 |

The main Task 2 test R-squared is **0.896979** because the model was trained and evaluated using `log(selling_price)`. The original-price R-squared is **0.883156**. The two values differ because exponentiation is nonlinear; they are metrics calculated on different scales.

For comparison, the Assignment 1 Random Forest produced an original-price test R-squared of **0.929745** and RMSE of approximately **₹122,173**. Therefore, A1 is more accurate on the held-out test data. A2's advantage is transparency and its demonstration of the training methods required by this assignment.

## Feature importance

The A2 class ranks features by the absolute magnitude of their coefficients while preserving each coefficient's sign in the graph. Year and maximum power are among the strongest numeric features, and several brand categories also have large coefficients.

This interpretation has limitations. Numeric and polynomial numeric features are standardized, but one-hot encoded categories remain binary. Their coefficient magnitudes are therefore not perfectly comparable. Correlated polynomial terms can also share or redistribute importance, and coefficients describe model associations rather than causal effects.

## Environment setup

Python 3.14 was used for the final experiment. From the repository root, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r assignment2/requirements.txt
```

On Windows PowerShell, activation is normally:

```powershell
venv\Scripts\Activate.ps1
```

The saved `joblib` files depend on compatible NumPy, pandas, and scikit-learn versions. The Docker requirements pin the versions used by the application.

## Run the notebook

Start Jupyter from the project root:

```bash
source venv/bin/activate
jupyter notebook notebooks/a2_car_price_prediction.ipynb
```

The completed MLflow experiment does not need to be repeated just to read the notebook. Section 4 loads the saved CSV, summary, and model artifacts.

## Run the MLflow experiment

Only run the experiment again when intentionally creating a new set of results:

```bash
source venv/bin/activate
python notebooks/a2_mlflow.py
```

The script produces or updates:

- `data/a2_mlflow.db` — MLflow tracking database;
- `data/a2_experiment_results.csv` — ranked 144-row comparison table;
- `data/a2_best_model_summary.json` — best configuration and test metrics; and
- `models/best_a2_model.joblib` — preprocessing, model, target scaling, and metadata.

MLflow keeps old runs when the script is executed again, while the CSV is overwritten with the newest 144-row result table. Consequently, MLflow may show more than 144 runs after repeated executions.

## Open MLflow and capture matching screenshots

Use the root-level database that matches the final CSV. The absolute path avoids accidentally opening an empty or older database:

```bash
venv/bin/mlflow ui \
  --backend-store-uri sqlite:////Users/krown/AIT/ML/Assignments/A1_Car_Prediction/data/a2_mlflow.db \
  --port 5000
```

Open <http://127.0.0.1:5000> and select:

```text
A2 Task 2 - Based on A1 Car Price Notebook
```

MLflow normally sorts by time, while the CSV is sorted by R-squared. Display `cv_r2` and `cv_mse`, then sort `cv_r2` from highest to lowest. The first run should show:

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

Capture:

1. the runs table sorted by `cv_r2`; and
2. the best run's parameters and metrics.

Save the images as:

```text
docs/a2_mlflow_runs.png
docs/a2_mlflow_best_run.png
```

The final notebook displays these images when they exist. If MLflow contains repeated runs, use the latest group of 144 runs whose best metrics match the CSV values above.

## Run the Streamlit application without Docker

From the repository root:

```bash
source venv/bin/activate
streamlit run assignment2/app/app.py
```

Open <http://localhost:8501>. The sidebar provides:

- **Old model - A1 Random Forest**; and
- **New model - A2 Regression**.

Users may leave numeric fields blank or select **Not sure** for categorical fields. The saved preprocessing pipelines impute missing values before prediction.

## Build and test the Docker application locally

The Docker build context must be the repository root because the image copies files from `models/` and `notebooks/`:

```bash
docker build \
  -f assignment2/app/Dockerfile \
  -t car-price-a2:local .
```

Run it:

```bash
docker run --rm -p 8501:8501 car-price-a2:local
```

Open <http://localhost:8501>. The health endpoint is available at <http://localhost:8501/_stcore/health>.

## Publish the image to Docker Hub

The course server uses the `linux/amd64` architecture. Build and push that platform explicitly:

```bash
docker login

docker buildx build \
  --platform linux/amd64 \
  -f assignment2/app/Dockerfile \
  -t lha007/car-price-a2:latest \
  --push .
```

The current image is available as `lha007/car-price-a2:latest`.

## Connect to the course server

The public key must first be installed by the TA. Connect using the registered private key:

```bash
ssh -i ~/.ssh/id_ed25519 st127132@ml.brain.cs.ait.ac.th
```

After connecting, verify Docker access:

```bash
docker ps
```

The private key must never be committed, uploaded to GitHub, or copied to the server. Only the `.pub` public key should be shared with the TA.

## Upload the Compose file

The `scp` command must be run from the local Mac terminal, not from inside the remote SSH session:

```bash
scp -i ~/.ssh/id_ed25519 \
  /Users/krown/AIT/ML/Assignments/A1_Car_Prediction/assignment2/app/docker-compose.yaml \
  st127132@ml.brain.cs.ait.ac.th:~/car-price-a2/
```

If the destination does not exist yet, first create it on the server:

```bash
mkdir -p ~/car-price-a2
```

## Deploy on `ml-brain`

The current course server uses:

- external Traefik network: `web`;
- HTTPS entry point: `websecure`; and
- certificate resolver: `letsencrypt`.

These values are already configured in `assignment2/app/docker-compose.yaml`.

On the server:

```bash
cd ~/car-price-a2
docker compose config
docker compose pull
docker compose up -d
docker compose ps
```

Inspect the application logs:

```bash
docker compose logs --tail=100
```

Confirm that Traefik and the application share the `web` network:

```bash
docker network inspect web \
  --format '{{range .Containers}}{{.Name}}{{"\n"}}{{end}}'
```

The output should include `traefik` and `st127132-car-price-a2-car-price-a2-1`.

After Traefik obtains the certificate, open:

<https://st127132.ml.brain.cs.ait.ac.th>

## Update the deployed website later

After changing the application or model, build and push the same Docker tag again:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f assignment2/app/Dockerfile \
  -t lha007/car-price-a2:latest \
  --push .
```

Then connect to `ml-brain` and refresh the service:

```bash
cd ~/car-price-a2
docker compose pull
docker compose up -d --force-recreate
docker compose ps
```

The public URL remains unchanged.

## Troubleshooting

### MLflow and CSV show different orders

The CSV is already sorted by cross-validation R-squared. MLflow normally sorts by creation time. Sort the MLflow table by `cv_r2` descending and make sure the tracking database is `data/a2_mlflow.db`.

### MLflow shows more than 144 runs

MLflow appends runs whenever the experiment script is executed. Use the latest 144 runs that match the final CSV. Re-running the script is not required for the notebook or deployment.

### `ModuleNotFoundError: a2_regularization`

The A2 model was serialized with the custom class from `notebooks/a2_regularization.py`. The Dockerfile copies this module beside the application before loading the model. When running locally, start Streamlit with `assignment2/app/app.py`, which adds the notebook directory to the module path.

### `'str' object has no attribute 'derivation'`

The scratch regression class requires a regularization object such as `NoRegularization()`, `Lasso(...)`, or `Ridge(...)`, not a string. `notebooks/a2_mlflow.py` converts the experiment labels into the correct objects with `make_regularization()`.

### Container remains in `health: starting`

Wait approximately 30–60 seconds and run:

```bash
docker compose ps
docker compose logs --tail=100
```

Streamlit is ready when the log says that the Uvicorn server started on `0.0.0.0:8501` and the container becomes healthy.

### Traefik reports `invalid cluster node`

The old assignment template referred to `traefik_default`, which is not the current shared network. The final Compose file uses the server's `web` network.

### Traefik serves its default certificate

Check for certificate-resolver errors:

```bash
docker logs traefik --since 10m 2>&1 \
  | grep -iE 'st127132|certificate|resolver|acme|error'
```

The final Compose file uses the server's configured resolver, `letsencrypt`. Do not disable HTTPS or bypass browser certificate warnings.

### Website returns a gateway error

Verify the container status, application logs, and network membership:

```bash
docker compose ps
docker compose logs --tail=100
docker network inspect web
```

The label `traefik.docker.network=web` tells Traefik to connect through the correct shared network.

## Limitations

- Predictions are most reliable for cars similar to the training data.
- The application returns an estimate, not a guaranteed market valuation.
- Rare brands and expensive vehicles have less training support.
- The random train-test split may not measure future price changes.
- The A2 coefficient graph is only an approximate importance interpretation because numeric and one-hot features do not have identical scaling.
- The A2 experiment compares the assignment's required settings, but it does not exhaust every polynomial degree, regularization strength, epoch budget, or momentum value.

## Submission checklist

- [ ] `notebooks/a2_car_price_prediction.ipynb` runs and includes the report.
- [ ] The notebook shows the complete 144-row comparison table.
- [ ] The notebook reports test MSE and R-squared on the log scale clearly.
- [ ] The notebook includes the coefficient-based feature-importance graph.
- [ ] MLflow screenshots match `data/a2_experiment_results.csv`.
- [ ] No private SSH key, password, or Docker token is committed.
- [ ] `assignment2/app/` contains the application and deployment files.
- [ ] The Docker image is available as `lha007/car-price-a2:latest`.
- [ ] The public website opens at <https://st127132.ml.brain.cs.ait.ac.th>.

## Author

- Student ID: `st127132`
- Docker Hub: `lha007`
