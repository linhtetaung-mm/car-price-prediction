# Assignment 2 - Predicting Car Price

This submission extends Assignment 1 with a from-scratch linear-regression implementation, a 144-configuration MLflow experiment, a two-model Streamlit application, and Docker/Traefik deployment files.

## Deliverables

- `a2_car_price_prediction.ipynb`: implementation, experiment tables, final evaluation, coefficient importance, and findings
- `linear_regression.py`: reusable from-scratch estimator with R², Xavier initialization, momentum, and coefficient plotting
- `experiment_results.csv`: complete cross-validation comparison
- `models/car_price_prediction_a2.joblib`: fitted preprocessing and best A2 model
- `app/`: two-model Streamlit application, Dockerfile, requirements, and server Compose file

## Run the experiment

From the repository root:

```bash
pip install -r assignment2/requirements.txt
cd assignment2
python run_experiment.py
```

The script logs all configurations to `assignment2/mlflow_a2.db`, writes the final comparison table, and serializes the best model. The experiment compares:

- normal, polynomial, lasso, and ridge regression
- with and without momentum
- stochastic, mini-batch, and batch gradient descent
- zero and Xavier initialization
- learning rates 0.01, 0.001, and 0.0001

This is `4 × 2 × 3 × 2 × 3 = 144` configurations, each evaluated with five-fold cross-validation. MSE and R² are computed on log selling price.

## Open MLflow

From `assignment2/`:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_a2.db --port 5000
```

Open <http://localhost:5000>, select **A2 Car Price From-Scratch Regression**, sort by `cv_r2`, and capture screenshots for the notebook before submission.

## Run the two-model website locally

Build from the repository root because the image includes the A1 and A2 models:

```bash
docker build -f assignment2/app/Dockerfile -t car-price-a2 .
docker run --rm -p 8501:8501 car-price-a2
```

Open <http://localhost:8501>. The sidebar switches between the A1 Random Forest and A2 from-scratch model.

## Deploy to the course server

1. Replace `YOUR_DOCKERHUB_USERNAME` and every `STUDENT_ID` placeholder in `app/docker-compose.yaml`.
2. Build for the server architecture and push to Docker Hub:

```bash
docker login
docker buildx build --platform linux/amd64 -f assignment2/app/Dockerfile \
  -t YOUR_DOCKERHUB_USERNAME/car-price-a2:latest --push .
```

3. Upload or copy `docker-compose.yaml` to your account on `ml-brain`.
4. Connect using the private SSH key approved by the TA.
5. On the server, run:

```bash
docker compose pull
docker compose up -d
docker compose ps
```

6. Check Traefik and then open `https://STUDENT_ID.ml.brain.cs.ait.ac.th`.

Do not commit private SSH keys, passwords, or Docker Hub access tokens.

## Method notes

- Preprocessing is fitted independently inside every fold to avoid leakage.
- Numeric features use median imputation and standardization. Categorical features use most-frequent imputation and one-hot encoding.
- Polynomial regression adds degree-two terms to standardized numeric inputs.
- The target is log-transformed and standardized within each training fold for stable gradient descent.
- Gradient norm clipping protects stochastic configurations from extreme vehicle observations while preserving the assignment's learning-rate grid.
- Coefficient magnitudes are interpreted only after feature scaling.
