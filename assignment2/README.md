# Assignment 2 - Predicting Car Price

This submission extends Assignment 1 with a from-scratch linear-regression implementation, a 144-configuration MLflow experiment, a two-model Streamlit application, and Docker/Traefik deployment files.

## Deliverables

- `../notebooks/a2_car_price_prediction.ipynb`: experiment tables, final evaluation, coefficient importance, and findings
- `../notebooks/a2_regularization.py`: from-scratch estimator with R², Xavier initialization, momentum, and coefficient plotting
- `../data/a2_experiment_results.csv`: complete cross-validation comparison
- `../models/best_a2_model.joblib`: fitted preprocessing and best A2 model
- `app/`: two-model Streamlit application, Dockerfile, requirements, and server Compose file

## Run the experiment

From the repository root:

```bash
source venv/bin/activate
python notebooks/a2_mlflow.py
```

The script logs all configurations to `data/a2_mlflow.db`, writes the final comparison table, and serializes the best model. The experiment compares:

- normal, polynomial, lasso, and ridge regression
- with and without momentum
- stochastic, mini-batch, and batch gradient descent
- zero and Xavier initialization
- learning rates 0.01, 0.001, and 0.0001

This is `4 × 2 × 3 × 2 × 3 = 144` configurations, each evaluated with five-fold cross-validation. MSE and R² are computed on log selling price.

## Open MLflow

From the repository root:

```bash
venv/bin/mlflow ui --backend-store-uri sqlite:///data/a2_mlflow.db --port 5000
```

Open <http://localhost:5000>, select **A2 Task 2 - Based on A1 Car Price Notebook**, sort by `cv_r2`, and capture screenshots for the notebook before submission.

## Run the two-model website locally

Build from the repository root because the image includes the A1 and A2 models:

```bash
docker build -f assignment2/app/Dockerfile -t car-price-a2 .
docker run --rm -p 8501:8501 car-price-a2
```

Open <http://localhost:8501>. The sidebar switches between the A1 Random Forest and A2 from-scratch model.

The A1 model is more accurate on the held-out test set (R² 0.9297 versus 0.8832). The A2 page therefore describes its real improvement: transparent coefficients and a from-scratch training process whose initialization, momentum, optimizer, regularization, and learning rate can be inspected. It also displays the two estimates side by side for the same input.

## Deploy to the course server

1. Replace `YOUR_DOCKERHUB_USERNAME` and every `STUDENT_ID` placeholder in `app/docker-compose.yaml`.
2. Build for the server architecture and push to Docker Hub:

```bash
docker login
docker buildx build --platform linux/amd64 -f assignment2/app/Dockerfile \
  -t YOUR_DOCKERHUB_USERNAME/car-price-a2:latest --push .
```

3. Upload or copy `assignment2/app/docker-compose.yaml` to your account on `ml-brain`.
4. Connect using the private SSH key approved by the TA.
5. On the server, run:

```bash
docker compose pull
docker compose up -d
docker compose ps
```

6. Check Traefik and then open `https://STUDENT_ID.ml.brain.cs.ait.ac.th`.

If `docker compose ps` does not show the service as running, inspect it with
`docker compose logs --tail=100`. If `docker ps` itself gives a permission error,
contact the TA as required by the assignment brief.

Do not commit private SSH keys, passwords, or Docker Hub access tokens.

## Method notes

- Preprocessing is fitted independently inside every fold to avoid leakage.
- Numeric features use median imputation and standardization. Categorical features use most-frequent imputation and one-hot encoding.
- Polynomial regression adds degree-two terms to standardized numeric inputs.
- The target is log-transformed and standardized within each training fold for stable gradient descent.
- Gradient norm clipping protects stochastic configurations from extreme vehicle observations while preserving the assignment's learning-rate grid.
- Coefficient magnitudes are interpreted only after feature scaling.
