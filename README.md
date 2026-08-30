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

## Project files

- `app.py`: Streamlit prediction interface
- `models/car_price_prediction_a1.joblib`: fitted preprocessing and Random Forest pipeline
- `notebooks/a1_car_price_prediction.ipynb`: data preparation, EDA, model training, and evaluation
- `REPORT.md`: concise description of the machine-learning methods and results
- `Dockerfile`: reproducible container definition

The predicted value is in Indian rupees (₹). It is an estimate and is most reliable for inputs similar to the training data.
