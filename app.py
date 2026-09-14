from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


UNKNOWN = "Not sure"
FEATURE_COLUMNS = [
    "year", "km_driven", "owner", "mileage", "engine", "max_power", "seats",
    "brand", "fuel", "seller_type", "transmission",
]
BRANDS = [
    "Ambassador", "Ashok", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun",
    "Fiat", "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep",
    "Kia", "Land", "Lexus", "MG", "Mahindra", "Maruti", "Mercedes-Benz",
    "Mitsubishi", "Nissan", "Opel", "Renault", "Skoda", "Tata", "Toyota",
    "Volkswagen", "Volvo",
]

# Both were calculated on the same held-out A1/A2 split.
A1_TEST = {"r2": 0.9297451439, "rmse": 122172.5906, "mae": 71125.8555}
A2_TEST = {"r2": 0.8831559296, "rmse": 157557.4516, "mae": 87678.9260}


def first_existing_path(*paths):
    """Find a model in either the Docker or local project layout."""
    return next((path for path in paths if path.exists()), paths[0])


A1_PATH = ( ROOT / "models" / "car_price_prediction_a1.joblib")
A2_PATH = (ROOT / "models" / "best_a2_model.joblib")


@st.cache_resource
def load_models():
    missing_paths = [str(path) for path in [A1_PATH, A2_PATH] if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing_paths))
    return joblib.load(A1_PATH), joblib.load(A2_PATH)


def optional_selectbox(label, options, key, default=None):
    choices = [UNKNOWN, *options]
    index = choices.index(default) if default in choices else 0
    return st.selectbox(label, choices, index=index, key=key)


def missing(value):
    return np.nan if value is None or value == UNKNOWN else value


def prediction_form(page_key):
    """Render the form shared by the old and new model pages."""
    with st.form(f"{page_key}-prediction-form"):
        left, right = st.columns(2)
        with left:
            brand = optional_selectbox("Brand", BRANDS, f"{page_key}-brand", "Maruti")
            year = st.number_input(
                "Year", 1983, 2026, value=None, step=1,
                placeholder="Not sure", key=f"{page_key}-year",
            )
            km_driven = st.number_input(
                "Kilometres driven", 0, 2_500_000, value=None, step=1_000,
                placeholder="Not sure", key=f"{page_key}-km",
            )
            fuel = optional_selectbox(
                "Fuel", ["Diesel", "Petrol"], f"{page_key}-fuel", "Petrol"
            )
            seller = optional_selectbox(
                "Seller type", ["Dealer", "Individual", "Trustmark Dealer"],
                f"{page_key}-seller", "Individual",
            )
            transmission = optional_selectbox(
                "Transmission", ["Automatic", "Manual"],
                f"{page_key}-transmission", "Manual",
            )
        with right:
            owner = optional_selectbox(
                "Ownership history",
                ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
                f"{page_key}-owner", "First Owner",
            )
            mileage = st.number_input(
                "Mileage (km/l)", 5.0, 45.0, value=None, step=0.1,
                placeholder="Not sure", key=f"{page_key}-mileage",
            )
            engine = st.number_input(
                "Engine (CC)", 500, 4_000, value=None, step=10,
                placeholder="Not sure", key=f"{page_key}-engine",
            )
            power = st.number_input(
                "Maximum power (bhp)", 20.0, 450.0, value=None, step=1.0,
                placeholder="Not sure", key=f"{page_key}-power",
            )
            seats = st.number_input(
                "Seats", 2, 14, value=None, step=1,
                placeholder="Not sure", key=f"{page_key}-seats",
            )
        submitted = st.form_submit_button(
            "Predict price", type="primary", width="stretch"
        )

    owner_map = {
        "First Owner": 1, "Second Owner": 2, "Third Owner": 3,
        "Fourth & Above Owner": 4, UNKNOWN: np.nan,
    }
    row = pd.DataFrame([{
        "year": missing(year), "km_driven": missing(km_driven),
        "owner": owner_map[owner], "mileage": missing(mileage),
        "engine": missing(engine), "max_power": missing(power),
        "seats": missing(seats), "brand": missing(brand), "fuel": missing(fuel),
        "seller_type": missing(seller), "transmission": missing(transmission),
    }], columns=FEATURE_COLUMNS)
    return submitted, row


def predict_a1(model, row):
    return float(np.exp(model.predict(row)[0]))


def predict_a2(bundle, row):
    transformed = bundle["preprocessor"].transform(row)
    # The course LinearRegression class expects a leading intercept column.
    transformed = np.column_stack([np.ones(transformed.shape[0]), transformed])
    prediction_log = (
        bundle["model"].predict(transformed)[0] * bundle["target_scale"]
        + bundle["target_mean"]
    )
    return float(np.exp(prediction_log))


def show_missing_message(row):
    count = int(row.isna().sum(axis=1).iloc[0])
    if count:
        st.caption(
            f"The trained preprocessing pipeline automatically filled {count} missing "
            f"field{'s' if count != 1 else ''}."
        )


def show_model_comparison():
    comparison = pd.DataFrame([
        {
            "Model": "A1 Random Forest", "Test R²": A1_TEST["r2"],
            "Test RMSE": A1_TEST["rmse"], "Test MAE": A1_TEST["mae"],
            "Main advantage": "Higher predictive accuracy",
        },
        {
            "Model": "A2 polynomial regression", "Test R²": A2_TEST["r2"],
            "Test RMSE": A2_TEST["rmse"], "Test MAE": A2_TEST["mae"],
            "Main advantage": "Transparent coefficients and training",
        },
    ])
    st.dataframe(
        comparison.style.format({
            "Test R²": "{:.4f}", "Test RMSE": "₹{:,.0f}", "Test MAE": "₹{:,.0f}",
        }),
        hide_index=True,
        width="stretch",
    )


st.set_page_config(
    page_title="Car Price Model Comparison", page_icon="🚗", layout="centered"
)
try:
    a1_model, a2_bundle = load_models()
except Exception as exc:
    st.error(f"The prediction models could not be loaded. {exc}")
    st.stop()

st.sidebar.title("🚗 Car Price Models")
page = st.sidebar.radio(
    "Choose a page", ["Old model - A1 Random Forest", "New model - A2 Regression"]
)
st.sidebar.caption("Both pages use the same inputs, so their estimates are comparable.")
st.title("Used Car Price Predictor")

if page.startswith("Old"):
    st.subheader("Old page: A1 Random Forest")
    st.write(
        "This is the original Assignment 1 model. It combines many decision trees and has "
        "the stronger held-out test accuracy."
    )
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Test R²", f"{A1_TEST['r2']:.4f}")
    metric_right.metric("Test RMSE", f"₹{A1_TEST['rmse']:,.0f}")
    st.info(
        "Enter the details you know and select **Predict price**. Leave a numeric field "
        "blank or choose **Not sure** when information is unavailable; the pipeline will "
        "fill it using values learned from the training data."
    )
    submitted, row = prediction_form("a1")
    if submitted:
        try:
            st.success(f"A1 estimated selling price: ₹{predict_a1(a1_model, row):,.0f}")
            show_missing_message(row)
        except Exception as exc:
            st.error(f"Prediction could not be produced. Please check the inputs. {exc}")
else:
    st.subheader("New page: A2 polynomial regression")
    st.write(
        "The new model was selected from 144 MLflow-tracked configurations using five-fold "
        "cross-validation. Its main improvement is **transparency**, not raw accuracy: it was "
        "implemented from scratch and exposes its coefficients, initialization, momentum, "
        "gradient-descent method, and learning rate."
    )
    st.warning(
        "The A1 Random Forest remains more accurate on the same held-out test set. A2 is "
        "better for understanding how the model was trained and how features contribute."
    )
    show_model_comparison()
    st.info(
        "Enter the details you know and select **Predict price**. Missing fields are allowed. "
        "After submission, this page shows both model estimates for the same car."
    )
    submitted, row = prediction_form("a2")
    if submitted:
        try:
            a2_prediction = predict_a2(a2_bundle, row)
            a1_prediction = predict_a1(a1_model, row)
            new_column, old_column = st.columns(2)
            new_column.metric("New A2 estimate", f"₹{a2_prediction:,.0f}")
            old_column.metric("Old A1 estimate", f"₹{a1_prediction:,.0f}")
            difference = a2_prediction - a1_prediction
            st.caption(
                f"For this input, A2 is ₹{abs(difference):,.0f} "
                f"{'higher' if difference >= 0 else 'lower'} than A1."
            )
            show_missing_message(row)
        except Exception as exc:
            st.error(f"Prediction could not be produced. Please check the inputs. {exc}")

with st.expander("How were the models compared?"):
    st.write(
        "Both results use the same cleaned data, 80/20 split, random state 42, and log-price "
        "target. The test set was held back during model selection. Higher R² and lower RMSE/MAE "
        "indicate better predictive performance."
    )

st.caption(
    "Predictions are estimates in Indian rupees, not guaranteed market valuations. They are "
    "most reliable for vehicles similar to the training data."
)
