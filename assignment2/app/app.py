from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
A1_PATH = APP_DIR / "models" / "car_price_prediction_a1.joblib"
A2_PATH = APP_DIR / "models" / "car_price_prediction_a2.joblib"
UNKNOWN = "Not sure"


@st.cache_resource
def load_models():
    return joblib.load(A1_PATH), joblib.load(A2_PATH)


def optional_selectbox(label, options, default=None):
    choices = [UNKNOWN, *options]
    return st.selectbox(label, choices, index=choices.index(default) if default in choices else 0)


def missing(value):
    return np.nan if value is None or value == UNKNOWN else value


def prediction_form(key):
    brands = [
        "Ambassador", "Ashok", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun",
        "Fiat", "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep",
        "Kia", "Land", "Lexus", "MG", "Mahindra", "Maruti", "Mercedes-Benz",
        "Mitsubishi", "Nissan", "Opel", "Renault", "Skoda", "Tata", "Toyota",
        "Volkswagen", "Volvo",
    ]
    with st.form(key):
        left, right = st.columns(2)
        with left:
            brand = optional_selectbox("Brand", brands, "Maruti")
            year = st.number_input("Year", 1983, 2026, value=None, placeholder="Not sure")
            km_driven = st.number_input("Kilometres driven", 0, 2_500_000, value=None, placeholder="Not sure")
            fuel = optional_selectbox("Fuel", ["Diesel", "Petrol"], "Petrol")
            seller = optional_selectbox("Seller type", ["Dealer", "Individual", "Trustmark Dealer"], "Individual")
            transmission = optional_selectbox("Transmission", ["Automatic", "Manual"], "Manual")
        with right:
            owner = optional_selectbox("Ownership history", [
                "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"
            ], "First Owner")
            mileage = st.number_input("Mileage (km/l)", 5.0, 45.0, value=None, placeholder="Not sure")
            engine = st.number_input("Engine (CC)", 500, 4_000, value=None, placeholder="Not sure")
            power = st.number_input("Maximum power (bhp)", 20.0, 450.0, value=None, placeholder="Not sure")
            seats = st.number_input("Seats", 2, 14, value=None, placeholder="Not sure")
        submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

    owner_map = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3,
                 "Fourth & Above Owner": 4, UNKNOWN: np.nan}
    row = pd.DataFrame([{
        "year": missing(year), "km_driven": missing(km_driven),
        "owner": owner_map[owner], "mileage": missing(mileage),
        "engine": missing(engine), "max_power": missing(power),
        "seats": missing(seats), "brand": missing(brand), "fuel": missing(fuel),
        "seller_type": missing(seller), "transmission": missing(transmission),
    }])
    return submitted, row


st.set_page_config(page_title="Car Price Models", page_icon="🚗", layout="centered")
try:
    a1_model, a2_bundle = load_models()
except Exception as exc:
    st.error(f"Models could not be loaded: {exc}")
    st.stop()

st.sidebar.title("Car Price Models")
page = st.sidebar.radio("Choose a prediction page", ["A1 - Random Forest", "A2 - From-scratch Regression"])
st.title("🚗 Used Car Price Predictor")

if page.startswith("A1"):
    st.subheader("Assignment 1 model")
    st.write("This page uses the original Random Forest model. It is flexible and usually provides the stronger raw predictive accuracy.")
else:
    st.subheader("Assignment 2 model")
    st.write(
        "This page uses the best from-scratch regression configuration selected by five-fold "
        "cross-validation and tracked with MLflow. Compared with A1, it is more transparent: "
        "its optimization, regularization, initialization, momentum, and coefficients can all be inspected."
    )

st.info(
    "Enter the details you know and click **Predict price**. Leave numeric fields blank or "
    "choose **Not sure** when information is unavailable; the fitted preprocessing pipeline will impute it."
)
submitted, row = prediction_form("car-form")

if submitted:
    try:
        if page.startswith("A1"):
            prediction = float(np.exp(a1_model.predict(row)[0]))
        else:
            transformed = a2_bundle["preprocessor"].transform(row)
            prediction_log = (
                a2_bundle["model"].predict(transformed)[0] * a2_bundle["target_scale"]
                + a2_bundle["target_mean"]
            )
            prediction = float(np.exp(prediction_log))
        st.success(f"Estimated selling price: ₹{prediction:,.0f}")
        count = int(row.isna().sum(axis=1).iloc[0])
        if count:
            st.caption(f"The preprocessing pipeline imputed {count} missing field{'s' if count != 1 else ''}.")
    except Exception as exc:
        st.error(f"Prediction could not be produced: {exc}")

st.caption("Predictions are estimates in Indian rupees and are most reliable for cars similar to the training data.")
