from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_price_prediction_a1.joblib"

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    """Load the fitted preprocessing and Random Forest pipeline once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"The prediction model could not be loaded. {exc}")
    st.stop()

brands = [
    "Ambassador", "Ashok", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun",
    "Fiat", "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep",
    "Kia", "Land", "MG", "Mahindra", "Maruti", "Mercedes-Benz", "Mitsubishi",
    "Nissan", "Opel", "Renault", "Skoda", "Tata", "Toyota", "Volkswagen", "Volvo",
]

st.title("🚙 Car Price Predictor")
st.caption("Estimate a used car's selling price with the trained Random Forest model.")
st.info(
    "Enter the information you know, then click **Predict price**. "
    "Choose **Not sure** for any unknown field; the model will automatically fill missing "
    "numeric values with the training median and missing categories with the most frequent value."
)

UNKNOWN = "Not sure"


def optional_selectbox(label, options, default=None):
    """Create a select box with an explicit missing-value option."""
    choices = [UNKNOWN, *options]
    selected_index = choices.index(default) if default in choices else 0
    return st.selectbox(label, choices, index=selected_index)


def missing_if_unknown(value):
    """Convert the form's unknown marker into a value understood by the pipeline."""
    return np.nan if value == UNKNOWN or value is None else value

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        brand = optional_selectbox("Brand", brands, default="Maruti")
        year = st.number_input(
            "Year", min_value=1983, max_value=2026, value=None, step=1,
            placeholder="Not sure",
        )
        km_driven = st.number_input(
            "Kilometres driven", min_value=0, max_value=2_500_000, value=None,
            step=1_000, placeholder="Not sure",
        )
        fuel = optional_selectbox("Fuel", ["Diesel", "Petrol"], default="Petrol")
        seller_type = optional_selectbox(
            "Seller type", ["Dealer", "Individual", "Trustmark Dealer"],
            default="Individual",
        )
        transmission = optional_selectbox(
            "Transmission", ["Automatic", "Manual"], default="Manual"
        )

    with right:
        owner_label = optional_selectbox(
            "Ownership history",
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
            default="First Owner",
        )
        mileage = st.number_input(
            "Mileage (km/l)", min_value=5.0, max_value=45.0, value=None,
            step=0.1, placeholder="Not sure",
        )
        engine = st.number_input(
            "Engine (CC)", min_value=500, max_value=4_000, value=None,
            step=10, placeholder="Not sure",
        )
        max_power = st.number_input(
            "Maximum power (bhp)", min_value=20.0, max_value=450.0, value=None,
            step=1.0, placeholder="Not sure",
        )
        seats = st.number_input(
            "Seats", min_value=2, max_value=14, value=None, step=1,
            placeholder="Not sure",
        )

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    owner_mapping = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        UNKNOWN: np.nan,
    }
    car = pd.DataFrame([{
        "brand": missing_if_unknown(brand),
        "year": missing_if_unknown(year),
        "km_driven": missing_if_unknown(km_driven),
        "fuel": missing_if_unknown(fuel),
        "seller_type": missing_if_unknown(seller_type),
        "transmission": missing_if_unknown(transmission),
        "owner": owner_mapping[owner_label],
        "mileage": missing_if_unknown(mileage),
        "engine": missing_if_unknown(engine),
        "max_power": missing_if_unknown(max_power),
        "seats": missing_if_unknown(seats),
    }])

    try:
        predicted_log_price = model.predict(car)[0]
        predicted_price = float(np.exp(predicted_log_price))
        st.success(f"Estimated selling price: ₹{predicted_price:,.0f}")
        missing_count = int(car.isna().sum(axis=1).iloc[0])
        if missing_count:
            st.caption(
                f"The model automatically imputed {missing_count} missing "
                f"field{'s' if missing_count != 1 else ''}."
            )
        st.caption(
            "This is a model estimate in Indian rupees, not a guaranteed market valuation. "
            "Predictions are most reliable for cars similar to the training data."
        )
    except Exception as exc:
        st.error(f"Prediction failed. Please check the entered values. {exc}")

with st.expander("About the model"):
    st.write(
        "The app uses a Random Forest regression pipeline trained on the logarithm of selling "
        "price. The pipeline applies numeric preprocessing and one-hot encoding before prediction."
    )
