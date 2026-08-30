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

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        brand = st.selectbox("Brand", brands, index=brands.index("Maruti"))
        year = st.number_input("Year", min_value=1983, max_value=2026, value=2018, step=1)
        km_driven = st.number_input(
            "Kilometres driven", min_value=0, max_value=2_500_000, value=45_000, step=1_000
        )
        fuel = st.selectbox("Fuel", ["Diesel", "Petrol"], index=1)
        seller_type = st.selectbox(
            "Seller type", ["Dealer", "Individual", "Trustmark Dealer"], index=1
        )
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"], index=1)

    with right:
        owner_label = st.selectbox(
            "Ownership history",
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
        )
        mileage = st.number_input(
            "Mileage (km/l)", min_value=5.0, max_value=45.0, value=20.0, step=0.1
        )
        engine = st.number_input(
            "Engine (CC)", min_value=500, max_value=4_000, value=1_197, step=10
        )
        max_power = st.number_input(
            "Maximum power (bhp)", min_value=20.0, max_value=450.0, value=82.0, step=1.0
        )
        seats = st.number_input("Seats", min_value=2, max_value=14, value=5, step=1)

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    owner_mapping = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
    }
    car = pd.DataFrame([{
        "brand": brand,
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner_mapping[owner_label],
        "mileage": float(mileage),
        "engine": float(engine),
        "max_power": float(max_power),
        "seats": float(seats),
    }])

    try:
        predicted_log_price = model.predict(car)[0]
        predicted_price = float(np.exp(predicted_log_price))
        st.success(f"Estimated selling price: ₹{predicted_price:,.0f}")
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
