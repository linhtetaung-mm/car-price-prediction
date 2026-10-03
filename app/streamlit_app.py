"""Run from repository root: streamlit run app/streamlit_app.py"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from a3.data import PRICE_LABELS

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
            "Predict price category", type="primary", width="stretch"
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


st.set_page_config(page_title="Car Price Categories | A3", page_icon="🚗", layout="centered")
st.title("Used Car Price Category")
st.write("Predict one of four selling-price bands from a car's specifications.")
st.caption("Assignment 3 · Multinomial logistic regression · Prices in Indian rupees")

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/best_a3_model.joblib")

try:
    model = load_model()
except Exception as exc:
    st.error(f"The trained A3 model could not be loaded: {exc}")
    st.stop()

with st.expander("Price bands and how to use this model"):
    for c, label in enumerate(PRICE_LABELS):
        st.write(f"Class {c}: {label}")
    st.write("Each upper boundary belongs to the lower band. Leave unknown fields blank; "
             "the fitted preprocessing pipeline fills them with training-set values. "
             "This predicts a category, so it does not provide an exact price.")
submitted, row = prediction_form("a3")
if submitted:
    try:
        probabilities = model.predict_proba(row)[0]
        predicted = int(probabilities.argmax())
        st.success(f"Class {predicted} — {PRICE_LABELS[predicted]}")
        st.metric("Model probability", f"{probabilities[predicted]:.1%}")
        st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=PRICE_LABELS))
        missing_count = int(row.isna().sum(axis=1).iloc[0])
        if missing_count:
            st.caption(f"Filled {missing_count} missing fields using training-set values.")
        st.caption("Probabilities are model estimates and have not been calibrated. "
                   "Actual market prices may differ.")
    except (ValueError, TypeError) as exc:
        st.error(f"Please check the supplied car details: {exc}")
