"""A1/A2 cleaning and fold-local preprocessing, reused without changing its rules."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

NUMERIC = ["year", "km_driven", "owner", "mileage", "engine", "max_power", "seats"]
CATEGORICAL = ["brand", "fuel", "seller_type", "transmission"]
FEATURES = NUMERIC + CATEGORICAL
# Fixed, interpretable rupee thresholds: not estimated using held-out test prices.
PRICE_EDGES = [0, 300_000, 600_000, 1_000_000, float("inf")]
PRICE_LABELS = ["Budget: up to ₹300,000", "Mid-range: ₹300,000–600,000",
                "Upper mid-range: ₹600,000–1,000,000", "Premium: above ₹1,000,000"]


def price_classes(prices):
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or not len(prices) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Selling prices must be finite, positive, and one-dimensional")
    return np.asarray(pd.cut(prices, bins=PRICE_EDGES, labels=False, right=True), dtype=int)


def clean_data(frame):
    # Repeat the attached Assignment 1 notebook's cleaning order exactly.
    data = frame.copy()
    data = data.loc[~data["fuel"].isin(["CNG", "LPG"])]
    data["owner"] = data["owner"].map(
        {
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
            "Test Drive Car": 5,
        }
    )
    for column in ["mileage", "engine", "max_power"]:
        data[column] = pd.to_numeric(
            data[column].astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False),
            errors="coerce",
        )
    data = data.rename(columns={"name": "brand"})
    data["brand"] = data["brand"].astype("string").str.split().str[0]
    data = data.drop(columns=["torque"])
    data = data.loc[data["owner"] != 5].copy()

    # Deleted the small group of rows missing any of these four fields.
    required_columns = ["mileage", "engine", "max_power", "seats"]
    missing_mask = data[required_columns].isnull().any(axis=1)
    data = data.loc[~missing_mask].copy().reset_index(drop=True)

    # Deduplication happens after parsing and brand extraction in A1.
    return data.drop_duplicates().reset_index(drop=True).copy()


def make_preprocessor(polynomial=False):
    """Create fold-local imputation, scaling, encoding, and optional polynomial terms."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
    if polynomial:
        numeric_steps.extend(
            [
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("polynomial_scaler", StandardScaler()),
            ]
        )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )

