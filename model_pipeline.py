from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from vehicle_features import RAW_FEATURE_COLUMNS, VehicleFeatureEngineer


RANDOM_STATE = 42
TARGET_COLUMN = "selling_price"

NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "seats",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "torque_rpm_low",
    "torque_rpm_high",
    "car_age",
    "km_per_year",
    "power_to_engine",
    "torque_to_engine",
    "torque_rpm_band",
    "listing_name_length",
    "listing_token_count",
    "owner_ordinal",
    "is_first_owner",
    "is_automatic",
    "mileage_missing",
    "engine_missing",
    "power_missing",
    "torque_missing",
    "seats_missing",
    "log_km_driven",
    "log_engine_cc",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage_unit",
]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_price_bins(series: pd.Series, q: int = 10) -> pd.Series:
    usable_bins = int(min(q, series.nunique()))
    return pd.qcut(series, q=usable_bins, duplicates="drop")


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.drop_duplicates().copy()
    model_df[TARGET_COLUMN] = pd.to_numeric(model_df[TARGET_COLUMN], errors="coerce")
    model_df = model_df.loc[model_df[TARGET_COLUMN].notna() & (model_df[TARGET_COLUMN] > 0)].reset_index(drop=True)
    return model_df


def build_champion_pipeline():
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    regressor = Pipeline(
        steps=[
            ("feature_engineer", VehicleFeatureEngineer(min_brand_count=30)),
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.1,
                    max_iter=250,
                    max_leaf_nodes=63,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_champion_model_from_csv(data_path: Path):
    raw_df = pd.read_csv(data_path)
    model_df = clean_training_frame(raw_df)

    X = model_df[RAW_FEATURE_COLUMNS].copy()
    y = model_df[TARGET_COLUMN].astype(float).copy()

    price_bins = make_price_bins(y, q=10)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=price_bins,
    )

    train_bins = make_price_bins(y_train_full, q=10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=train_bins,
    )

    X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_dev = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    model = build_champion_pipeline()
    model.fit(X_dev, y_dev)
    return model
