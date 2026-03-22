from __future__ import annotations

from typing import Any, Union
import __main__
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_FEATURE_COLUMNS = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]

REFERENCE_YEAR = 2021


def extract_first_float(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def parse_mileage_columns(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = series.fillna("").astype(str).str.lower().str.strip()
    mileage_value = extract_first_float(cleaned)
    mileage_unit = np.select(
        [
            cleaned.str.contains("km/kg", regex=False),
            cleaned.str.contains("kmpl", regex=False),
        ],
        [
            "km_per_kg",
            "km_per_liter",
        ],
        default="unknown",
    )
    return mileage_value, pd.Series(mileage_unit, index=series.index)


def parse_torque_columns(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    torque_value = extract_first_float(cleaned)
    is_kgm = cleaned.str.contains("kgm", regex=False)
    torque_nm = torque_value.where(~is_kgm, torque_value * 9.80665)

    rpm_match = cleaned.str.extract(r"(?:@|at)\s*([0-9]+)(?:\s*-\s*([0-9]+))?")
    torque_rpm_low = pd.to_numeric(rpm_match[0], errors="coerce")
    torque_rpm_high = pd.to_numeric(rpm_match[1], errors="coerce").fillna(torque_rpm_low)

    return torque_nm, torque_rpm_low, torque_rpm_high


def normalise_listing_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


class VehicleFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year: Union[int, str] = "auto", min_brand_count: int = 25):
        self.reference_year = reference_year
        self.min_brand_count = min_brand_count

    def fit(self, X, y=None):
        X = self._coerce_frame(X)
        years = pd.to_numeric(X["year"], errors="coerce")
        if self.reference_year == "auto":
            self.reference_year_ = int(years.max()) + 1
        else:
            self.reference_year_ = int(self.reference_year)

        brands = (
            X["name"]
            .fillna("Unknown")
            .astype(str)
            .str.split()
            .str[0]
            .fillna("Unknown")
        )
        brand_counts = brands.value_counts()
        self.kept_brands_ = set(brand_counts[brand_counts >= self.min_brand_count].index)
        return self

    def transform(self, X):
        X = self._coerce_frame(X)
        missing_columns = [col for col in RAW_FEATURE_COLUMNS if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing raw inference columns: {missing_columns}")

        listing_name = X["name"].fillna("Unknown").astype(str)
        brand_raw = listing_name.str.split().str[0].fillna("Unknown")
        brand = brand_raw.where(brand_raw.isin(self.kept_brands_), "Other")

        year = pd.to_numeric(X["year"], errors="coerce")
        km_driven = pd.to_numeric(X["km_driven"], errors="coerce")
        seats = pd.to_numeric(X["seats"], errors="coerce")
        engine_cc = extract_first_float(X["engine"])
        max_power_bhp = extract_first_float(X["max_power"])
        mileage_value, mileage_unit = parse_mileage_columns(X["mileage"])
        torque_nm, torque_rpm_low, torque_rpm_high = parse_torque_columns(X["torque"])

        owner = X["owner"].fillna("Unknown").astype(str)
        owner_map = {
            "Test Drive Car": 0,
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
        }
        owner_ordinal = owner.map(owner_map).fillna(4).astype(float)

        car_age = (self.reference_year_ - year).clip(lower=0)
        usage_years = car_age.replace(0, 1)
        km_per_year = km_driven / usage_years

        listing_title_clean = normalise_listing_text(listing_name)
        listing_name_length = listing_title_clean.str.len()
        listing_token_count = listing_title_clean.str.split().str.len()

        engine_nonzero = engine_cc.replace(0, np.nan)
        power_to_engine = max_power_bhp / engine_nonzero
        torque_to_engine = torque_nm / engine_nonzero
        torque_rpm_band = torque_rpm_high - torque_rpm_low

        return pd.DataFrame(
            {
                "listing_title_clean": listing_title_clean,
                "brand": brand,
                "fuel": X["fuel"].fillna("Unknown").astype(str),
                "seller_type": X["seller_type"].fillna("Unknown").astype(str),
                "transmission": X["transmission"].fillna("Unknown").astype(str),
                "owner": owner,
                "mileage_unit": mileage_unit,
                "year": year,
                "km_driven": km_driven,
                "seats": seats,
                "mileage_value": mileage_value,
                "engine_cc": engine_cc,
                "max_power_bhp": max_power_bhp,
                "torque_nm": torque_nm,
                "torque_rpm_low": torque_rpm_low,
                "torque_rpm_high": torque_rpm_high,
                "car_age": car_age,
                "km_per_year": km_per_year,
                "power_to_engine": power_to_engine,
                "torque_to_engine": torque_to_engine,
                "torque_rpm_band": torque_rpm_band,
                "listing_name_length": listing_name_length,
                "listing_token_count": listing_token_count,
                "owner_ordinal": owner_ordinal,
                "is_first_owner": (owner == "First Owner").astype(int),
                "is_automatic": (
                    X["transmission"].fillna("").astype(str).str.lower() == "automatic"
                ).astype(int),
                "mileage_missing": mileage_value.isna().astype(int),
                "engine_missing": engine_cc.isna().astype(int),
                "power_missing": max_power_bhp.isna().astype(int),
                "torque_missing": torque_nm.isna().astype(int),
                "seats_missing": seats.isna().astype(int),
                "log_km_driven": np.log1p(km_driven),
                "log_engine_cc": np.log1p(engine_cc),
            },
            index=X.index,
        )

    @staticmethod
    def _coerce_frame(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS).copy()


def build_parsed_summary(raw_input: dict[str, Any], reference_year: int = REFERENCE_YEAR) -> dict[str, Any]:
    frame = pd.DataFrame([raw_input], columns=RAW_FEATURE_COLUMNS)
    mileage_value, mileage_unit = parse_mileage_columns(frame["mileage"])
    torque_nm, torque_rpm_low, torque_rpm_high = parse_torque_columns(frame["torque"])
    engine_cc = extract_first_float(frame["engine"])
    max_power_bhp = extract_first_float(frame["max_power"])
    year = pd.to_numeric(frame["year"], errors="coerce")
    km_driven = pd.to_numeric(frame["km_driven"], errors="coerce")
    seats = pd.to_numeric(frame["seats"], errors="coerce")

    car_age = (reference_year - year).clip(lower=0)
    usage_years = car_age.replace(0, 1)
    km_per_year = km_driven / usage_years

    summary = {
        "brand": str(frame["name"].iloc[0]).split(" ")[0] if str(frame["name"].iloc[0]).strip() else "Unknown",
        "car_age_years": None if pd.isna(car_age.iloc[0]) else float(car_age.iloc[0]),
        "km_per_year": None if pd.isna(km_per_year.iloc[0]) else float(km_per_year.iloc[0]),
        "mileage_value": None if pd.isna(mileage_value.iloc[0]) else float(mileage_value.iloc[0]),
        "mileage_unit": str(mileage_unit.iloc[0]),
        "engine_cc": None if pd.isna(engine_cc.iloc[0]) else float(engine_cc.iloc[0]),
        "max_power_bhp": None if pd.isna(max_power_bhp.iloc[0]) else float(max_power_bhp.iloc[0]),
        "torque_nm": None if pd.isna(torque_nm.iloc[0]) else float(torque_nm.iloc[0]),
        "torque_rpm_low": None if pd.isna(torque_rpm_low.iloc[0]) else float(torque_rpm_low.iloc[0]),
        "torque_rpm_high": None if pd.isna(torque_rpm_high.iloc[0]) else float(torque_rpm_high.iloc[0]),
        "seats": None if pd.isna(seats.iloc[0]) else float(seats.iloc[0]),
        "parsed_title": normalise_listing_text(frame["name"]).iloc[0],
    }
    return summary


def register_pickle_shim() -> None:
    setattr(__main__, "VehicleFeatureEngineer", VehicleFeatureEngineer)
    main_module = sys.modules.get("main")
    if main_module is not None:
        setattr(main_module, "VehicleFeatureEngineer", VehicleFeatureEngineer)
