from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from model_pipeline import train_champion_model_from_csv
from vehicle_features import (
    RAW_FEATURE_COLUMNS,
    build_parsed_summary,
    register_pickle_shim,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "used_car_price_champion.joblib"
BENCHMARK_PATH = BASE_DIR / "reports" / "benchmark_results.csv"
FINAL_METRICS_PATH = BASE_DIR / "reports" / "final_test_metrics.csv"
SAMPLE_PAYLOAD_PATH = BASE_DIR / "artifacts" / "sample_inference_payload.json"
CONTRACT_PATH = BASE_DIR / "artifacts" / "inference_contract.json"
DATA_PATH = BASE_DIR / "Car details v3.csv"
INR_PER_GBP = 105.0

FUEL_OPTIONS = ["Diesel", "Petrol", "CNG", "LPG"]
SELLER_OPTIONS = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSION_OPTIONS = ["Manual", "Automatic"]
OWNER_OPTIONS = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]


st.set_page_config(
    page_title="Used Car Price Estimator",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(204, 97, 37, 0.10), transparent 28%),
                radial-gradient(circle at bottom left, rgba(22, 71, 77, 0.10), transparent 30%),
                linear-gradient(180deg, #f6f1e8 0%, #efe7db 100%);
            color: #132629;
        }

        html, body, [class*="css"] {
            font-family: "Space Grotesk", sans-serif;
        }

        h1, h2, h3 {
            font-family: "Source Serif 4", serif;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        .hero-shell {
            background: linear-gradient(135deg, #17353a 0%, #1f4c53 58%, #cc6125 100%);
            border: 1px solid rgba(19, 38, 41, 0.15);
            border-radius: 26px;
            padding: 1.45rem 1.55rem 1.3rem 1.55rem;
            color: #fffaf3;
            box-shadow: 0 22px 52px rgba(19, 38, 41, 0.18);
            margin-bottom: 0.8rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.77rem;
            opacity: 0.85;
            margin-bottom: 0.55rem;
        }

        .hero-shell h1 {
            margin: 0 0 0.45rem 0;
            font-size: 2.7rem;
            line-height: 1.02;
            color: #fffaf3;
        }

        .hero-shell p {
            margin: 0;
            max-width: 43rem;
            font-size: 0.98rem;
            line-height: 1.5;
            color: rgba(255, 250, 243, 0.88);
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-chip {
            border-radius: 999px;
            border: 1px solid rgba(255, 250, 243, 0.18);
            background: rgba(255, 250, 243, 0.10);
            padding: 0.4rem 0.7rem;
            font-size: 0.84rem;
            color: rgba(255, 250, 243, 0.92);
        }

        .stat-card,
        .panel-card,
        .prediction-card {
            border-radius: 22px;
            border: 1px solid rgba(19, 38, 41, 0.10);
            background: rgba(255, 250, 243, 0.84);
            box-shadow: 0 16px 40px rgba(19, 38, 41, 0.08);
        }

        .stat-card {
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 118px;
        }

        .stat-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #6d6b63;
            margin-bottom: 0.5rem;
        }

        .stat-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #132629;
            margin-bottom: 0.25rem;
        }

        .stat-subtext {
            font-size: 0.92rem;
            color: #415356;
        }

        .panel-card {
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            margin-bottom: 1rem;
        }

        .prediction-card {
            padding: 1.35rem;
            background: linear-gradient(180deg, rgba(255,250,243,0.96) 0%, rgba(249,238,225,0.98) 100%);
        }

        .prediction-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
            color: #6c6a64;
            margin-bottom: 0.6rem;
        }

        .prediction-value {
            font-family: "Source Serif 4", serif;
            font-size: 2.8rem;
            line-height: 1;
            color: #17353a;
            margin-bottom: 0.45rem;
        }

        .prediction-sub {
            font-size: 1rem;
            color: #37494c;
            margin-bottom: 0;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .mini-card {
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            background: rgba(23, 53, 58, 0.04);
            border: 1px solid rgba(19, 38, 41, 0.08);
        }

        .mini-label {
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #6d6b63;
            margin-bottom: 0.32rem;
        }

        .mini-value {
            font-size: 1rem;
            color: #17353a;
            font-weight: 700;
        }

        .section-title {
            font-family: "Source Serif 4", serif;
            font-size: 1.55rem;
            color: #17353a;
            margin: 0 0 0.55rem 0;
        }

        .spec-panel {
            padding: 1rem 1.05rem 0.95rem 1.05rem;
        }

        .spec-profile {
            font-size: 1.02rem;
            font-weight: 700;
            color: #17353a;
            margin-bottom: 0.7rem;
        }

        .spec-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
        }

        .spec-pill {
            border-radius: 16px;
            padding: 0.8rem 0.85rem;
            background: rgba(23, 53, 58, 0.04);
            border: 1px solid rgba(19, 38, 41, 0.08);
        }

        .spec-pill-label {
            display: block;
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #6d6b63;
            margin-bottom: 0.28rem;
        }

        .spec-pill-value {
            display: block;
            font-size: 1rem;
            color: #17353a;
            font-weight: 700;
        }

        .support-note {
            margin-top: 0.8rem;
            font-size: 0.92rem;
            color: #55686b;
        }

        .compact-note {
            font-size: 0.93rem;
            color: #55686b;
        }

        div[data-testid="stForm"] {
            border: none;
            background: rgba(255, 250, 243, 0.64);
            border-radius: 22px;
            padding: 0.35rem 0.2rem 0.2rem 0.2rem;
        }

        .streamlit-expanderHeader {
            font-weight: 600;
            color: #17353a;
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 999px;
            border: 1px solid rgba(19, 38, 41, 0.12);
            background: linear-gradient(135deg, #17353a 0%, #235861 100%);
            color: #fffaf3;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(23, 53, 58, 0.06);
            padding: 0.45rem 0.9rem;
        }

        @media (max-width: 1100px) {
            .spec-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .hero-shell h1 {
                font-size: 2.2rem;
            }

            .spec-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }

            .mini-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_samples() -> list[dict[str, Any]]:
    if not SAMPLE_PAYLOAD_PATH.exists():
        return []
    return json.loads(SAMPLE_PAYLOAD_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_contract() -> dict[str, Any]:
    if not CONTRACT_PATH.exists():
        return {"raw_feature_columns": RAW_FEATURE_COLUMNS}
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    if not FINAL_METRICS_PATH.exists():
        return pd.DataFrame(columns=["R2", "MAE", "RMSE"])
    return pd.read_csv(FINAL_METRICS_PATH, index_col=0)


@st.cache_data(show_spinner=False)
def load_benchmarks() -> pd.DataFrame:
    if not BENCHMARK_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(BENCHMARK_PATH)
    return df.sort_values("val_rmse", ascending=True).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dataset_snapshot() -> dict[str, Any]:
    if not DATA_PATH.exists():
        return {"raw_rows": None, "distinct_rows": None}
    df = pd.read_csv(DATA_PATH)
    return {
        "raw_rows": int(len(df)),
        "distinct_rows": int(len(df.drop_duplicates())),
    }


@st.cache_data(show_spinner=False)
def load_vehicle_catalog() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    df["brand"] = df["name"].fillna("Unknown").astype(str).str.split().str[0]
    df["seats_text"] = df["seats"].fillna("").astype(str)

    grouped = (
        df.groupby(
            [
                "brand",
                "name",
                "fuel",
                "transmission",
                "engine",
                "max_power",
                "torque",
                "seats_text",
                "mileage",
            ],
            dropna=False,
        )
        .agg(
            profile_count=("name", "size"),
            typical_year=("year", "median"),
            typical_km_driven=("km_driven", "median"),
            seller_mode=("seller_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "Individual"),
            owner_mode=("owner", lambda s: s.mode().iloc[0] if not s.mode().empty else "First Owner"),
        )
        .reset_index()
    )

    grouped["name_total_count"] = grouped.groupby(["brand", "name"])["profile_count"].transform("sum")
    grouped["spec_id"] = grouped.index.astype(str)
    grouped["spec_label"] = grouped.apply(
        lambda row: (
            f'{row["fuel"]} | {row["transmission"]} | {row["engine"]} | '
            f'{row["max_power"]} | {row["mileage"]}'
        ),
        axis=1,
    )
    grouped = grouped.sort_values(
        ["brand", "name_total_count", "name", "profile_count"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    return grouped


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    register_pickle_shim()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        print(f"Primary model artifact load failed: {exc}")
        print("Rebuilding champion model from CSV inside the deployment environment.")
        return train_champion_model_from_csv(DATA_PATH)


def format_price(value: float) -> str:
    return f"Rs {value:,.0f}"


def format_short_price(value: float) -> str:
    return f"{value / 100000:.2f} lakh"


def format_gbp(value: float) -> str:
    return f"GBP {value / INR_PER_GBP:,.0f}"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {format_price(abs(value))}"


def format_delta_gbp(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {format_gbp(abs(value))}"


def metric_value(metrics: pd.DataFrame, key: str) -> float:
    if metrics.empty:
        return 0.0
    return float(metrics.iloc[0][key])


def default_listing(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if samples:
        return samples[0].copy()
    return {
        "name": "Maruti Swift Dzire VDI",
        "year": 2014,
        "km_driven": 145500,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage": "23.4 kmpl",
        "engine": "1248 CC",
        "max_power": "74 bhp",
        "torque": "190Nm@ 2000rpm",
        "seats": 5.0,
    }


def shorten_text(text: str, max_chars: int = 28) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def example_button_label(sample: dict[str, Any], idx: int) -> str:
    name = str(sample.get("name", "")).strip()
    if not name:
        return f"Try Example {idx + 1}"
    core = " ".join(name.split()[:4])
    return f"Try {shorten_text(core, 26)}"


def example_button_caption(sample: dict[str, Any]) -> str:
    bits = []
    year = sample.get("year")
    if year:
        bits.append(str(int(year)))
    for key in ["fuel", "transmission"]:
        value = sample.get(key)
        if value:
            bits.append(str(value))
    return " | ".join(bits)


def _normalise_seats_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def sync_guided_selection_from_listing(listing: dict[str, Any], catalog: pd.DataFrame) -> None:
    brand = str(listing.get("name", "")).split(" ", 1)[0] if listing.get("name") else "Unknown"
    st.session_state["guided_brand"] = brand
    st.session_state["guided_name"] = listing.get("name", "")

    seats_text = _normalise_seats_text(listing.get("seats"))
    matches = catalog[
        (catalog["name"] == listing.get("name"))
        & (catalog["fuel"] == listing.get("fuel"))
        & (catalog["transmission"] == listing.get("transmission"))
        & (catalog["engine"] == listing.get("engine"))
        & (catalog["max_power"] == listing.get("max_power"))
        & (catalog["torque"] == listing.get("torque"))
        & (catalog["mileage"] == listing.get("mileage"))
        & (catalog["seats_text"].apply(_normalise_seats_text) == seats_text)
    ]

    if matches.empty:
        matches = catalog[catalog["name"] == listing.get("name")]

    if not matches.empty:
        spec_row = matches.iloc[0]
        st.session_state["guided_brand"] = spec_row["brand"]
        st.session_state["guided_name"] = spec_row["name"]
        st.session_state["guided_spec_id"] = spec_row["spec_id"]
        st.session_state["last_guided_spec_id"] = spec_row["spec_id"]


def apply_listing_to_state(listing: dict[str, Any], catalog: pd.DataFrame) -> None:
    for key in RAW_FEATURE_COLUMNS:
        st.session_state[f"input_{key}"] = listing.get(key)
    st.session_state["input_mode"] = "Guided selector"
    sync_guided_selection_from_listing(listing, catalog)
    st.session_state["prediction_result"] = None


def ensure_form_state(samples: list[dict[str, Any]], catalog: pd.DataFrame) -> None:
    listing = default_listing(samples)
    needs_guided_sync = any(
        key not in st.session_state
        for key in ["guided_brand", "guided_name", "guided_spec_id", "last_guided_spec_id"]
    )
    for key, value in listing.items():
        st.session_state.setdefault(f"input_{key}", value)
    st.session_state.setdefault("input_mode", "Guided selector")
    st.session_state.setdefault("prediction_result", None)
    st.session_state.setdefault("last_guided_spec_id", None)
    st.session_state.setdefault("guided_brand", listing["name"].split(" ", 1)[0])
    st.session_state.setdefault("guided_name", listing["name"])
    if needs_guided_sync:
        sync_guided_selection_from_listing(listing, catalog)


def get_brand_options(catalog: pd.DataFrame) -> list[str]:
    return sorted(catalog["brand"].dropna().unique().tolist())


def get_name_options(catalog: pd.DataFrame, brand: str) -> list[str]:
    brand_df = catalog[catalog["brand"] == brand]
    ordered_names = (
        brand_df[["name", "name_total_count"]]
        .drop_duplicates()
        .sort_values(["name_total_count", "name"], ascending=[False, True])["name"]
        .tolist()
    )
    return ordered_names


def ensure_guided_selection_state(catalog: pd.DataFrame) -> pd.Series:
    brand_options = get_brand_options(catalog)
    current_brand = st.session_state.get("guided_brand")
    if current_brand not in brand_options:
        current_brand = brand_options[0]
        st.session_state["guided_brand"] = current_brand

    name_options = get_name_options(catalog, current_brand)
    current_name = st.session_state.get("guided_name")
    if current_name not in name_options:
        current_name = name_options[0]
        st.session_state["guided_name"] = current_name

    spec_df = catalog[(catalog["brand"] == current_brand) & (catalog["name"] == current_name)].copy()
    spec_options = spec_df["spec_id"].tolist()
    current_spec_id = st.session_state.get("guided_spec_id")
    if current_spec_id not in spec_options:
        current_spec_id = spec_options[0]
        st.session_state["guided_spec_id"] = current_spec_id

    return spec_df.loc[spec_df["spec_id"] == current_spec_id].iloc[0]


def apply_guided_spec_defaults(spec_row: pd.Series) -> None:
    spec_id = spec_row["spec_id"]
    spec_changed = st.session_state.get("last_guided_spec_id") != spec_id

    st.session_state["input_name"] = spec_row["name"]
    st.session_state["input_fuel"] = spec_row["fuel"]
    st.session_state["input_transmission"] = spec_row["transmission"]
    st.session_state["input_engine"] = spec_row["engine"]
    st.session_state["input_max_power"] = spec_row["max_power"]
    st.session_state["input_torque"] = spec_row["torque"]
    st.session_state["input_mileage"] = spec_row["mileage"]

    try:
        st.session_state["input_seats"] = float(spec_row["seats_text"])
    except Exception:
        pass

    if spec_changed:
        st.session_state["input_year"] = int(round(float(spec_row["typical_year"])))
        st.session_state["input_km_driven"] = int(round(float(spec_row["typical_km_driven"])))
        if spec_row["seller_mode"] in SELLER_OPTIONS:
            st.session_state["input_seller_type"] = spec_row["seller_mode"]
        if spec_row["owner_mode"] in OWNER_OPTIONS:
            st.session_state["input_owner"] = spec_row["owner_mode"]

    st.session_state["last_guided_spec_id"] = spec_id


def collect_form_input() -> dict[str, Any]:
    return {
        "name": str(st.session_state["input_name"]).strip(),
        "year": int(st.session_state["input_year"]),
        "km_driven": int(st.session_state["input_km_driven"]),
        "fuel": st.session_state["input_fuel"],
        "seller_type": st.session_state["input_seller_type"],
        "transmission": st.session_state["input_transmission"],
        "owner": st.session_state["input_owner"],
        "mileage": str(st.session_state["input_mileage"]).strip(),
        "engine": str(st.session_state["input_engine"]).strip(),
        "max_power": str(st.session_state["input_max_power"]).strip(),
        "torque": str(st.session_state["input_torque"]).strip(),
        "seats": float(st.session_state["input_seats"]),
    }


def validate_listing(listing: dict[str, Any]) -> list[str]:
    errors = []
    if not listing["name"]:
        errors.append("Listing title is required.")
    if listing["year"] < 1980 or listing["year"] > 2035:
        errors.append("Year should be between 1980 and 2035.")
    if listing["km_driven"] < 0:
        errors.append("Kilometres driven cannot be negative.")
    if listing["seats"] <= 0:
        errors.append("Seats must be greater than zero.")
    return errors


def render_autofilled_spec_panel(listing: dict[str, Any]) -> None:
    spec_items = [
        ("Fuel", listing["fuel"]),
        ("Transmission", listing["transmission"]),
        ("Seats", f"{_normalise_seats_text(listing['seats'])} seats"),
        ("Mileage", listing["mileage"]),
        ("Engine", listing["engine"]),
        ("Max power", listing["max_power"]),
        ("Torque", listing["torque"]),
    ]
    pill_html = "".join(
        f"""
        <div class="spec-pill">
            <span class="spec-pill-label">{escape(label)}</span>
            <span class="spec-pill-value">{escape(str(value))}</span>
        </div>
        """
        for label, value in spec_items
    )
    st.markdown(
        f"""
        <div class="panel-card spec-panel">
            <div class="mini-label">Auto-filled technical profile</div>
            <div class="spec-profile">{escape(str(listing["name"]))}</div>
            <div class="spec-grid">{pill_html}</div>
            <div class="support-note">
                These technical values come from the selected catalogue profile, so users do not have to type fragile units or torque strings by hand.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_input_frame(listing: dict[str, Any], contract: dict[str, Any]) -> pd.DataFrame:
    columns = contract.get("raw_feature_columns", RAW_FEATURE_COLUMNS)
    return pd.DataFrame([listing], columns=columns)


def make_prediction(listing: dict[str, Any], model, contract: dict[str, Any], metrics: pd.DataFrame) -> dict[str, Any]:
    prediction = float(model.predict(build_input_frame(listing, contract))[0])
    parsed_summary = build_parsed_summary(listing)
    mae = metric_value(metrics, "MAE")
    rmse = metric_value(metrics, "RMSE")

    return {
        "raw_input": listing,
        "prediction": prediction,
        "band_low": max(prediction - mae, 0),
        "band_high": prediction + mae,
        "parsed_summary": parsed_summary,
        "test_mae": mae,
        "test_rmse": rmse,
    }


def render_stat_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(result: dict[str, Any], metrics: pd.DataFrame) -> None:
    r2 = metric_value(metrics, "R2")
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Resale Price</div>
            <div class="prediction-value">{format_gbp(result["prediction"])}</div>
            <p class="prediction-sub">Reference display: {format_price(result["prediction"])} | {format_short_price(result["prediction"])}</p>
            <p class="prediction-sub">Typical error band: {format_gbp(result["band_low"])} to {format_gbp(result["band_high"])}</p>
            <p class="prediction-sub">Reference exchange rate: 1 GBP = {INR_PER_GBP:.0f} INR</p>
            <p class="prediction-sub">Champion model test score: R2 {r2:.3f} | MAE {format_gbp(result["test_mae"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_parsed_summary(summary: dict[str, Any]) -> None:
    def render_value(value: Any, suffix: str = "") -> str:
        if value is None:
            return "Not parsed"
        if isinstance(value, float):
            if suffix in {" years", " seats"}:
                return f"{value:,.0f}{suffix}"
            return f"{value:,.1f}{suffix}"
        return f"{value}{suffix}"

    cols = st.columns(2)
    cards = [
        ("Brand", str(summary["brand"])),
        ("Car Age", render_value(summary["car_age_years"], " years")),
        ("Km / Year", render_value(summary["km_per_year"])),
        ("Mileage", render_value(summary["mileage_value"]) + f" ({summary['mileage_unit']})"),
        ("Engine", render_value(summary["engine_cc"], " cc")),
        ("Max Power", render_value(summary["max_power_bhp"], " bhp")),
        ("Torque", render_value(summary["torque_nm"], " Nm")),
        ("Seats", render_value(summary["seats"], " seats")),
    ]
    for idx, (label, value) in enumerate(cards):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-label">{label}</div>
                    <div class="mini-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_prediction_state() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">Prediction Desk</div>
            <p>Choose a vehicle profile, adjust year and mileage, and run the estimate. Guided mode keeps the hardest technical fields out of the user’s way.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    samples = load_samples()
    contract = load_contract()
    metrics = load_metrics()
    benchmark_df = load_benchmarks()
    dataset_snapshot = load_dataset_snapshot()
    vehicle_catalog = load_vehicle_catalog()

    ensure_form_state(samples, vehicle_catalog)

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Portfolio ML Product</div>
            <h1>Used Car Price Estimator</h1>
            <p>
                Choose a real vehicle profile, adjust the ownership and usage details, and get a resale estimate
                without typing fragile technical specs by hand.
            </p>
            <div class="hero-meta">
                <div class="hero-chip">1. Choose brand and profile</div>
                <div class="hero-chip">2. Adjust year and kilometres</div>
                <div class="hero-chip">3. Review the GBP estimate</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"GBP is the primary display for this app. Source prices come from India-market resale listings and are shown with INR and lakh as supporting references at 1 GBP = {INR_PER_GBP:.0f} INR."
    )

    quick_cols = st.columns(4)
    if samples:
        for idx, sample in enumerate(samples[:3]):
            with quick_cols[idx]:
                if st.button(example_button_label(sample, idx), use_container_width=True):
                    apply_listing_to_state(sample, vehicle_catalog)
                    st.rerun()
                st.caption(example_button_caption(sample))
    with quick_cols[3]:
        if st.button("Reset Form", use_container_width=True):
            apply_listing_to_state(default_listing(samples), vehicle_catalog)
            st.rerun()
        st.caption("Return to the default guided profile.")

    with st.expander("Model confidence and dataset coverage", expanded=False):
        stat_cols = st.columns(4)
        with stat_cols[0]:
            render_stat_card("Champion", "HistGradientBoosting", "Selected after validation-based tuning.")
        with stat_cols[1]:
            render_stat_card("Held-Out Test R2", f"{metric_value(metrics, 'R2'):.3f}", "Generalisation on untouched data.")
        with stat_cols[2]:
            render_stat_card("Held-Out Test MAE", format_price(metric_value(metrics, "MAE")), "Typical absolute miss size.")
        with stat_cols[3]:
            distinct_rows = dataset_snapshot.get("distinct_rows")
            render_stat_card("Distinct Listings", f"{distinct_rows:,}" if distinct_rows else "n/a", "Rows kept after deduplication.")
        st.caption("This keeps the portfolio evidence accessible without slowing down the main valuation flow.")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Vehicle Details</div>', unsafe_allow_html=True)
        st.radio(
            "Input mode",
            options=["Guided selector", "Manual entry"],
            horizontal=True,
            key="input_mode",
            help="Guided selector auto-fills technical specs from the dataset. Switch to manual only when your exact vehicle profile is not available.",
        )

        selected_listing = None

        if st.session_state["input_mode"] == "Guided selector":
            st.caption("Choose the car from a searchable catalogue and let the app fill in the fragile technical fields for you.")

            ensure_guided_selection_state(vehicle_catalog)

            brand_options = get_brand_options(vehicle_catalog)
            st.selectbox("Brand", options=brand_options, key="guided_brand")

            name_options = get_name_options(vehicle_catalog, st.session_state["guided_brand"])
            st.selectbox(
                "Vehicle profile",
                options=name_options,
                key="guided_name",
                help="Type in the dropdown to search trims and variants.",
            )

            spec_df = vehicle_catalog[
                (vehicle_catalog["brand"] == st.session_state["guided_brand"])
                & (vehicle_catalog["name"] == st.session_state["guided_name"])
            ].copy()
            spec_label_map = dict(zip(spec_df["spec_id"], spec_df["spec_label"]))

            if st.session_state.get("guided_spec_id") not in spec_df["spec_id"].tolist():
                st.session_state["guided_spec_id"] = spec_df.iloc[0]["spec_id"]

            if len(spec_df) > 1:
                st.selectbox(
                    "Spec preset",
                    options=spec_df["spec_id"].tolist(),
                    format_func=lambda spec_id: spec_label_map[spec_id],
                    key="guided_spec_id",
                    help="Some titles appear with multiple technical configurations. Pick the closest preset.",
                )
            else:
                st.caption(f"Auto-filled spec: {spec_df.iloc[0]['spec_label']}")
                st.session_state["guided_spec_id"] = spec_df.iloc[0]["spec_id"]

            selected_spec = spec_df.loc[spec_df["spec_id"] == st.session_state["guided_spec_id"]].iloc[0]
            apply_guided_spec_defaults(selected_spec)

            row_a = st.columns(2)
            row_a[0].number_input("Year", min_value=1980, max_value=2035, step=1, key="input_year")
            row_a[1].number_input("Km driven", min_value=0, step=1000, key="input_km_driven")

            row_owner = st.columns(2)
            row_owner[0].selectbox("Seller type", SELLER_OPTIONS, key="input_seller_type")
            row_owner[1].selectbox("Owner history", OWNER_OPTIONS, key="input_owner")

            selected_listing = {
                "name": str(selected_spec["name"]).strip(),
                "year": int(st.session_state["input_year"]),
                "km_driven": int(st.session_state["input_km_driven"]),
                "fuel": str(selected_spec["fuel"]),
                "seller_type": st.session_state["input_seller_type"],
                "transmission": str(selected_spec["transmission"]),
                "owner": st.session_state["input_owner"],
                "mileage": str(selected_spec["mileage"]).strip(),
                "engine": str(selected_spec["engine"]).strip(),
                "max_power": str(selected_spec["max_power"]).strip(),
                "torque": str(selected_spec["torque"]).strip(),
                "seats": float(selected_spec["seats_text"]),
            }
            render_autofilled_spec_panel(selected_listing)
        else:
            st.caption("Manual entry is still available, but it is best reserved for cars not covered by the guided catalogue.")

            st.text_input(
                "Listing title",
                key="input_name",
                help="Example: Mahindra XUV500 AT W9 2WD",
                placeholder="Mahindra XUV500 AT W9 2WD",
            )

            row_a = st.columns(3)
            row_a[0].number_input("Year", min_value=1980, max_value=2035, step=1, key="input_year")
            row_a[1].number_input("Km driven", min_value=0, step=1000, key="input_km_driven")
            row_a[2].number_input("Seats", min_value=2.0, max_value=10.0, step=1.0, key="input_seats")

            row_b = st.columns(2)
            row_b[0].selectbox("Fuel", FUEL_OPTIONS, key="input_fuel")
            row_b[1].selectbox("Transmission", TRANSMISSION_OPTIONS, key="input_transmission")

            row_c = st.columns(2)
            row_c[0].selectbox("Seller type", SELLER_OPTIONS, key="input_seller_type")
            row_c[1].selectbox("Owner history", OWNER_OPTIONS, key="input_owner")

            row_d = st.columns(2)
            row_d[0].text_input(
                "Mileage",
                key="input_mileage",
                placeholder="16.5 kmpl",
                help="Examples: 16.5 kmpl, 17.3 km/kg",
            )
            row_d[1].text_input(
                "Engine",
                key="input_engine",
                placeholder="1493 CC",
                help="Example: 1493 CC",
            )

            row_e = st.columns(2)
            row_e[0].text_input(
                "Max power",
                key="input_max_power",
                placeholder="70 bhp",
                help="Example: 103.52 bhp",
            )
            row_e[1].text_input(
                "Torque",
                key="input_torque",
                placeholder="195Nm@ 1400-2200rpm",
                help="Example: 250Nm@ 1500-2500rpm",
            )

            selected_listing = collect_form_input()

        submitted = st.button("Estimate Resale Price", use_container_width=True)

        if submitted:
            listing = selected_listing if selected_listing is not None else collect_form_input()
            errors = validate_listing(listing)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    model = load_model()
                    st.session_state["prediction_result"] = make_prediction(listing, model, contract, metrics)
                    st.success("Model scored the listing successfully.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with right_col:
        result = st.session_state.get("prediction_result")
        if result is None:
            render_empty_prediction_state()
        else:
            render_prediction_card(result, metrics)
            st.write("")
            st.markdown('<div class="section-title">Vehicle Summary</div>', unsafe_allow_html=True)
            render_parsed_summary(result["parsed_summary"])
            st.caption("These values are parsed from the chosen profile and your input before the model scores the listing.")

    result = st.session_state.get("prediction_result")
    tab_predict, tab_whatif, tab_model, tab_notes = st.tabs(
        ["How The Car Was Read", "Price Sensitivity", "Model Quality", "About The App"]
    )

    with tab_predict:
        if result is None:
            st.info("Run a prediction first to inspect the parsed feature summary and raw input payload.")
        else:
            raw_df = pd.DataFrame([result["raw_input"]])
            parsed_df = pd.DataFrame([result["parsed_summary"]]).T.rename(columns={0: "value"})
            st.caption("Use this section to verify exactly what the app sent into the model and how it parsed the selected vehicle.")
            with st.expander("View raw vehicle payload", expanded=False):
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
            with st.expander("View parsed feature table", expanded=False):
                st.dataframe(parsed_df, use_container_width=True)

    with tab_whatif:
        if result is None:
            st.info("Run a base prediction first, then explore how mileage, year, and ownership change the estimate.")
        else:
            base_input = result["raw_input"].copy()
            st.markdown("#### Scenario explorer")
            scenario_cols = st.columns(3)
            scenario_year = scenario_cols[0].slider(
                "Model year",
                min_value=1990,
                max_value=2035,
                value=int(base_input["year"]),
                step=1,
            )
            scenario_km = scenario_cols[1].slider(
                "Km driven",
                min_value=0,
                max_value=300000,
                value=int(base_input["km_driven"]),
                step=5000,
            )
            scenario_owner = scenario_cols[2].selectbox(
                "Owner history",
                OWNER_OPTIONS,
                index=OWNER_OPTIONS.index(base_input["owner"]),
            )

            scenario_input = base_input.copy()
            scenario_input["year"] = scenario_year
            scenario_input["km_driven"] = scenario_km
            scenario_input["owner"] = scenario_owner

            try:
                scenario_prediction = float(load_model().predict(build_input_frame(scenario_input, contract))[0])
                scenario_delta = scenario_prediction - result["prediction"]
                metric_cols = st.columns(3)
                metric_cols[0].metric("Base estimate", format_gbp(result["prediction"]))
                metric_cols[1].metric("Scenario estimate", format_gbp(scenario_prediction), delta=format_delta_gbp(scenario_delta))
                metric_cols[2].metric(
                    "Reference INR",
                    format_price(scenario_prediction),
                    delta=format_delta(scenario_delta),
                )
                st.caption(
                    f"This section reuses the same saved pipeline. Primary display is GBP at 1 GBP = {INR_PER_GBP:.0f} INR. Secondary reference: {format_short_price(scenario_prediction)}."
                )
            except Exception as exc:
                st.error(f"Scenario analysis failed: {exc}")

    with tab_model:
        st.caption("This section is for reviewers, recruiters, and anyone who wants to inspect the model evidence behind the estimate.")
        stat_cols = st.columns(4)
        with stat_cols[0]:
            render_stat_card("Champion", "HistGradientBoosting", "Best validation result after tuning.")
        with stat_cols[1]:
            render_stat_card("Held-Out Test R2", f"{metric_value(metrics, 'R2'):.3f}", "Primary generalisation score.")
        with stat_cols[2]:
            render_stat_card("Held-Out Test MAE", format_price(metric_value(metrics, "MAE")), "Typical error in source currency.")
        with stat_cols[3]:
            distinct_rows = dataset_snapshot.get("distinct_rows")
            render_stat_card("Distinct Listings", f"{distinct_rows:,}" if distinct_rows else "n/a", "Deduplicated training coverage.")
        st.write("")
        st.markdown("#### Benchmark ladder")
        if benchmark_df.empty:
            st.warning("Benchmark report not found.")
        else:
            chart_df = benchmark_df.set_index("model")[["val_rmse"]]
            st.bar_chart(chart_df)
            with st.expander("View full benchmark table", expanded=False):
                benchmark_view = benchmark_df.copy()
                benchmark_view["val_rmse"] = benchmark_view["val_rmse"].round(0)
                benchmark_view["val_mae"] = benchmark_view["val_mae"].round(0)
                benchmark_view["val_r2"] = benchmark_view["val_r2"].round(3)
                st.dataframe(
                    benchmark_view[["model", "val_r2", "val_mae", "val_rmse", "cv_rmse_mean"]],
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption("Validation RMSE is lower-is-better. HistGradientBoosting became the production choice after targeted tuning.")

    with tab_notes:
        st.markdown("#### What this app is designed to do")
        st.markdown(
            """
            - Help a user estimate a likely resale price quickly without typing error-prone technical strings.
            - Keep the same training-time feature engineering inside the deployed pipeline so the app and model stay aligned.
            - Surface a portfolio-grade level of model evidence without overwhelming the main valuation flow.
            - Show GBP first for readability, while keeping INR and lakh visible as source-data references.
            """
        )
        st.markdown("#### Limits to keep in mind")
        st.markdown(
            """
            - The training data reflects India-market used-car listings, so the app is best treated as a guided estimator rather than a dealer-grade quote engine.
            - Rare fuels and unusually expensive trims are less stable than common diesel and petrol listings.
            - GBP values use a fixed reference rate, not a live FX feed.
            """
        )
        with st.expander("Files and artifacts used by the app", expanded=False):
            st.code(
                "\n".join(
                    [
                        str(MODEL_PATH),
                        str(CONTRACT_PATH),
                        str(SAMPLE_PAYLOAD_PATH),
                        str(FINAL_METRICS_PATH),
                        str(BENCHMARK_PATH),
                    ]
                )
            )


if __name__ == "__main__":
    main()
