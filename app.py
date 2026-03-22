from __future__ import annotations

import json
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
            padding: 1.8rem 1.7rem 1.5rem 1.7rem;
            color: #fffaf3;
            box-shadow: 0 22px 52px rgba(19, 38, 41, 0.18);
            margin-bottom: 1.1rem;
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
            font-size: 3rem;
            line-height: 1.02;
            color: #fffaf3;
        }

        .hero-shell p {
            margin: 0;
            max-width: 50rem;
            font-size: 1rem;
            line-height: 1.55;
            color: rgba(255, 250, 243, 0.88);
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

        div[data-testid="stForm"] {
            border: none;
            background: rgba(255, 250, 243, 0.64);
            border-radius: 22px;
            padding: 0.35rem 0.2rem 0.2rem 0.2rem;
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


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {format_price(abs(value))}"


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


def apply_listing_to_state(listing: dict[str, Any]) -> None:
    for key in RAW_FEATURE_COLUMNS:
        st.session_state[f"input_{key}"] = listing.get(key)
    st.session_state["prediction_result"] = None


def ensure_form_state(samples: list[dict[str, Any]]) -> None:
    listing = default_listing(samples)
    for key, value in listing.items():
        st.session_state.setdefault(f"input_{key}", value)
    st.session_state.setdefault("prediction_result", None)


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
            <div class="prediction-value">{format_price(result["prediction"])}</div>
            <p class="prediction-sub">{format_short_price(result["prediction"])} in the dataset's original price scale.</p>
            <p class="prediction-sub">Typical error band from held-out testing: {format_price(result["band_low"])} to {format_price(result["band_high"])}</p>
            <p class="prediction-sub">Champion model test score: R2 {r2:.3f} | MAE {format_price(result["test_mae"])}</p>
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
            <p>Enter a realistic used-car listing, then run the model to see the estimated selling price, parsed vehicle summary, what-if sensitivity, and benchmark context.</p>
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

    ensure_form_state(samples)

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Portfolio ML Product</div>
            <h1>Used Car Price Estimator</h1>
            <p>
                A polished Streamlit frontend for the champion resale-price model. The app accepts raw listing fields,
                runs the same training-time feature engineering under the hood, and returns a deployable price estimate
                with model context instead of a bare number.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.write("")

    quick_cols = st.columns(4)
    if samples:
        for idx, sample in enumerate(samples[:3]):
            with quick_cols[idx]:
                if st.button(f"Load Example {idx + 1}", use_container_width=True):
                    apply_listing_to_state(sample)
                    st.rerun()
    with quick_cols[3]:
        if st.button("Reset Form", use_container_width=True):
            apply_listing_to_state(default_listing(samples))
            st.rerun()

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Vehicle Details</div>', unsafe_allow_html=True)
        st.caption("Keep the fields close to a real listing format so the trained parser can read them well.")

        with st.form("prediction_form", clear_on_submit=False):
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

            submitted = st.form_submit_button("Estimate Resale Price", use_container_width=True)

        if submitted:
            listing = collect_form_input()
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
            st.markdown('<div class="section-title">Parsed Feature Snapshot</div>', unsafe_allow_html=True)
            render_parsed_summary(result["parsed_summary"])
            st.caption("These values are derived from the raw form input before the model scores the listing.")

    result = st.session_state.get("prediction_result")
    tab_predict, tab_whatif, tab_model, tab_notes = st.tabs(
        ["Parsed Details", "What-If Analysis", "Model Benchmarks", "Project Notes"]
    )

    with tab_predict:
        if result is None:
            st.info("Run a prediction first to inspect the parsed feature summary and raw input payload.")
        else:
            raw_df = pd.DataFrame([result["raw_input"]])
            parsed_df = pd.DataFrame([result["parsed_summary"]]).T.rename(columns={0: "value"})
            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### Raw input payload")
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
            with right:
                st.markdown("#### Derived view")
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
                metric_cols[0].metric("Base estimate", format_price(result["prediction"]))
                metric_cols[1].metric("Scenario estimate", format_price(scenario_prediction), delta=format_delta(scenario_delta))
                metric_cols[2].metric(
                    "Shift in lakh terms",
                    format_short_price(scenario_prediction),
                    delta=f"{scenario_delta / 100000:+.2f} lakh",
                )
                st.caption("This section reuses the same saved pipeline, so the what-if output stays aligned with the model artifact.")
            except Exception as exc:
                st.error(f"Scenario analysis failed: {exc}")

    with tab_model:
        st.markdown("#### Benchmark ladder")
        if benchmark_df.empty:
            st.warning("Benchmark report not found.")
        else:
            benchmark_view = benchmark_df.copy()
            benchmark_view["val_rmse"] = benchmark_view["val_rmse"].round(0)
            benchmark_view["val_mae"] = benchmark_view["val_mae"].round(0)
            benchmark_view["val_r2"] = benchmark_view["val_r2"].round(3)
            st.dataframe(
                benchmark_view[["model", "val_r2", "val_mae", "val_rmse", "cv_rmse_mean"]],
                use_container_width=True,
                hide_index=True,
            )
            chart_df = benchmark_df.set_index("model")[["val_rmse"]]
            st.bar_chart(chart_df)
            st.caption("Validation RMSE is lower-is-better. HistGradientBoosting became the production choice after tuning.")

    with tab_notes:
        st.markdown("#### Project framing")
        st.markdown(
            """
            - The notebook compares linear, regularised, tree-ensemble, and boosting regressors on the same deduplicated split.
            - The champion model is trained on raw listing inputs, then handles parsing and feature engineering internally.
            - The held-out test score is strong enough for a portfolio demo, but rare fuels and very expensive automatic listings still need caution.
            - Prices are displayed in the dataset's original numeric scale, which corresponds to Indian rupees.
            """
        )
        st.markdown("#### Files used by this app")
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
