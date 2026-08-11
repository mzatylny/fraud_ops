from __future__ import annotations

import json
import logging
import time

import pandas as pd
import streamlit as st

from src.config import METRICS_CSV, MONITORING_BASELINE_JSON, THRESHOLD_CSV, TRANSACTIONS_CSV
from src.database import (
    analytics_snapshot,
    get_review_cases,
    get_review_history,
    get_transactions,
    init_db,
    update_transaction_status,
)
from src.features import reset_feature_store
from src.monitoring import evaluate_operations
from src.stream_processor import process_transaction

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Fraud Operations Platform",
    page_icon="🛡️",
    layout="wide",
)
st.title("Fraud Operations Platform")
st.caption(
    "Handbook-inspired simulation → online behavioural features → two-stage detection "
    "→ operational decision → analyst feedback"
)


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    if not TRANSACTIONS_CSV.exists():
        return pd.DataFrame()
    frame = pd.read_csv(TRANSACTIONS_CSV)
    frame["tx_datetime"] = pd.to_datetime(frame["tx_datetime"])
    return frame.sort_values(["tx_datetime", "transaction_id"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_thresholds() -> pd.DataFrame:
    return pd.read_csv(THRESHOLD_CSV) if THRESHOLD_CSV.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_monitoring_baseline() -> dict:
    if not MONITORING_BASELINE_JSON.exists():
        return {}
    return json.loads(MONITORING_BASELINE_JSON.read_text(encoding="utf-8"))


def style_action(value: str) -> str:
    styles = {
        "APPROVE": "background-color: #e8f5e9; color: #1b5e20",
        "REVIEW": "background-color: #fff8e1; color: #8a5a00",
        "BLOCK": "background-color: #ffebee; color: #b71c1c",
    }
    return styles.get(value, "")


def initialise_session_state() -> None:
    defaults = {
        "streaming": False,
        "stream_records": [],
        "stream_position": 0,
        "latest_result": None,
        "stream_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_stream(row_count: int) -> None:
    dataset = load_transactions()
    if dataset.empty:
        st.session_state.stream_error = "Dataset missing. Run `python train_models.py` first."
        return
    sample = dataset.sample(
        min(row_count, len(dataset)),
        random_state=time.time_ns() % (2**32 - 1),
    ).sort_values(["tx_datetime", "transaction_id"])
    reset_feature_store()
    st.session_state.stream_records = sample.to_dict("records")
    st.session_state.stream_position = 0
    st.session_state.latest_result = None
    st.session_state.stream_error = None
    st.session_state.streaming = True


init_db()
initialise_session_state()

with st.sidebar:
    st.header("System controls")
    stream_speed = st.slider("Delay between events", 0.05, 1.00, 0.20, 0.05)
    stream_rows = st.slider("Events in this run", 50, 2000, 500, 50)

    if st.session_state.streaming:
        if st.button("Stop live stream", type="primary", width="stretch"):
            st.session_state.streaming = False
            st.rerun()
    elif st.button("Start live stream", type="primary", width="stretch"):
        start_stream(stream_rows)
        st.rerun()

    if st.button("Reset online feature store", width="stretch"):
        st.session_state.streaming = False
        st.session_state.stream_records = []
        st.session_state.stream_position = 0
        st.session_state.latest_result = None
        reset_feature_store()
        st.success("Online feature store reset.")

    total = len(st.session_state.stream_records)
    if total:
        position = min(st.session_state.stream_position, total)
        st.progress(position / total, text=f"Processed {position:,} / {total:,}")
    st.info("Train artifacts first with `python train_models.py`.")

# Process one event per Streamlit rerun. This keeps controls responsive and avoids
# a long blocking loop in the UI thread.
if st.session_state.streaming:
    position = st.session_state.stream_position
    records = st.session_state.stream_records
    if position >= len(records):
        st.session_state.streaming = False
    else:
        try:
            result = process_transaction(records[position])
            st.session_state.latest_result = result
            st.session_state.stream_position += 1
            if st.session_state.stream_position >= len(records):
                st.session_state.streaming = False
        except Exception as exc:  # Surface operational failures without losing the dashboard.
            LOGGER.exception("Live transaction processing failed")
            st.session_state.streaming = False
            st.session_state.stream_error = str(exc)

if st.session_state.stream_error:
    st.error(st.session_state.stream_error)

tab_live, tab_queue, tab_analytics, tab_eval, tab_data = st.tabs(
    ["Live traffic", "Analyst review", "Monitoring", "Model evaluation", "Dataset explorer"]
)

with tab_live:
    st.subheader("Live transaction stream")
    latest = st.session_state.latest_result
    if latest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision", latest["action"])
        c2.metric("Risk score", f"{latest['risk_score']:.3f}")
        c3.metric("Detection layer", latest["layer"])
        c4.metric("Inference latency", f"{latest['latency_ms']:.2f} ms")
        st.caption(latest["reasons"])

    recent_live = get_transactions(60)
    if recent_live.empty:
        st.info("No processed transactions yet. Start the live stream.")
    else:
        live_columns = [
            "tx_datetime",
            "tx_id",
            "customer_id",
            "terminal_id",
            "tx_amount",
            "risk_score",
            "action",
            "layer",
            "model_release",
            "policy_version",
            "latency_ms",
        ]
        st.dataframe(
            recent_live[live_columns].style.map(style_action, subset=["action"]),
            width="stretch",
            hide_index=True,
        )

with tab_queue:
    st.subheader("Manual case management")
    st.caption("Every status change is validated and appended to the audit trail.")
    status_filter = st.selectbox(
        "Queue status",
        ["PENDING", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE"],
    )
    cases = get_review_cases(status_filter, limit=500)
    if cases.empty:
        st.success(f"No {status_filter} cases.")
    else:
        st.warning(f"{len(cases)} cases found.")
        case_id = st.selectbox("Select case", cases["tx_id"].tolist())
        case = cases[cases["tx_id"] == case_id].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk score", f"{case.risk_score:.3f}")
        c2.metric("Amount", f"{case.tx_amount:.2f}")
        c3.metric("Layer", case.layer)
        c4.metric("Ground truth", "Fraud" if int(case.ground_truth) == 1 else "Legitimate")
        st.info(case.reasons)
        st.dataframe(
            pd.DataFrame([case])[
                [
                    "tx_id",
                    "tx_datetime",
                    "customer_id",
                    "terminal_id",
                    "country",
                    "device_id",
                    "channel",
                    "merchant_category",
                    "fraud_scenario",
                    "status",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        notes = st.text_area(
            "Analyst notes",
            value=str(case.analyst_notes or ""),
            key=f"notes_{case_id}",
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Start investigation", width="stretch"):
            update_transaction_status(case_id, "UNDER_REVIEW", notes or "Investigation started")
            st.rerun()
        if b2.button("Confirm fraud", width="stretch"):
            update_transaction_status(case_id, "CONFIRMED_FRAUD", notes)
            st.rerun()
        if b3.button("Mark false positive", width="stretch"):
            update_transaction_status(case_id, "FALSE_POSITIVE", notes)
            st.rerun()

        history = get_review_history(case_id)
        if not history.empty:
            with st.expander("Audit history"):
                st.dataframe(history, width="stretch", hide_index=True)

with tab_analytics:
    st.subheader("Operational monitoring")
    recent = get_transactions(5000)
    if recent.empty:
        st.info("No processed transactions yet. Start the stream first.")
    else:
        snapshot = analytics_snapshot(5000)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Processed", f"{snapshot.get('total', 0):,}")
        m2.metric("Review rate", f"{snapshot.get('review_rate', 0) * 100:.2f}%")
        m3.metric("Block rate", f"{snapshot.get('block_rate', 0) * 100:.2f}%")
        m4.metric("Stage 2 rate", f"{snapshot.get('stage2_rate', 0) * 100:.2f}%")
        m5.metric("Average latency", f"{snapshot.get('avg_latency_ms', 0):.2f} ms")

        baseline = load_monitoring_baseline()
        if baseline:
            monitoring = evaluate_operations(recent, baseline)
            st.markdown("#### Model health")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Health", monitoring["status"].replace("_", " ").upper())
            h2.metric("Data-quality errors", f"{monitoring['data_quality_error_rate']:.2%}")
            h3.metric("p95 latency", f"{monitoring['p95_latency_ms']:.2f} ms")
            h4.metric("Monitoring window", f"{monitoring['sample_size']:,}")
            signal_frame = pd.DataFrame(monitoring["signals"])
            st.dataframe(signal_frame, width="stretch", hide_index=True)
            for alert in monitoring["alerts"]:
                st.warning(alert)
            if monitoring["active_model_releases"]:
                st.caption(
                    "Model release(s): "
                    + ", ".join(monitoring["active_model_releases"])
                    + " · Policy version(s): "
                    + ", ".join(monitoring["active_policy_versions"])
                )
        else:
            st.info("Monitoring baseline missing. Retrain models to enable drift detection.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Decision distribution")
            st.bar_chart(recent["action"].value_counts())
        with col2:
            st.markdown("#### Stage routing")
            st.bar_chart(recent["layer"].value_counts())

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Review outcomes")
            st.bar_chart(recent["status"].value_counts())
        with col4:
            st.markdown("#### Risk score distribution")
            histogram = pd.cut(
                recent["risk_score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True
            ).value_counts().sort_index()
            histogram.index = histogram.index.astype(str)
            st.bar_chart(histogram)

        timeline = recent.copy()
        timeline["minute"] = pd.to_datetime(timeline["tx_datetime"]).dt.floor("min")
        by_minute = timeline.groupby("minute").agg(
            transactions=("tx_id", "count"),
            reviews=("action", lambda values: (values == "REVIEW").sum()),
            blocks=("action", lambda values: (values == "BLOCK").sum()),
        )
        st.markdown("#### Traffic over time")
        st.line_chart(by_minute[["transactions", "reviews", "blocks"]])

with tab_eval:
    st.subheader("Academic model evaluation")
    metrics = load_metrics()
    thresholds = load_thresholds()
    if metrics.empty:
        st.info("No metrics found. Run `python train_models.py` first.")
    else:
        st.dataframe(metrics, width="stretch", hide_index=True)
        numeric_columns = [
            column
            for column in ["precision", "recall", "f1", "roc_auc", "pr_auc", "precision_at_1_percent"]
            if column in metrics
        ]
        if numeric_columns:
            st.bar_chart(metrics.set_index("model")[numeric_columns])
    if not thresholds.empty:
        st.markdown("#### Threshold sensitivity")
        st.line_chart(
            thresholds.set_index("threshold")[["precision", "recall", "f1", "false_positive_rate"]]
        )

with tab_data:
    st.subheader("Dataset explorer")
    dataset = load_transactions()
    if dataset.empty:
        st.info("No dataset found. Run `python train_models.py` first.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows", f"{len(dataset):,}")
        d2.metric("Fraud rate", f"{dataset.tx_fraud.mean() * 100:.3f}%")
        d3.metric("Customers", f"{dataset.customer_id.nunique():,}")
        d4.metric("Terminals", f"{dataset.terminal_id.nunique():,}")

        scenarios = st.multiselect(
            "Fraud scenarios",
            sorted(dataset["fraud_scenario"].unique()),
        )
        filtered = dataset[
            dataset["fraud_scenario"].isin(scenarios)
        ] if scenarios else dataset
        st.markdown("#### Fraud scenarios")
        st.bar_chart(dataset["fraud_scenario"].value_counts())
        st.dataframe(filtered.head(500), width="stretch", hide_index=True)

if st.session_state.streaming:
    time.sleep(stream_speed)
    st.rerun()
