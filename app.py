from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import METRICS_CSV, THRESHOLD_CSV, TRANSACTIONS_CSV
from src.database import analytics_snapshot, get_review_cases, get_transactions, init_db, update_transaction_status
from src.features import reset_feature_store
from src.stream_processor import process_transaction

st.set_page_config(page_title="Fraud Operations Platform", layout="wide")
st.title("Fraud Operations Platform — Real-Time Two-Stage Fraud Detection")
st.caption("Handbook-inspired simulator → online features → fast screen → advanced ensemble → decision engine → analyst workflow")

init_db()

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "reset_features" not in st.session_state:
    st.session_state.reset_features = False

with st.sidebar:
    st.header("System Controls")
    if st.button("Start / Stop Live Stream"):
        st.session_state.streaming = not st.session_state.streaming
    if st.button("Reset Online Feature Store"):
        reset_feature_store()
        st.success("Online feature store reset.")
    stream_speed = st.slider("Stream delay", 0.05, 1.00, 0.20, 0.05)
    stream_rows = st.slider("Rows per run", 50, 2000, 500, 50)
    st.markdown("Run `python train_models.py` first if models/data are missing.")


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    if not TRANSACTIONS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRANSACTIONS_CSV)
    df["tx_datetime"] = pd.to_datetime(df["tx_datetime"])
    return df.sort_values("tx_datetime")


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    if METRICS_CSV.exists():
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_thresholds() -> pd.DataFrame:
    if THRESHOLD_CSV.exists():
        return pd.read_csv(THRESHOLD_CSV)
    return pd.DataFrame()


def style_action(val: str) -> str:
    if val == "APPROVE":
        return "background-color: #e8f5e9; color: #1b5e20"
    if val == "REVIEW":
        return "background-color: #fff8e1; color: #8a5a00"
    if val == "BLOCK":
        return "background-color: #ffebee; color: #b71c1c"
    return ""


tab_live, tab_queue, tab_analytics, tab_eval, tab_data = st.tabs(
    ["Live Traffic", "Analyst Review", "Monitoring", "Model Evaluation", "Dataset Explorer"]
)

with tab_live:
    st.subheader("Live Transaction Stream")
    live_placeholder = st.empty()
    latest_placeholder = st.empty()

with tab_queue:
    st.subheader("Manual Case Management")
    st.caption("The analyst workflow creates an audit trail and can be reused as a feedback source for retraining.")
    status_filter = st.selectbox("Queue status", ["PENDING", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE"])
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
        c4.metric("Ground truth", "Fraud" if int(case.ground_truth) == 1 else "Legit")
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
            use_container_width=True,
        )
        notes = st.text_area("Analyst notes", value=str(case.analyst_notes or ""))
        b1, b2, b3 = st.columns(3)
        if b1.button("Start investigation"):
            update_transaction_status(case_id, "UNDER_REVIEW", notes or "Investigation started")
            st.rerun()
        if b2.button("Confirm fraud"):
            update_transaction_status(case_id, "CONFIRMED_FRAUD", notes)
            st.rerun()
        if b3.button("Mark false positive"):
            update_transaction_status(case_id, "FALSE_POSITIVE", notes)
            st.rerun()

with tab_analytics:
    st.subheader("Operational Monitoring")
    recent = get_transactions(5000)
    if recent.empty:
        st.info("No processed transactions yet. Start the stream first.")
    else:
        snap = analytics_snapshot(5000)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Processed", snap.get("total", 0))
        m2.metric("Review rate", f"{snap.get('review_rate', 0) * 100:.2f}%")
        m3.metric("Block rate", f"{snap.get('block_rate', 0) * 100:.2f}%")
        m4.metric("Stage 2 rate", f"{snap.get('stage2_rate', 0) * 100:.2f}%")
        m5.metric("Avg latency", f"{snap.get('avg_latency_ms', 0):.2f} ms")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Decision Distribution")
            st.bar_chart(recent["action"].value_counts())
        with col2:
            st.markdown("#### Stage Routing")
            st.bar_chart(recent["layer"].value_counts())

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Review Outcomes")
            st.bar_chart(recent["status"].value_counts())
        with col4:
            st.markdown("#### Risk Score Histogram")
            hist = pd.cut(recent["risk_score"], bins=[0, .2, .4, .6, .8, 1.0]).value_counts().sort_index()
            st.bar_chart(hist)

        timeline = recent.copy()
        timeline["minute"] = pd.to_datetime(timeline["tx_datetime"]).dt.floor("min")
        by_minute = timeline.groupby("minute").agg(
            transactions=("tx_id", "count"),
            avg_score=("risk_score", "mean"),
            reviews=("action", lambda s: (s == "REVIEW").sum()),
            blocks=("action", lambda s: (s == "BLOCK").sum()),
        )
        st.markdown("#### Traffic Over Time")
        st.line_chart(by_minute[["transactions", "reviews", "blocks"]])

with tab_eval:
    st.subheader("Academic Model Evaluation")
    metrics = load_metrics()
    thresholds = load_thresholds()
    if metrics.empty:
        st.info("No metrics found. Run `python train_models.py` first.")
    else:
        st.dataframe(metrics, use_container_width=True)
        numeric_cols = [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc", "precision_at_1_percent"] if c in metrics]
        if numeric_cols:
            chart = metrics.set_index("model")[numeric_cols]
            st.bar_chart(chart)
    if not thresholds.empty:
        st.markdown("#### Threshold Sensitivity")
        st.line_chart(thresholds.set_index("threshold")[["precision", "recall", "f1", "false_positive_rate"]])

with tab_data:
    st.subheader("Dataset Explorer")
    dataset = load_transactions()
    if dataset.empty:
        st.info("No dataset found. Run `python train_models.py` first.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows", f"{len(dataset):,}")
        d2.metric("Fraud rate", f"{dataset.tx_fraud.mean() * 100:.3f}%")
        d3.metric("Customers", dataset.customer_id.nunique())
        d4.metric("Terminals", dataset.terminal_id.nunique())
        st.markdown("#### Fraud Scenarios")
        st.bar_chart(dataset["fraud_scenario"].value_counts())
        st.dataframe(dataset.head(200), use_container_width=True)

# Streaming is rendered after tab definitions so placeholders exist.
if st.session_state.streaming:
    data = load_transactions()
    if data.empty:
        st.error("Dataset missing. Run `python train_models.py` first.")
    else:
        stream = data.sample(min(stream_rows, len(data)), random_state=int(time.time()) % 99999).sort_values("tx_datetime")
        for _, row in stream.iterrows():
            if not st.session_state.streaming:
                break
            result = process_transaction(row.to_dict())
            recent = get_transactions(60)
            if not recent.empty:
                with live_placeholder.container():
                    st.dataframe(
                        recent[
                            [
                                "tx_datetime",
                                "tx_id",
                                "customer_id",
                                "terminal_id",
                                "tx_amount",
                                "risk_score",
                                "action",
                                "layer",
                                "latency_ms",
                            ]
                        ].style.applymap(style_action, subset=["action"]),
                        use_container_width=True,
                    )
                with latest_placeholder.container():
                    st.success(
                        f"Latest: {result['action']} | score={result['risk_score']:.3f} | "
                        f"{result['layer']} | {result['latency_ms']:.2f} ms"
                    )
                    st.caption(result["reasons"])
            time.sleep(stream_speed)
