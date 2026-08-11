from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = DATA_DIR / "fraud_ops.db"
TRANSACTIONS_CSV = DATA_DIR / "handbook_inspired_transactions.csv"
CUSTOMERS_CSV = DATA_DIR / "customer_profiles.csv"
TERMINALS_CSV = DATA_DIR / "terminal_profiles.csv"
METRICS_CSV = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_CSV = REPORTS_DIR / "threshold_sweep.csv"
EXPERIMENT_METADATA_JSON = REPORTS_DIR / "experiment_metadata.json"
MONITORING_BASELINE_JSON = REPORTS_DIR / "monitoring_baseline.json"
MODEL_MANIFEST_JSON = MODELS_DIR / "model_manifest.json"

# The same cold-start baseline is used by batch and online feature engineering.
# Keeping it explicit prevents training-serving skew for a customer's first event.
DEFAULT_CUSTOMER_AVG_AMOUNT = 75.0
FEATURES = [
    "tx_amount",
    "hour",
    "is_night",
    "customer_avg_amount",
    "amount_ratio_customer",
    "time_since_last_tx_seconds",
    "customer_tx_count_1h",
    "customer_tx_count_24h",
    "terminal_tx_count_24h",
    "terminal_risk_score",
    "distance_to_terminal",
    "is_new_terminal_for_customer",
    "is_foreign_country",
    "is_new_device",
]
STAGE1_THRESHOLD = 0.25
REVIEW_THRESHOLD = 0.35
BLOCK_THRESHOLD = 0.85
POLICY_VERSION = "2026-08-11-v1"
