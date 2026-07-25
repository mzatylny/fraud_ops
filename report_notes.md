# Report Notes You Can Reuse

## Dataset paragraph

The dataset generator was designed using the conceptual structure of the Fraud Detection Handbook. It generates customer profiles with individual spending habits and geographic positions, terminal profiles with merchant location and risk attributes, then produces daily transaction streams using customer-specific transaction frequencies and amount distributions. Fraud labels are introduced through multiple scenarios: high-value obvious fraud, temporarily compromised terminals, customer/card compromise with increased spending, card-testing bursts, and stealth account takeover using new devices and foreign countries. This design creates an imbalanced, temporal, behavioural fraud-detection problem that is more realistic than a simple random classification dataset.

## System paragraph

The implemented system is an end-to-end fraud operations prototype. A transaction stream is ingested, online behavioural features are previewed from historical state, a fast Stage 1 model screens the transaction, suspicious cases are routed to an advanced Stage 2 ensemble, and a validated decision engine returns approve, review, or block. Scored transactions are persisted in SQLite before feature state is committed, preventing failed events from contaminating later features. Suspicious cases enter an analyst workflow with validated status transitions, notes, timestamps, and an audit history. A responsive Streamlit dashboard provides live traffic, review management, monitoring analytics, dataset exploration, and model evaluation.

## Evaluation paragraph

The evaluation compares Logistic Regression, Random Forest, Extra Trees, an Advanced Soft-Voting Ensemble, and the final two-stage hybrid pipeline. A chronological holdout is used instead of a random split. Metrics include precision, recall, F1-score, ROC-AUC, PR-AUC, false positive rate, false negative rate, precision at top 1%, and inference latency. The two-stage architecture is also evaluated operationally by measuring the percentage of transactions routed to Stage 2. This allows the project to justify the design as an accuracy-latency trade-off rather than only as a model-selection exercise.

## Limitations paragraph

The main limitation is that the dataset is synthetic. Although the generator uses realistic fraud-detection concepts such as customer spending behaviour, terminal compromise, class imbalance, and temporal fraud windows, it cannot fully reproduce the complexity of real payment networks. The real-time component is also a simulation rather than a production streaming infrastructure. Future work could include integration with a message queue, model retraining from analyst feedback, drift detection, fairness analysis, and evaluation on real anonymised transaction data where legally and ethically available.
