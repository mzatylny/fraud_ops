# Report Notes You Can Reuse

## Dataset paragraph

The dataset generator was designed using the conceptual structure of the Fraud Detection Handbook. It generates customer profiles with individual spending habits and geographic positions, terminal profiles with merchant location and risk attributes, then produces daily transaction streams using customer-specific transaction frequencies and amount distributions. Fraud labels are introduced through multiple scenarios: high-value obvious fraud, temporarily compromised terminals, customer/card compromise with increased spending, card-testing bursts, and stealth account takeover using new devices and foreign countries. This design creates an imbalanced, temporal, behavioural fraud-detection problem that is more realistic than a simple random classification dataset.

## System paragraph

The implemented system is an end-to-end fraud operations prototype. A transaction stream is ingested, online behavioural features are previewed from historical state, a fast Stage 1 model screens the transaction, suspicious cases are routed to an advanced Stage 2 ensemble, and a validated decision engine returns approve, review, or block. Scored transactions are persisted in SQLite before feature state is committed, preventing failed events from contaminating later features. Suspicious cases enter an analyst workflow with validated status transitions, notes, timestamps, and an audit history. A responsive Streamlit dashboard provides live traffic, review management, monitoring analytics, dataset exploration, and model evaluation.

## Evaluation paragraph

The evaluation compares Logistic Regression, Random Forest, Extra Trees, an Advanced Soft-Voting Ensemble, and the final two-stage hybrid pipeline. A chronological holdout is used instead of a random split. Metrics include precision, recall, F1-score, ROC-AUC, PR-AUC, false positive rate, false negative rate, precision at top 1%, and inference latency. The two-stage architecture is also evaluated operationally by measuring the percentage of transactions routed to Stage 2. This allows the project to justify the design as an accuracy-latency trade-off rather than only as a model-selection exercise.

## Limitations paragraph

The main limitation is that the dataset is synthetic. Although the generator uses realistic fraud-detection concepts such as customer spending behaviour, terminal compromise, class imbalance, and temporal fraud windows, it cannot fully reproduce the complexity of real payment networks. The real-time component is also a simulation rather than production streaming infrastructure. The implemented PSI monitor detects distribution change but not causal performance or fairness regressions. Future work should add a durable message queue, governed retraining from analyst feedback, labelled-performance and fairness monitoring, signed release provenance, and evaluation on real anonymised transaction data where legally and ethically available.

## Governance paragraph

The release path treats both models, the ordered feature schema, decision thresholds, and policy version as a single governed unit. A content-derived release identifier and SHA-256 artifact checks are validated before activation, and every persisted decision records its model release and policy version. Operational monitoring compares recent amount, risk-score, and action distributions with the training baseline using Population Stability Index, while also checking input quality, p95 scoring latency, and mixed-release windows. These controls make model behaviour traceable and observable, while deliberately distinguishing artifact integrity and drift detection from stronger production controls such as signed provenance and labelled-performance monitoring.
