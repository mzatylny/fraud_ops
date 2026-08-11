# Operations Runbook

## Start locally

```bash
make install
make train
make run
```

The UI is available at <http://localhost:8501>. Training is required after a fresh clone because generated model artifacts are intentionally not versioned.

## Pre-release verification

```bash
make quality
```

This runs linting, source coverage, static security analysis, and dependency auditing. The dashboard smoke test is part of the test suite.

## Health signals

Use the Monitoring tab to inspect:

- traffic-window sample size
- amount, score, and action-distribution PSI
- invalid operational-data rate
- p95 scoring latency against the configured SLO
- active model release and policy version
- mixed-release or mixed-policy warnings

Treat `insufficient_data` as unknown health. A warning requires investigation; a critical signal should pause promotion and trigger input-pipeline, model, and policy checks.

## Incident triage

1. Confirm the active release and policy IDs.
2. Check whether the alert is input drift, prediction/action drift, data quality, latency, or mixed provenance.
3. Inspect recent transactions and audit events without modifying evidence.
4. Compare the event schema and upstream timestamp/order guarantees with the release contract.
5. For an artifact error, validate that both model files and the manifest came from one training run.
6. For replays, confirm upstream retry behaviour; do not remove the unique constraint.
7. Roll back the complete release bundle if the incident began with a deployment.
8. Record the timeline, affected release IDs, resolution, and preventive action.

## Database notes

SQLite WAL files are created beside `data/fraud_ops.db`. Back up the database and its WAL consistently, or stop writes before copying. The Docker Compose configuration stores `/app/data` in a named volume. SQLite is appropriate for this demonstrator; a horizontally scaled deployment needs a transactional managed store and a durable ordered event bus.

## Dependency response

If `pip-audit` reports a vulnerability, confirm whether the vulnerable package and code path are used, upgrade to a compatible fixed release, rerun all quality gates, and document any temporary exception with owner and expiry.
