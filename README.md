# Fraud Operations Platform — BBachelor Project Prototype

This is an end-to-end fraud detection platform prototype, designed to look like a real system rather than a single notebook experiment.

## What it contains

- Handbook-inspired synthetic transaction simulator
- Customer and terminal/merchant profile generation
- Time-dependent fraud scenarios
- Less-than-1%-style class imbalance target depending on configuration
- Batch feature engineering for academic model training
- Stateful online feature engineering for live transaction scoring
- Two-stage detection architecture:
  - Stage 1: fast Logistic Regression screen
  - Stage 2: advanced soft-voting ensemble
- Decision engine: approve, review, block
- SQLite operational database
- Analyst case-management workflow
- Dashboard monitoring and model evaluation tabs
- Threshold sweep and model-comparison outputs
- Unit tests

## Source inspiration

The synthetic data design is inspired by the *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook*. The handbook motivates customer profiles, terminal profiles, customer-terminal proximity, daily transaction generation, imbalanced fraud labels, and time-dependent fraud scenarios.

Important: this project does **not** copy the handbook code directly. It implements a new project-specific simulator and extends it with extra operational fields such as country, device, channel, merchant category, reasons, latency, and analyst workflow.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python train_models.py --customers 900 --terminals 1800 --days 55
streamlit run app.py
```

For a faster demo:

```bash
python train_models.py --customers 300 --terminals 600 --days 25 --max-transactions 25000
streamlit run app.py
```

## Suggested report title

**Design and Evaluation of a Real-Time Two-Stage Fraud Operations Platform Using Handbook-Inspired Synthetic Transaction Streams**

## Suggested report structure

1. Introduction and motivation
2. Background on card fraud detection
3. Dataset simulation design
4. System requirements
5. Architecture and data flow
6. Feature engineering
7. Model design and two-stage routing
8. Experimental evaluation
9. Operational dashboard and analyst workflow
10. Limitations, ethics, and future work
11. Conclusion

## Academic strengths

- It is a system, not just a classifier.
- It uses realistic fraud-detection challenges: imbalance, temporal behaviour, customer history, merchant/terminal risk, and delayed analyst review.
- It evaluates multiple models and compares operational trade-offs.
- It explicitly measures Stage 2 call rate and inference latency.
- It includes human-readable risk reasons and a feedback-oriented review workflow.

## Key files

- `src/dataset_generator.py` — synthetic dataset simulator
- `src/features.py` — batch and online features
- `train_models.py` — training, evaluation, metrics export
- `src/detector.py` — two-stage scoring and explainability
- `src/database.py` — SQLite persistence and review audit trail
- `src/stream_processor.py` — ingestion-to-decision pipeline
- `app.py` — Streamlit dashboard
- `tests/` — unit tests
