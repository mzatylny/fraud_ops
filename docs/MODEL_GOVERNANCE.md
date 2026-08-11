# Model Governance

## Release contract

Training creates two model artifacts and `models/model_manifest.json`. The manifest records:

- schema version and creation timestamp
- ordered feature names and their SHA-256 digest
- SHA-256 and byte size for both model files
- versioned decision policy and exact review/block thresholds
- deterministic 16-character release ID derived from the governed content

The application validates the complete contract before activating a release and again whenever any governed file changes. Missing, corrupt, stale, or incompatible content fails closed with an actionable error.

## Traceability

Every persisted decision includes the model release and policy version, alongside its score, action, layer, latency, and reason. This supports incident scoping, release comparison, and reconstruction of which controls produced an outcome.

## Promotion checklist

1. Run the chronological training and holdout evaluation.
2. Review precision/recall trade-offs, PR-AUC, false-positive and false-negative rates, top-1% precision, latency, and Stage 2 call rate.
3. Confirm the release manifest and monitoring baseline were generated together.
4. Run `make quality` and require all CI gates to pass.
5. Record the intended policy version and thresholds in the release review.
6. Promote the complete artifact set as one immutable unit.
7. After activation, verify one decision contains the expected release and policy IDs.
8. Observe data, score, action, and latency signals before increasing traffic.

## Rollback

Restore both model files and the matching manifest as a single release unit. Do not mix artifacts between releases. Restart or allow the detector's file-signature watcher to reload, then confirm the active release ID in the monitoring view. A production registry should retain immutable release bundles and approval records.

## Deliberate limitations

SHA-256 verifies integrity, not publisher identity. Production promotion should add signed provenance, artifact access control, independent approval, model cards, fairness and calibration checks, labelled-performance monitoring, and automated rollback criteria.
