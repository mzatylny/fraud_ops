# Changelog

All notable changes to this project are documented here.

## 3.0.0 - 2026-08-11

### Added

- Integrity-checked model release manifests with content-derived release IDs
- Feature-schema, threshold, and policy compatibility checks at model load time
- Per-decision model and policy provenance
- Source-event idempotency with application and database replay guards
- PSI-based input, prediction, and action-distribution monitoring
- Data-quality, p95 latency, and mixed-release operational alerts
- Monitoring, model-registry, replay, and Streamlit rendering tests
- Coverage, security, dependency-audit, and container delivery controls
- Model-governance, operations, threat-model, and architecture-decision documentation

### Changed

- Model metadata and monitoring artifacts are now written atomically
- The dashboard uses current Streamlit width APIs and exposes operational health signals
- CI now verifies Python 3.11 and 3.12 with explicit least-privilege permissions and timeouts

## 2.0.0 - 2026-08-10

- Added strict event validation, safe online feature-state commit, artifact reload checks, WAL persistence, audit-state validation, CI, and expanded regression coverage.

## 1.0.0

- Initial synthetic fraud simulator, feature pipeline, two-stage detector, persistence layer, and analyst dashboard.
