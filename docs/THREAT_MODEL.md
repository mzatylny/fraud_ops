# Threat Model

## Assets

- Integrity and availability of fraud decisions
- Model artifacts, thresholds, and feature contract
- Transaction and analyst-review history
- Operational monitoring and audit evidence

## Trust boundaries

1. Upstream event source to ingestion validation
2. Runtime filesystem to model loader
3. Application process to SQLite
4. Analyst interaction to review-state changes
5. Source repository to CI and container build

## Principal threats and controls

| Threat | Control | Residual risk |
| --- | --- | --- |
| Malformed or adversarial events | Typed normalisation, finite/range checks, allowed categories, chronological checks | Valid-looking adversarial behaviour may still evade the model |
| Replay causing duplicate decisions | Pre-flight existence check plus unique source-ID constraint | Multi-region deduplication needs a shared durable store |
| Model file corruption or partial deployment | Atomic writes, sizes and SHA-256 checks, schema/policy/threshold binding | Hashes do not authenticate the publisher |
| Training/serving skew | Shared ordered schema and batch/online parity tests | External preprocessing changes remain an integration risk |
| SQL injection or invalid workflow state | Parameterized SQL, controlled identifiers, transition allowlist | Authentication and per-user authorization are not implemented |
| Silent distribution or latency degradation | PSI, quality-rate, latency-SLO, and provenance alerts | PSI does not measure labelled accuracy or fairness |
| Dependency compromise | Pinned top-level dependencies, dependency audit, minimal container | Transitive supply-chain and build provenance need stronger controls |
| Container privilege escalation | Non-root user and read-only root filesystem | Host/runtime hardening is outside this repository |

## Out of scope

The prototype has no public API, real cardholder data, secrets, multi-tenant access, or payment authorization integration. Before such use, add authenticated encrypted interfaces, role-based access, secret management, retention policy, tamper-evident audit storage, signed builds and artifacts, rate limiting, abuse monitoring, and privacy/compliance review.
