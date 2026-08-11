# Security Policy

## Supported version

Security fixes target the latest code on `main`.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, or customer data. Use GitHub's private vulnerability reporting feature for this repository when available. Include the affected component, reproduction steps, impact, and any suggested mitigation.

## Secrets and data

This repository does not require API keys and should contain only synthetic data. Never commit payment data, personal data, access tokens, private model artifacts, or populated operations databases. Generated `data/`, `models/`, and `reports/` content is excluded from version control.

## Security controls

- Strict event and score validation
- Parameterized SQL and guarded review-state transitions
- Replay protection at application and database boundaries
- Model artifact checksum and compatibility validation
- Non-root container runtime with a read-only root filesystem
- Least-privilege GitHub Actions permissions
- Static application and dependency vulnerability scans in CI

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for scope and residual risks.
