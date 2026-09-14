# ADR 0002: Idempotent event processing

- Status: Accepted
- Date: 2026-08-11

## Context

Delivery systems commonly retry events. Processing the same source transaction twice would duplicate decisions, distort monitoring, and incorrectly advance behavioural feature state.

## Decision

Use `(simulation_id, transaction_id)` as the idempotency key. Each UI start creates an independent simulation and a feature store private to that browser session. Callers that omit `simulation_id`, including existing integrations, use the `default` namespace. Existing database records migrate to that namespace without losing duplicate protection. Reject known IDs before feature preview and enforce a unique partial database index as the authoritative race-safe boundary. Commit online feature state only after the database insert succeeds.

## Consequences

Sequential retries fail early, concurrent retries cannot both persist, and rejected events do not contaminate feature history. The caller must treat the duplicate-domain error as an acknowledged replay. Distributed deployments require a shared transactional idempotency store.
