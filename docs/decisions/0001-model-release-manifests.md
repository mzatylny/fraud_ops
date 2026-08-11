# ADR 0001: Content-addressed model release manifests

- Status: Accepted
- Date: 2026-08-11

## Context

Loading independent model files based only on modification time can activate a partial, corrupt, or schema-incompatible deployment. A decision must also be attributable to the exact artifacts and policy used.

## Decision

Treat the two model files, ordered feature schema, decision thresholds, and policy version as one release. Store their sizes and SHA-256 digests in an atomically written manifest and derive a stable release ID from that governed content. Validate the full contract before model activation and persist the release ID with every decision.

## Consequences

Mixed or tampered artifacts fail closed and decisions become traceable. Fresh clones must train or receive a complete release bundle. The manifest proves integrity but not publisher identity; signed provenance remains future work.
