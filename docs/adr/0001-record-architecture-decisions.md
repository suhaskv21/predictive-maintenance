# 0001. Record architecture decisions

- Status: Accepted
- Date: 2026-10-12
- Deciders: Suhas

## Context
The project is co-developed by a human and AI agents. Rationale must survive across sessions, and the spec (§8) requires explicit, justified decisions.

## Decision
Record significant decisions as ADRs in `docs/adr/`, numbered sequentially, using `0000-template.md`. Accepted ADRs are immutable; changes are made by superseding.

## Consequences
Agents can read ADRs for context and propose new ones. Small overhead per decision.

## Pending decisions (spec §8)
- 0002 Dataset source and file location
- 0003 Target, features, excluded columns
- 0004 Train/test and validation strategy
- 0005 Primary metric and operating threshold
- 0006 Model choice (interpretability vs performance)
- 0007 Application output shown to engineers
- 0008 Deployment provider
