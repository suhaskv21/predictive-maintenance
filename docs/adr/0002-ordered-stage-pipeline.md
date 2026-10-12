# 0002. Use an ordered stage pipeline with a thin orchestrator

- Status: Accepted
- Date: 2026-10-12
- Deciders: Suhas

## Context
An orchestration workflow must coordinate a sequence of processing steps while
allowing each step to focus on its own work. A workflow implemented as one large
procedure couples orchestration to domain logic, while independently coordinated
steps make ordering and shared execution behavior harder to understand.

The following filter pipeline demonstrates this decision: processing units implement a
common `Filter.run(data) -> data` protocol, and `Pipeline` invokes them in a
configured sequence. The current implementation uses Polars lazy frames; those are application-specific
choices rather than requirements of the architectural principle.

Spec reference: §4.2 requires a reproducible, modular preprocessing, training, and
evaluation pipeline in a Python package.

## Options considered
1. **One monolithic workflow procedure** — rejected: adding or changing a
   processing step would also require editing orchestration logic, and step-level
   behavior would be harder to reuse and test independently.
2. **Steps coordinating execution themselves** — rejected: ordering and shared
   execution policies would be distributed across the steps, obscuring the
   workflow as a whole.
3. **Ordered stages behind a shared contract, run by a thin orchestrator** — chosen.

## Decision
Represent a workflow as an explicitly ordered collection of stages that conform
to a small, shared input/output contract. A pipeline orchestrator invokes each
stage and passes its output to the next stage. Stages own their domain-specific
processing; the orchestrator owns workflow-level coordination, such as sequencing
and applicable cross-cutting execution policies.

The contract should express the data or artifacts exchanged by the workflow. It
may be a dataframe, a model artifact, a structured context, or another suitable
type. Keep it as narrow as practical, and do not require all workflows to use
 lazy-frame contract. Batching, branching, retries, persistence, and other
execution policies are added only when required by the workflow.

## Consequences
- The execution order is explicit and can be reviewed independently of each
  stage's implementation.
- Stages can be added, replaced, and tested behind the shared contract, without
  requiring the orchestrator to know their domain-specific details.
- The contract is an important boundary: incompatible data shapes require an
  explicit adapter or a deliberately broader contract.
- Sequential composition makes order significant. Workflows with dependencies,
  conditional branches, or parallel work need those semantics represented
  explicitly by the orchestrator rather than assumed to be a simple sequence.
- Cross-cutting policies remain centralized, so the orchestrator must avoid
  accumulating stage-specific business logic.

## Application in this project
Agreed with the human when adopting this ADR; implementation follows separately.
- **Contract:** a generic `Stage[T]` protocol (`run(data: T) -> T`). The data-prep
  workflow uses `T = polars.DataFrame`; training uses its own context type, added
  with the training ADRs. No single contract is forced across workflows.
- **Location:** `src/predictive_maintenance/pipeline/` holds only the generic
  `Stage[T]` and `Pipeline[T]` orchestrator. Concrete stages live in their domain
  packages (`data/`, `preprocessing/`, `training/`), never in `pipeline/`.
- **Initial scope:** orchestrator plus ingest and clean stages. Split and train
  stages wait for the pending decisions on target/features, split strategy, and
  seed/config ADRs listed as pending in `0001`, since they change the feature set and split
  strategy.
- **Cleaning stage rules:** strip the BOM from the first column name; validate
  required columns and dtypes; raise a validation error on nulls or duplicate
  `UDI`; drop no columns (feature selection is a later decision).
- **Leakage safety:** stages that fit anything (scaling, encoding, model) run only
  after the train/test split and fit on the train partition only.

## Follow-ups
- Number note: `0001` lists `0002` as "Dataset source and file location" under
  pending decisions. That list is a forward-looking plan inside an Accepted ADR and
  is not edited; renumber the pending decisions (dataset source becomes `0003`, and
  so on) when they are written.
