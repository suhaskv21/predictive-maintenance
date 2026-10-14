# 0015. Reproducibility: train only from a clean Git commit and record that commit in the manifest

- Status: Accepted
- Date: 2026-10-14
- Deciders: Suhas (accepted 2026-10-14); drafted by agent
- Refines: [0009](0009-app-output.md) and [0012](0012-five-hour-submission-scope.md) (contents of
  `<model>.manifest.json`). Builds on the split in [0005](0005-split-and-validation.md). Does not
  change any Accepted ADR.

## Context

The spec asks for a repeatable training and evaluation process with documented seeds, commands and
data expectations (`TECHNICAL_SPEC.md` §3, point 6). Today:

- Randomness is seeded from one value in `configs/default.toml`, and the stratified split is a
  deterministic function of that seed, `split.test_size` and the data (`evaluation/split.py`).
- The manifest already records library versions and the SHA-256 of the four sidecar files
  (`training/unit.py`); inference checks both at load (`inference/artifact.py`).
- The manifest does **not** say which data, split, config or code produced the model. If
  `data/raw/ai4i2020.csv` or the config changed, the same command would silently train a different
  model, and nothing in the artifact would show it.

Constraints: the dataset is 10,000 rows, public, committed, and static. `Product ID` is unique per
Reading (ADR 0005), so it identifies rows. Generated artifacts under `models/` are never committed
(`AGENTS.md`). The manifest ships with the WebAssembly site, so it must stay small and hold no secrets.

## Options considered

1. **Write the train and test partitions to `data/` and version those files.**
   - Pros: the exact rows are inspectable.
   - Cons: duplicates the raw data; the copies can drift from the raw file or be edited; a visible
     `test.csv` invites touching the holdout before the end (ADR 0005); the split is already
     reproducible from its inputs. Adds files to review for no extra guarantee.
2. **Record the data, split, config and lockfile hashes, plus the commit and a dirty flag.** Pros:
   works from a dirty tree; each input is checkable on its own. Cons: many fields that all say what
   one clean commit already says, and each needs code and tests.
3. **Train only from a clean commit and record that commit** (chosen). Everything that determines the
   model except the installed libraries is committed (code, `configs/`, `uv.lock`, and the raw CSV), so
   the commit identifies it. One field replaces the hashes and the dirty flag.
4. **Adopt a data and experiment versioning tool (DVC, MLflow).** Pros: lineage and registries.
   Cons: new dependency and infrastructure for one static 10k-row file; needs approval under
   `AGENTS.md`. Revisit if data sources become multiple or changing.
5. **Do nothing beyond library versions and file hashes.** Pros: no work. Cons: cannot tell what data
   or config produced a model; the spec's "data expectations" are undocumented in the artifact.

## Decision

Take option 3.

**1. Clean-commit gate.** The training command (`python -m predictive_maintenance.training.train`,
`just train`, and so the Pages build) exits with an error if `git status --porcelain` is not empty
(modified or untracked files; git-ignored files such as `models/` do not count) or if it cannot read
`HEAD`. The gate lives in the command, not in `write_unit`, so tests and notebooks that call
library functions are unaffected. The Pages build checks out the repository, so `HEAD` is available.

**2. One new manifest key.** `git`: the full commit SHA of `HEAD`. Because the commit pins the code,
`configs/default.toml` (seed, `test_size`, CV ratios), `uv.lock` and `data/raw/ai4i2020.csv`, no hash of
any of these is stored, and no dirty flag. To reproduce a model: check out the commit, `uv sync`,
`just train`. The existing `libraries` key stays, because it records the libraries actually installed
at training time, which the lockfile only intends.

**3. Escape hatch for development.** `train --allow-dirty` runs on a dirty tree and writes `"git": null`.
The deploy contract (`deploy/contract.py`, run by the Pages build) rejects a manifest whose `git` is
null or not a 40-character hex SHA, so such an artifact cannot be published. Local experiments are
not blocked, and the null is the only trace of them. Without the flag the gate is absolute.

**Not stored, on purpose:** dataset hash, split fingerprint, config hash, lockfile hash, dirty flag.
Not persisted either: the train and test partitions. The split is re-derived by the commit's code.

**Checks and tests** (mirrored under `tests/predictive_maintenance/`):

- Training refuses a dirty or untracked-file tree, and writes the commit SHA on a clean one (tested
  against a throwaway Git repository, not this one).
- `--allow-dirty` writes `"git": null`, and the contract rejects it.
- Training twice on one commit gives identical train and test `Product ID`s and identical golden
  scores. This is the only guard against nondeterminism that a commit cannot pin, such as a
  library behaving differently under the same version pin.
- A manifest without `git` still loads in inference (backward compatible); only the deploy contract
  requires it.

Pin remaining nondeterminism in the model itself (a `random_state` and a fixed `n_jobs` for any
parallel learner). Bitwise-identical `joblib` files are **not** promised across machines; the promise
is identical splits and scores within a floating-point tolerance on the golden Readings.

## Consequences

- Easier: every published model maps to exactly one commit, so data, config, seed, lockfile and code
  are all pinned by one field. Few things to track, no data copies, no hash bookkeeping.
- Harder: training needs a commit first, so every experiment that writes an artifact is a commit
  (or uses `--allow-dirty` and is never published). Training needs a Git checkout; a source tarball
  cannot train.
- A commit is only as durable as its history: a commit that is rewritten or never pushed cannot be
  checked out later. Published builds come from the pushed `main` branch via Pages, which covers it;
  a local model whose commit is lost is not reproducible.
- The commit does not say which data a model saw when the data later moves out of Git. Revisit and
  add a dataset hash and version then (and option 4 if data arrives in batches or from more than one
  source).
- A human cannot read the seed or split ratios from the manifest alone; they are in `configs/` at that
  commit.
- Accepted ADRs 0009 and 0012 describe the manifest without `git`; this ADR adds one key without
  contradicting them.
