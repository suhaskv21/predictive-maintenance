# 13: Final triggered commit and spec §6 checklist

**What to build:** The submission state: the live site was built from the repository's head commit, and every spec §6 acceptance item has been checked against evidence.

Parent: `.scratch/five-hour-submission/spec.md`.

**Blocked by:** 07 (Holdout from the CI-built artifact), 08 (Example readings), 09 (Out-of-range note and sensitivity expander), 10 (In-app golden check), 11 (Phone and Firefox cold test), 12 (README and system-design diagram)

**Status:** ready-for-human

- [ ] Cuttable tickets 08 to 10 are either done or explicitly recorded as cut (and the README's "not implemented" list agrees).
- [ ] The last commit pushed contains `TRIGGER_CI` (only the head commit of a push is checked), the build succeeds, and the live site's footer commit matches `master`'s head.
- [ ] `just check` passes on that commit and its output is shown.
- [ ] Spec §6 checklist walked item by item with evidence: clean install from the documented instructions; training and evaluation from documented commands; target, features, split and metrics documented; preprocessing fitted on train only and the Holdout scored once; results and metric rationale reported; the app loads the artifact and assesses a Reading; the UI explains its output without overstating certainty; the live URL in the README works; the diagram includes ingestion, training, serving, monitoring and production additions; the README has two monitoring metrics, retraining and validation, and risks with mitigations.
- [ ] `git ls-files` shows no files under the models directory other than its README; no Accepted ADR was edited; no secrets or `.env` are committed.
