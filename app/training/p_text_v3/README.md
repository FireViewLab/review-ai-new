# P_text v3 isolated pipeline

This package is independent of `app.training.p_text` and the production
runtime. It never derives labels from CANDIDATE, CONTROL, or DIVERSITY and it
never creates a train/validation/test split.

## Contracts

- Input workbook: `dataset` sheet with an approved `split` column.
- Human labels: only adjudicated `SUFFICIENT/PRESENT` becomes `SUSPICIOUS` and
  `SUFFICIENT/ABSENT` becomes `NORMAL`.
- Pilot records: hashed exclusion manifest applies to train, validation, and
  evaluation. Verified near-duplicate and v2-lineage groups are also consumed.
- Leakage: group, product, normalized content, duplicate cluster, and verified
  relation groups are fail-closed across splits.
- Canonical v2: readable with explicit legacy product-overlap audit. The v3
  builder can carry only the fixed v2 train split forward; its validation/test
  are not reused as the new validation/frozen test.
- Training: checkpoint selection, temperature, and threshold use validation
  only. The trainer does not predict the frozen test.
- Frozen test: a separate command consumes already frozen policies and refuses
  existing output directories.

All output paths must be new. Runtime activation is deliberately out of scope.

## Official human-review gate

`human_review_validator.py` must pass before `build_dataset.py` can write a
training workbook. The builder requires the v0.3 guideline and contract, the
immutable source packet contract, separate completed Reviewer A/B annotation
files, and the final adjudication file. It verifies complete and unique ID
sets, reviewer identity/timestamps, label and rationale vocabularies,
agreement, adjudication, final-label linkage, and role-level binary counts.

Files explicitly identified as AI-assisted staging are rejected. File evidence
cannot prove that reviewers worked independently or that credentials belong to
the named people; those items remain recorded operational confirmations. The
supported builder entry point has no option to skip this validation.

## Human-review candidate generation

`candidate_generation.py` consumes the unlabeled 150K pool, the canonical v2
workbook, and the preliminary Pilot manifest. Retrieval heuristics and legacy
candidate sources are provenance only: no binary label is created. Frozen Test
roles are assigned before human labels, followed by Validation and Train, with
global product/exact/cluster/verified-relation isolation.

The command defaults to a read-only dry run. `--materialize` is required to
write a new bundle below the git-ignored `artifacts/p_text_v3_candidates/`
directory. The two blind reviewer packets contain only annotation ID, display
order, and content. Published Pilot exclusions and F-3.4.2 relations can be
added without changing the selection contract.
