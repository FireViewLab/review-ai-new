# Synthetic Expansion v1

This directory contains the isolated P_text synthetic expansion v1 candidate
set and remains separate from `pilot_v1`.  The source of truth is
`annotation_synthetic_expansion_v1.jsonl`; the CSV file is a human-review view.

The expansion contains 30 contrastive pairs (60 rows), with one NORMAL and one
SUSPICIOUS candidate per pair.  All rows remain `DRAFT`, unassigned to a split,
and unavailable for training until the project annotation process is complete.

New rows must follow `schema.json`.  AI-generated rows always retain
`source_type=synthetic` and `is_llm_generated=true`, even after human review or
approval. `human_reviewed` and `human_approved` describe review state; they do
not rewrite generation provenance.

The label must be supported by a relationship visible in `content`.  Praise or
health-related keywords alone never determine the label.  Before any row is
used for training, it must pass the improvement validator, duplicate/leakage
checks, and the project annotation process.
