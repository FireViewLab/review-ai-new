# Human Review Round 1

- Review round: 1
- Review type: human
- Dataset: `synthetic_expansion_v1`
- Total pairs: 30
- APPROVE: 20
- REVISE: 10
- REJECT: 0

This log records the user's Human Review Round 1 decision. It is not a Codex
approval. Rows retain their synthetic and LLM-generated provenance.

## APPROVE

`p001`, `p002`, `p003`, `p004`, `p005`, `p006`, `p007`, `p008`, `p009`,
`p011`, `p014`, `p015`, `p016`, `p017`, `p018`, `p019`, `p020`, `p021`,
`p022`, `p023`

The content reviewed in Round 1 is preserved. These pairs may record
`human_reviewed=true` and `human_approved=true`; `source_type=synthetic` and
`is_llm_generated=true` remain unchanged.

## REVISE

The versions reviewed in Round 1 received a REVISE decision. The revised
versions are awaiting human re-review and therefore remain
`human_reviewed=false` and `human_approved=false`.

- `p010`: change the SUSPICIOUS `praise_intensity` from `extreme` to `medium`.
- `p012`: make the SUSPICIOUS sentence sound like a natural user review while
  retaining the uneaten/aroma-only evidence gap and the taste/quality claim.
- `p013`: clarify the expansion from a single experience to sustained or broad
  benefit and a recommendation; do not treat the immediate personal feeling
  itself as suspicious.
- `p024`: replace product-description wording with natural review language
  while retaining unused/specification-only evidence and a performance claim.
- `p025`: replace wording that exposes the dataset design while retaining the
  unworn-to-all-day-comfort claim.
- `p026`: remove artificial contrastive wording while retaining no evidence
  and universal exaggerated praise.
- `p027`: retain strong evaluative language but make the review natural while
  preserving the one-sip-to-quality/benefit overreach.
- `p028`: replace the unnatural pre-use satisfaction wording while preserving
  the unused-to-all-situations performance claim.
- `p029`: improve the unnatural connective while preserving the single local
  application-to-multiple-benefits/universal-skin claim.
- `p030`: make the exaggerated synthetic wording more review-like while
  preserving the photo/package-only-to-taste/freshness/universal claim.

## Status semantics

Round 1 is a single-human review. `review_status=AGREED` is not used because
that status is reserved for two-human agreement. All rows remain `DRAFT`, and
no row is training-ready.
