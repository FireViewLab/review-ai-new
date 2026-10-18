# P_text Positive Control + Human-reviewed Exploratory Panel

## Purpose

Panel A checks runtime/model/config reproducibility against previous v1 predictions. Panel B descriptively observes behavior on a small human-reviewed panel. The two provenance types are not interchangeable.

## Current runtime/model

- Model: `ptext-koelectra-v1-2epoch-20260929`
- Runtime: `app.analyzers.p_text.predict_text_score`
- Inference config: `{"threshold": 0.5, "max_length": 256, "score_formula": "round((1-p_suspicious)*100)", "dataset_sha256": "cd3812b26dd7c782e01ca919c0ff12e3a8bf57e81ecc218522c7e476a7490673"}`
- Current `text_score` applies T=2; predicted label uses raw probability and threshold 0.5.

## Panel A source/provenance

- NORMAL: previous v1 `real_NORMAL_human` rows with historical `predicted_default=0`.
- SUSPICIOUS: one historical `predicted_default=1` synthetic row per synthetic type.
- Content was resolved by exact Excel row, review ID, and label checks against the v1 source workbook.
- Synthetic SUSPICIOUS is not human or external ground truth.

## Panel A results

**PASS — 10/10 labels reproduced.**

| Control | Type | Expected | Current | Current p(susp.) | Text score | |Δp| | Match |
|---|---|---|---|---:|---:|---:|---|
| A-NORMAL-01-23917a49 | previous_model_real_normal | NORMAL | NORMAL | 0.002152453 | 96 | 0.000000001 | True |
| A-NORMAL-02-a27c6973 | previous_model_real_normal | NORMAL | NORMAL | 0.002239548 | 95 | 0.000000002 | True |
| A-NORMAL-03-2a0736a9 | previous_model_real_normal | NORMAL | NORMAL | 0.002166441 | 96 | 0.000000000 | True |
| A-NORMAL-04-8d17f4b7 | previous_model_real_normal | NORMAL | NORMAL | 0.002147833 | 96 | 0.000000002 | True |
| A-NORMAL-05-00725888 | previous_model_real_normal | NORMAL | NORMAL | 0.002233678 | 95 | 0.000000002 | True |
| A-SUSPICIOUS-01-ba5d225f | COPY_VARIATION | SUSPICIOUS | SUSPICIOUS | 0.996715069 | 5 | 0.000000000 | True |
| A-SUSPICIOUS-02-7c9b5bd5 | EXPERIENCE_FREE_CLAIM | SUSPICIOUS | SUSPICIOUS | 0.996761382 | 5 | 0.000000000 | True |
| A-SUSPICIOUS-03-96a22e8d | NO_USE_ASSERTIVE_RECOMMENDATION | SUSPICIOUS | SUSPICIOUS | 0.996603608 | 6 | 0.000000000 | True |
| A-SUSPICIOUS-04-a351ab5a | PRODUCT_DESCRIPTION_REPLACEMENT | SUSPICIOUS | SUSPICIOUS | 0.996909440 | 5 | 0.000000000 | True |
| A-SUSPICIOUS-05-dae1aab3 | PROMOTIONAL_TEMPLATE | SUSPICIOUS | SUSPICIOUS | 0.996961415 | 5 | 0.000000000 | True |

## Panel B source/provenance

- NORMAL 5: SHA-256 최소 순으로 선택한 v4.3 deduplicated human-reviewed NORMAL.
- SUSPICIOUS 4: v3.2의 원래 human-reviewed SUSPICIOUS 4개와 v4.3의 exact normalized-content 교집합 전부.
- Human-reviewed labels are not externally verified authenticity/fake-review facts.

## Panel B results

Descriptive counts only; these are not Accuracy, Recall, Precision, F1, or FPR estimates.

| Panel | Human label | Length | Current label | Current p(susp.) | Text score | Label match |
|---|---|---:|---|---:|---:|---|
| B-NORMAL-01-a517332e | NORMAL | 28 | NORMAL | 0.002274785 | 95 | True |
| B-NORMAL-02-f2267bf1 | NORMAL | 30 | NORMAL | 0.002190799 | 96 | True |
| B-NORMAL-03-3ad5f056 | NORMAL | 52 | NORMAL | 0.002161954 | 96 | True |
| B-NORMAL-04-5ea8f556 | NORMAL | 33 | NORMAL | 0.002265383 | 95 | True |
| B-NORMAL-05-5897a4f4 | NORMAL | 88 | NORMAL | 0.002194897 | 96 | True |
| B-SUSPICIOUS-01-1d1573b8 | SUSPICIOUS | 140 | NORMAL | 0.004860566 | 93 | False |
| B-SUSPICIOUS-02-4a8479b2 | SUSPICIOUS | 129 | NORMAL | 0.223571420 | 65 | False |
| B-SUSPICIOUS-03-83ef0d06 | SUSPICIOUS | 109 | NORMAL | 0.002189227 | 96 | False |
| B-SUSPICIOUS-04-daf381c3 | SUSPICIOUS | 102 | NORMAL | 0.002306703 | 95 | False |
- Human-reviewed NORMAL 5건: model NORMAL 5, model SUSPICIOUS 0, unavailable 0.
- Human-reviewed SUSPICIOUS 4건: model NORMAL 4, model SUSPICIOUS 0, unavailable 0.

## Interpretation

1. Runtime/model/config historical-label reproduction: **PASS**.
2. Panel A provides no evidence that runtime mismatch explains the low-information audit's all-NORMAL result.
3. Among four human-reviewed SUSPICIOUS examples, the model returned NORMAL 4 and SUSPICIOUS 0; this is descriptive, not a recall estimate.
4. Recommended priority: **candidate expansion**.

## Limitations

- Panel A validates reproduction of ten previous-model labels, not overall model quality.
- Panel A positives are synthetic and are not real human-positive ground truth.
- Panel B has only five NORMAL and four SUSPICIOUS examples and is not representative.
- Panel B labels are human-reviewed signals, not externally verified fake/real facts.
- No formal performance metric is estimated from Panel B.

## Recommended next step

- Expand a provenance-controlled human-reviewed candidate set before calibration; the four-case panel is too small for a calibration decision.
- Keep calibration, candidate expansion, and runtime investigation as separate hypotheses.
