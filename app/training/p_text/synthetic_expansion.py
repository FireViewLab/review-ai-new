"""Validation helpers for the isolated synthetic expansion v1 dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.training.p_text.contrastive_pilot import (
    CROSS_CORPUS_REPORT_FLOOR,
    EVIDENCE_LEVELS,
    EXPORT_COLUMNS,
    ForbiddenRecord,
    HEALTH_KEYWORDS,
    NEAR_TEMPLATE_THRESHOLD,
    PRAISE_INTENSITIES,
    CrossCorpusMatch,
    _cross_family_near_templates,
    _duplicate_groups,
    _keyword_only_pair_candidates,
    _label_conflicts,
    cross_corpus_near_template_matches,
    health_keyword_balance,
    normalized_text,
)


EXPANSION_CATEGORIES = (
    "EXPERIENCE_FREE_CLAIM",
    "UNUSED_RECOMMENDATION",
    "NATURAL_PROMOTIONAL",
    "PROMOTION_EXPERIENCE_MIX",
    "FEATURE_AS_PERSONAL_EXPERIENCE",
    "EXAGGERATED_PRAISE_WITHOUT_EVIDENCE",
)
EXPANSION_DOMAINS = frozenset(
    {"cosmetics", "health_supplement", "pharma_otc", "food", "fashion", "household", "electronics"}
)
EXPECTED_PAIR_COUNT = 30
EXPECTED_ROWS = 60
EXPECTED_PAIRS_PER_CATEGORY = 5
MAX_FAMILY_PAIRS = 2


@dataclass
class ExpansionValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    cross_corpus_matches: list[CrossCorpusMatch] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def validate_synthetic_expansion(
    rows: Sequence[Mapping[str, Any]],
    *,
    forbidden_records: Sequence[ForbiddenRecord] = (),
) -> ExpansionValidationReport:
    """Validate expansion structure, provenance, duplication, and leakage."""

    report = ExpansionValidationReport()
    required = set(EXPORT_COLUMNS)
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        missing = sorted(required.difference(row))
        if missing:
            report.errors.append(f"row {index}: missing required columns: {', '.join(missing)}")
        pairs[str(row.get("parent_pair_id", ""))].append(row)

    if len(rows) != EXPECTED_ROWS:
        report.errors.append(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len(pairs) != EXPECTED_PAIR_COUNT:
        report.errors.append(f"expected {EXPECTED_PAIR_COUNT} pairs, got {len(pairs)}")

    ids = [str(row.get("sample_id", "")) for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if not key or count > 1)
    if duplicate_ids:
        report.errors.append(f"invalid or duplicate sample_id: {duplicate_ids}")

    category_pairs: Counter[str] = Counter()
    domain_pairs: Counter[str] = Counter()
    difficulty_pairs: Counter[str] = Counter()
    family_pairs: dict[str, set[str]] = defaultdict(set)
    for pair_id, pair_rows in pairs.items():
        if not pair_id:
            report.errors.append("blank parent_pair_id")
            continue
        if len(pair_rows) != 2:
            report.errors.append(f"{pair_id}: expected exactly 2 rows, got {len(pair_rows)}")
            continue
        labels = Counter(str(row.get("label", "")) for row in pair_rows)
        if labels != Counter({"NORMAL": 1, "SUSPICIOUS": 1}):
            report.errors.append(f"{pair_id}: pair must contain one NORMAL and one SUSPICIOUS")
        for name in ("category", "domain", "difficulty", "generation_family_id"):
            if len({str(row.get(name, "")) for row in pair_rows}) != 1:
                report.errors.append(f"{pair_id}: pair {name} mismatch")
        category = str(pair_rows[0].get("category", ""))
        domain = str(pair_rows[0].get("domain", ""))
        difficulty = str(pair_rows[0].get("difficulty", ""))
        family = str(pair_rows[0].get("generation_family_id", ""))
        category_pairs[category] += 1
        domain_pairs[domain] += 1
        difficulty_pairs[difficulty] += 1
        family_pairs[family].add(pair_id)

    for category in EXPANSION_CATEGORIES:
        if category_pairs[category] != EXPECTED_PAIRS_PER_CATEGORY:
            report.errors.append(
                f"{category}: expected {EXPECTED_PAIRS_PER_CATEGORY} pairs, got {category_pairs[category]}"
            )
    unknown_categories = sorted(set(category_pairs).difference(EXPANSION_CATEGORIES))
    if unknown_categories:
        report.errors.append(f"unknown categories: {unknown_categories}")
    unknown_domains = sorted(set(domain_pairs).difference(EXPANSION_DOMAINS))
    if unknown_domains:
        report.errors.append(f"unknown domains: {unknown_domains}")
    if domain_pairs["health_supplement"] < 6:
        report.errors.append("health_supplement requires at least 6 pairs")
    if domain_pairs["pharma_otc"] < 4:
        report.errors.append("pharma_otc requires at least 4 pairs")
    if domain_pairs and max(domain_pairs.values()) / max(len(pairs), 1) > 0.30:
        report.errors.append("domain concentration exceeds 30% of pairs")

    family_violations = {
        family: len(pair_ids) for family, pair_ids in family_pairs.items()
        if not family or len(pair_ids) > MAX_FAMILY_PAIRS
    }
    if family_violations:
        report.errors.append(f"generation family limit exceeded: {family_violations}")

    for index, row in enumerate(rows, start=1):
        label = str(row.get("label", ""))
        category = str(row.get("category", ""))
        praise = row.get("praise_intensity")
        evidence = row.get("evidence_level")
        if not str(row.get("content", "")).strip():
            report.errors.append(f"row {index}: content is blank")
        if label not in {"NORMAL", "SUSPICIOUS"}:
            report.errors.append(f"row {index}: invalid label")
        if row.get("source_type") != "synthetic":
            report.errors.append(f"row {index}: source_type must be synthetic")
        if row.get("is_llm_generated") is not True:
            report.errors.append(f"row {index}: is_llm_generated must be true")
        if row.get("human_reviewed") is not False:
            report.errors.append(f"row {index}: human_reviewed must be false")
        if row.get("human_approved") is not False:
            report.errors.append(f"row {index}: human_approved must be false")
        if row.get("review_status") != "DRAFT":
            report.errors.append(f"row {index}: review_status must be DRAFT")
        if str(row.get("split", "")):
            report.errors.append(f"row {index}: split must remain unassigned")
        if praise not in PRAISE_INTENSITIES:
            report.errors.append(f"row {index}: invalid praise_intensity {praise!r}")
        if evidence not in EVIDENCE_LEVELS:
            report.errors.append(f"row {index}: invalid evidence_level {evidence!r}")
        if category == "EXAGGERATED_PRAISE_WITHOUT_EVIDENCE" and label == "SUSPICIOUS":
            if praise != "extreme" or evidence not in {"none", "limited"}:
                report.errors.append(
                    f"row {index}: suspicious exaggerated praise requires extreme praise and none/limited evidence"
                )

    exact_groups = _duplicate_groups(rows, lambda text: text)
    normalized_groups = _duplicate_groups(rows, normalized_text)
    conflicts = _label_conflicts(rows)
    keyword_only = _keyword_only_pair_candidates(pairs)
    cross_family = _cross_family_near_templates(rows)
    if exact_groups:
        report.errors.append(f"exact duplicate groups: {len(exact_groups)}")
    if normalized_groups:
        report.errors.append(f"normalized duplicate groups: {len(normalized_groups)}")
    if conflicts:
        report.errors.append(f"label conflict groups: {len(conflicts)}")
    if keyword_only:
        report.errors.append(f"keyword-only minimal pair candidates: {', '.join(keyword_only)}")
    if cross_family:
        report.errors.append(f"cross-family near-template candidates: {len(cross_family)}")

    matches = cross_corpus_near_template_matches(rows, forbidden_records)
    report.cross_corpus_matches = matches
    forbidden_exact = {record.content for record in forbidden_records if record.content}
    forbidden_normalized = {normalized_text(text) for text in forbidden_exact}
    exact_overlap = [row["sample_id"] for row in rows if row["content"] in forbidden_exact]
    normalized_overlap = [
        row["sample_id"] for row in rows if normalized_text(str(row["content"])) in forbidden_normalized
    ]
    if exact_overlap:
        report.errors.append(f"forbidden exact overlap: {len(exact_overlap)}")
    if normalized_overlap:
        report.errors.append(f"forbidden normalized overlap: {len(normalized_overlap)}")
    protected = [m for m in matches if m.candidate and m.corpus_kind in {"blind", "challenge", "pilot"}]
    if protected:
        report.errors.append(f"protected corpus similarity >= {NEAR_TEMPLATE_THRESHOLD}: {len(protected)}")
    training = [m for m in matches if m.candidate and m.corpus_kind == "training"]
    if training:
        report.warnings.append(f"training similarity >= {NEAR_TEMPLATE_THRESHOLD}: {len(training)}")

    keyword_balance = health_keyword_balance(rows)
    for keyword, label_counts in keyword_balance.items():
        populated_labels = [label for label, count in label_counts.items() if count]
        if len(populated_labels) == 1 and sum(label_counts.values()) >= 2:
            report.warnings.append(
                f"keyword {keyword!r} appears only in {populated_labels[0]} rows: {label_counts}"
            )

    report.counts = {
        "rows": len(rows),
        "pairs": len(pairs),
        "category_pairs": dict(category_pairs),
        "domain_pairs": dict(domain_pairs),
        "difficulty_pairs": dict(difficulty_pairs),
        "praise_intensity_rows": dict(Counter(str(row.get("praise_intensity", "")) for row in rows)),
        "evidence_level_rows": dict(Counter(str(row.get("evidence_level", "")) for row in rows)),
        "synthetic_rows": sum(row.get("source_type") == "synthetic" for row in rows),
        "llm_generated_rows": sum(row.get("is_llm_generated") is True for row in rows),
        "human_reviewed_rows": sum(row.get("human_reviewed") is True for row in rows),
        "human_approved_rows": sum(row.get("human_approved") is True for row in rows),
        "training_ready_rows": sum(row.get("review_status") in {"AGREED", "ADJUDICATED"} for row in rows),
        "exact_duplicate_groups": len(exact_groups),
        "normalized_duplicate_groups": len(normalized_groups),
        "label_conflict_groups": len(conflicts),
        "keyword_only_pair_candidates": len(keyword_only),
        "cross_family_near_template_candidates": len(cross_family),
        "generation_family_violations": len(family_violations),
        "forbidden_exact_overlap": len(exact_overlap),
        "forbidden_normalized_overlap": len(normalized_overlap),
        "cross_corpus_similarity_bands": dict(Counter(m.band for m in matches)),
        "cross_corpus_ge_085_by_kind": dict(Counter(m.corpus_kind for m in matches if m.candidate)),
        "health_keyword_balance": keyword_balance,
    }
    return report
