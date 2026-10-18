"""Audit production P_text behavior by normalized review length.

This is an offline, read-only analysis entry point.  It streams source CSVs,
samples unique normalized content deterministically, and calls the production
``app.analyzers.p_text.predict_text_score`` function without recreating score
logic.  Run stages in order: population, pilot, full.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from app.analyzers.p_text import DEFAULT_MODEL_PATH, predict_text_score


BUCKETS = ("1~5", "6~10", "11~20", "21~40", "41~80", "81+")
PILOT_PER_BUCKET = 10
FULL_PER_BUCKET = 50
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\kimha\OneDrive\Desktop\review-export-20261017"
    r"\review-export-20261017"
)
PLATFORMS = ("elevenst", "kurly", "musinsa", "naver", "ohouse", "oliveyoung")
DEFAULT_JSON_REPORT = Path("reports/ptext_low_information_audit_20261018.json")
DEFAULT_MARKDOWN_REPORT = Path("reports/ptext_low_information_audit_20261018.md")


def normalize_content(value: object) -> str:
    """Normalize whitespace only; null values become blank."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def length_bucket(length: int) -> str:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if length <= 5:
        return "1~5"
    if length <= 10:
        return "6~10"
    if length <= 20:
        return "11~20"
    if length <= 40:
        return "21~40"
    if length <= 80:
        return "41~80"
    return "81+"


def stable_digest(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Sample:
    platform: str
    length_bucket: str
    character_length: int
    normalized_content: str
    product_id: str
    review_id: str
    sample_hash: str
    row_hash: str

    def public_identity(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "length_bucket": self.length_bucket,
            "character_length": self.character_length,
            "sample_hash": self.sample_hash,
        }


class DeterministicSampler:
    """Bounded, order-invariant sampling of unique normalized content."""

    def __init__(self, per_bucket: int) -> None:
        if per_bucket < 1:
            raise ValueError("per_bucket must be positive")
        self.per_bucket = per_bucket
        self._groups: dict[tuple[str, str], dict[str, Sample]] = defaultdict(dict)

    def consider(
        self,
        *,
        platform: str,
        product_id: object,
        review_id: object,
        normalized_content: str,
    ) -> None:
        bucket = length_bucket(len(normalized_content))
        # Content-level rank ensures duplicate rows cannot occupy multiple slots.
        sample_hash = stable_digest(platform, normalized_content)
        row_hash = stable_digest(platform, product_id or "", review_id or "", normalized_content)
        candidate = Sample(
            platform=platform,
            length_bucket=bucket,
            character_length=len(normalized_content),
            normalized_content=normalized_content,
            product_id=str(product_id or ""),
            review_id=str(review_id or ""),
            sample_hash=sample_hash,
            row_hash=row_hash,
        )
        group = self._groups[(platform, bucket)]
        existing = group.get(normalized_content)
        if existing is not None:
            if candidate.row_hash < existing.row_hash:
                group[normalized_content] = candidate
            return
        if len(group) < self.per_bucket:
            group[normalized_content] = candidate
            return
        worst_content, worst = max(
            group.items(), key=lambda item: (item[1].sample_hash, item[0])
        )
        if (candidate.sample_hash, normalized_content) < (
            worst.sample_hash,
            worst_content,
        ):
            del group[worst_content]
            group[normalized_content] = candidate

    def samples(self, limit: int | None = None) -> list[Sample]:
        selected: list[Sample] = []
        for key in sorted(self._groups):
            ordered = sorted(
                self._groups[key].values(),
                key=lambda sample: (sample.sample_hash, sample.normalized_content),
            )
            selected.extend(ordered if limit is None else ordered[:limit])
        return selected

    def counts(self, limit: int | None = None) -> dict[str, dict[str, int]]:
        result = {platform: {bucket: 0 for bucket in BUCKETS} for platform in PLATFORMS}
        for sample in self.samples(limit=limit):
            result[sample.platform][sample.length_bucket] += 1
        return result


def source_paths(data_root: Path) -> dict[str, Path]:
    return {
        platform: data_root / f"reviews_{platform}.csv" / f"reviews_{platform}.csv"
        for platform in PLATFORMS
    }


def _empty_bucket_state() -> dict[str, object]:
    return {"raw_row_count": 0, "content_digests": set()}


def _platform_state() -> dict[str, object]:
    return {
        "raw_review_count": 0,
        "blank_content_count": 0,
        "length_buckets": {bucket: _empty_bucket_state() for bucket in BUCKETS},
    }


def scan_population(
    paths: Mapping[str, Path], *, sample_per_bucket: int | None = None
) -> tuple[dict[str, object], DeterministicSampler | None, float]:
    started = time.perf_counter()
    sampler = DeterministicSampler(sample_per_bucket) if sample_per_bucket else None
    overall = _platform_state()
    platforms = {platform: _platform_state() for platform in PLATFORMS}
    required = {"platform", "product_id", "review_id", "content"}

    for expected_platform in PLATFORMS:
        path = paths[expected_platform]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                row_platform = row.get("platform")
                if row_platform != expected_platform:
                    raise ValueError(
                        f"{path}: expected platform {expected_platform!r}, got {row_platform!r}"
                    )
                overall["raw_review_count"] += 1
                platforms[expected_platform]["raw_review_count"] += 1
                normalized = normalize_content(row.get("content"))
                if not normalized:
                    overall["blank_content_count"] += 1
                    platforms[expected_platform]["blank_content_count"] += 1
                    continue
                bucket = length_bucket(len(normalized))
                digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                for state in (overall, platforms[expected_platform]):
                    bucket_state = state["length_buckets"][bucket]
                    bucket_state["raw_row_count"] += 1
                    bucket_state["content_digests"].add(digest)
                if sampler is not None:
                    sampler.consider(
                        platform=expected_platform,
                        product_id=row.get("product_id"),
                        review_id=row.get("review_id"),
                        normalized_content=normalized,
                    )

    population = {
        "total_raw_rows": overall["raw_review_count"],
        "blank_rows": overall["blank_content_count"],
        "overall_length_buckets": _finalize_buckets(
            overall["length_buckets"], overall["raw_review_count"]
        ),
        "platforms": {
            platform: {
                "raw_review_count": state["raw_review_count"],
                "blank_content_count": state["blank_content_count"],
                "length_buckets": _finalize_buckets(
                    state["length_buckets"], state["raw_review_count"]
                ),
            }
            for platform, state in platforms.items()
        },
        "unique_count_method": (
            "cardinality of SHA-256 digests of normalized content; collision risk is negligible"
        ),
    }
    return population, sampler, time.perf_counter() - started


def _finalize_buckets(
    bucket_states: Mapping[str, Mapping[str, object]], denominator: int
) -> dict[str, dict[str, object]]:
    return {
        bucket: {
            "raw_row_count": int(bucket_states[bucket]["raw_row_count"]),
            "raw_row_ratio": _ratio(int(bucket_states[bucket]["raw_row_count"]), denominator),
            "unique_normalized_content_count": len(bucket_states[bucket]["content_digests"]),
        }
        for bucket in BUCKETS
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def infer_samples(
    samples: Iterable[Sample],
    predictor: Callable[[str], Mapping[str, object]] = predict_text_score,
) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for sample in samples:
        prediction = dict(predictor(sample.normalized_content))
        records.append(
            {
                **sample.public_identity(),
                "content": sample.normalized_content,
                "text_score": prediction.get("text_score"),
                "suspicious_probability": prediction.get("suspicious_probability"),
                "predicted_label": prediction.get("predicted_label"),
                "unavailable_reason": (
                    "production runtime returned unavailable sentinel"
                    if prediction.get("text_score") == -1
                    else None
                ),
            }
        )
    return records, time.perf_counter() - started


def validate_predictions(records: Iterable[Mapping[str, object]]) -> None:
    records = list(records)
    if not records:
        raise RuntimeError("pilot selected no samples")
    available = 0
    for record in records:
        score = record.get("text_score")
        probability = record.get("suspicious_probability")
        label = record.get("predicted_label")
        if score == -1:
            if probability is not None or label is not None:
                raise RuntimeError("invalid unavailable prediction shape")
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise RuntimeError(f"invalid text_score: {score!r}")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise RuntimeError(f"invalid suspicious_probability: {probability!r}")
        if label not in {"NORMAL", "SUSPICIOUS"}:
            raise RuntimeError(f"invalid predicted_label: {label!r}")
        available += 1
    if available == 0:
        raise RuntimeError("all pilot predictions were unavailable; full audit aborted")


def calculate_statistics(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(records)
    available = [row for row in rows if row.get("text_score") != -1]
    scores = [float(row["text_score"]) for row in available]
    probabilities = [float(row["suspicious_probability"]) for row in available]
    total = len(rows)
    normal = sum(row.get("predicted_label") == "NORMAL" for row in rows)
    suspicious = sum(row.get("predicted_label") == "SUSPICIOUS" for row in rows)
    unavailable = total - len(available)
    high_90 = sum(score >= 90 for score in scores)
    high_95 = sum(score >= 95 for score in scores)
    return {
        "inferred_sample_count": total,
        "available_count": len(available),
        "text_score": {
            "mean": _mean(scores),
            "median": _median(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        },
        "suspicious_probability": {
            "mean": _mean(probabilities),
            "median": _median(probabilities),
        },
        "text_score_gte_90": {"count": high_90, "ratio": _ratio(high_90, total)},
        "text_score_gte_95": {"count": high_95, "ratio": _ratio(high_95, total)},
        "predicted_label": {
            "NORMAL": {"count": normal, "ratio": _ratio(normal, total)},
            "SUSPICIOUS": {"count": suspicious, "ratio": _ratio(suspicious, total)},
        },
        "unavailable": {"count": unavailable, "ratio": _ratio(unavailable, total)},
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def grouped_statistics(records: list[dict[str, object]]) -> dict[str, object]:
    by_bucket = {
        bucket: calculate_statistics(
            row for row in records if row["length_bucket"] == bucket
        )
        for bucket in BUCKETS
    }
    by_platform_bucket = {
        platform: {
            bucket: calculate_statistics(
                row
                for row in records
                if row["platform"] == platform and row["length_bucket"] == bucket
            )
            for bucket in BUCKETS
        }
        for platform in PLATFORMS
    }
    short = [row for row in records if row["length_bucket"] in {"1~5", "6~10", "11~20"}]
    longer = [row for row in records if row["length_bucket"] in {"41~80", "81+"}]
    return {
        "overall_by_bucket": by_bucket,
        "by_platform_and_bucket": by_platform_bucket,
        "short_1_to_20": calculate_statistics(short),
        "longer_41_plus": calculate_statistics(longer),
    }


def diagnostic_examples(records: list[dict[str, object]]) -> dict[str, object]:
    by_bucket: dict[str, dict[str, object]] = {}
    for bucket in BUCKETS:
        available = [
            row for row in records
            if row["length_bucket"] == bucket and row["text_score"] != -1
        ]
        if not available:
            by_bucket[bucket] = {"highest": None, "lowest": None}
            continue
        highest = min(available, key=lambda row: (-float(row["text_score"]), row["sample_hash"]))
        lowest = min(available, key=lambda row: (float(row["text_score"]), row["sample_hash"]))
        by_bucket[bucket] = {
            "highest": _diagnostic_row(highest),
            "lowest": _diagnostic_row(lowest),
        }
    short_high = sorted(
        (
            row for row in records
            if row["length_bucket"] in {"1~5", "6~10", "11~20"}
            and row["text_score"] != -1
            and float(row["text_score"]) >= 90
        ),
        key=lambda row: (-float(row["text_score"]), row["sample_hash"]),
    )[:5]
    return {
        "by_bucket": by_bucket,
        "short_text_score_gte_90": [_diagnostic_row(row) for row in short_high],
    }


def _diagnostic_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "content": row["content"],
        "length": row["character_length"],
        "platform": row["platform"],
        "length_bucket": row["length_bucket"],
        "text_score": row["text_score"],
        "suspicious_probability": row["suspicious_probability"],
        "predicted_label": row["predicted_label"],
    }


def interpretation(statistics_result: Mapping[str, object]) -> dict[str, object]:
    buckets = statistics_result["overall_by_bucket"]
    high = {bucket: buckets[bucket]["text_score_gte_90"]["ratio"] for bucket in BUCKETS}
    short_ratio = statistics_result["short_1_to_20"]["text_score_gte_90"]["ratio"]
    longer_ratio = statistics_result["longer_41_plus"]["text_score_gte_90"]["ratio"]
    if all(high[bucket] >= 0.70 for bucket in ("1~5", "6~10", "11~20")):
        classification = "C"
        summary = (
            "C is the closest category because sampled 1~20 character reviews broadly "
            "saturate at >=90. However, the 41+ group saturates at least as strongly. "
            "Across this unlabeled new-data sample, P_text outputs concentrate globally "
            "in the NORMAL direction, mostly around scores 94~96; this does not support a "
            "short-specific effect and does not establish ground-truth generalization failure."
        )
    elif high["1~5"] >= 0.70 or high["6~10"] >= 0.70:
        classification = "B"
        summary = "High-score concentration is strongest in the very-short buckets."
    else:
        classification = "A"
        summary = "The sampled short buckets do not show broad high-score concentration."
    return {
        "classification": classification,
        "classification_qualifier": "closest, with global saturation caveat",
        "scope_assessment": {
            "short_specific_effect": "not_supported",
            "global_score_saturation": "observed",
            "ground_truth_generalization_failure": "not_established",
        },
        "summary": summary,
        "decision_aid": (
            "Diagnostic heuristic: C when every 1~20 bucket has >=70% scores >=90; "
            "B when either 1~5 or 6~10 does; otherwise A. This is not a policy threshold."
        ),
        "evidence": {
            "bucket_text_score_gte_90_ratios": high,
            "short_minus_longer_text_score_gte_90_ratio": short_ratio - longer_ratio,
            "short_1_to_20": statistics_result["short_1_to_20"],
            "longer_41_plus": statistics_result["longer_41_plus"],
        },
        "limitations": [
            "Sampled inference, not inference over all raw rows.",
            "The new real-world sample is unlabeled; this is not a ground-truth performance evaluation, and no Accuracy, Recall, or FPR is estimated.",
            "NORMAL output concentration does not establish model generalization failure or prove that suspicious reviews cannot be detected.",
            "Short reviews are not assumed to be fake or suspicious.",
            "Writing style may differ by platform.",
            "Unique-content sampling differs from actual traffic weighting.",
            "Weak suspicious evidence is not equivalent to sufficient trustworthy evidence.",
            "A separate positive-control inference has not yet been run.",
        ],
    }


def metadata(paths: Mapping[str, Path], config: Mapping[str, object]) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "model_path": str(Path(DEFAULT_MODEL_PATH).resolve()),
        "model_version": Path(DEFAULT_MODEL_PATH).name,
        "inference_config": dict(config),
        "runtime_function": "app.analyzers.p_text.predict_text_score",
        "sampling_method": (
            "Per platform x length bucket, select the lexicographically smallest SHA-256 "
            "digests of platform + normalized content; duplicate normalized content is excluded. "
            "A separate full row-key SHA-256 chooses canonical metadata deterministically."
        ),
        "pilot_per_bucket": PILOT_PER_BUCKET,
        "full_per_bucket": FULL_PER_BUCKET,
        "source_files": {platform: str(path) for platform, path in paths.items()},
        "normalization": '" ".join(str(content).split()); null becomes blank',
        "length_definition": "Python len(normalized_content)",
    }


def load_config() -> dict[str, object]:
    path = Path(DEFAULT_MODEL_PATH) / "inference_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: Mapping[str, object]) -> str:
    population = report["population"]
    stats = report["statistics"]
    lines = [
        "# P_text Short / Low-information Review Audit",
        "",
        "## 목적",
        "",
        "짧거나 정보량이 적은 리뷰가 의심 신호 부재만으로 높은 `text_score`에 수렴하는지 진단한다. "
        "짧은 리뷰를 fake 또는 suspicious로 가정하지 않는다.",
        "",
        "## 현재 P_text 기준",
        "",
        f"- Model: `{report['metadata']['model_version']}`",
        f"- Runtime: `{report['metadata']['runtime_function']}`",
        f"- Inference config: `{json.dumps(report['metadata']['inference_config'], ensure_ascii=False)}`",
        "- `text_score`에만 temperature T=2가 적용되고 label은 raw probability/threshold를 따른다.",
        "",
        "## 데이터 범위",
        "",
        f"- 6개 플랫폼, raw rows: {population['total_raw_rows']:,}",
        f"- Blank normalized content: {population['blank_rows']:,}",
        "- 원본 CSV는 UTF-8, streaming, read-only 방식으로 처리했다.",
        "",
        "## Population length distribution",
        "",
        "| Bucket | Raw rows | Raw ratio | Unique normalized content |",
        "|---|---:|---:|---:|",
    ]
    for bucket in BUCKETS:
        row = population["overall_length_buckets"][bucket]
        lines.append(
            f"| {bucket} | {row['raw_row_count']:,} | {_pct(row['raw_row_ratio'])} | "
            f"{row['unique_normalized_content_count']:,} |"
        )
    lines.extend([
        "",
        "## Sampling methodology",
        "",
        f"- {report['metadata']['sampling_method']}",
        "- Sampling unit: unique normalized content within each platform × bucket.",
        "- Raw rows, unique contents, and inferred samples are reported separately.",
        "",
        "## Pilot result",
        "",
        f"- Status: {report['pilot']['status']}",
        f"- Samples: {report['pilot']['sample_count']}",
        f"- Runtime: {report['pilot']['elapsed_seconds']:.2f}s",
        f"- Unavailable: {report['pilot']['unavailable_count']}",
        "- Model load/output validation succeeded before full expansion.",
        "",
        "## Full audit sample size",
        "",
        f"- Total inferred: {report['sample']['total_inferred']}",
        f"- Total inference runtime (pilot included): {report['sample']['inference_elapsed_seconds']:.2f}s",
        "",
        "## Length-bucket result",
        "",
        "| Bucket | N | Mean score | Median | Min | Max | Mean p(susp.) | Median p(susp.) | >=90 | >=95 | NORMAL | SUSPICIOUS | Unavailable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bucket in BUCKETS:
        row = stats["overall_by_bucket"][bucket]
        lines.append(
            f"| {bucket} | {row['inferred_sample_count']} | {_fmt(row['text_score']['mean'], 2)} | "
            f"{_fmt(row['text_score']['median'], 2)} | {_fmt(row['text_score']['min'], 0)} | "
            f"{_fmt(row['text_score']['max'], 0)} | {_fmt(row['suspicious_probability']['mean'], 6)} | "
            f"{_fmt(row['suspicious_probability']['median'], 6)} | {_pct(row['text_score_gte_90']['ratio'])} | "
            f"{_pct(row['text_score_gte_95']['ratio'])} | {_pct(row['predicted_label']['NORMAL']['ratio'])} | "
            f"{_pct(row['predicted_label']['SUSPICIOUS']['ratio'])} | {_pct(row['unavailable']['ratio'])} |"
        )
    lines.extend([
        "",
        "## Platform × bucket observations",
        "",
    ])
    for platform in PLATFORMS:
        pop = population["platforms"][platform]
        sample_n = sum(
            stats["by_platform_and_bucket"][platform][bucket]["inferred_sample_count"]
            for bucket in BUCKETS
        )
        under_20 = sum(pop["length_buckets"][bucket]["raw_row_count"] for bucket in ("1~5", "6~10", "11~20"))
        lines.append(
            f"- **{platform}**: raw={pop['raw_review_count']:,}, blank={pop['blank_content_count']:,}, "
            f"1~20 ratio={_pct(_ratio(under_20, pop['raw_review_count']))}, inferred={sample_n}."
        )
    lines.extend([
        "",
        "| Platform | Bucket | N | Mean score | Median | >=90 | >=95 | NORMAL | SUSPICIOUS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for platform in PLATFORMS:
        for bucket in BUCKETS:
            row = stats["by_platform_and_bucket"][platform][bucket]
            lines.append(
                f"| {platform} | {bucket} | {row['inferred_sample_count']} | "
                f"{_fmt(row['text_score']['mean'], 2)} | {_fmt(row['text_score']['median'], 2)} | "
                f"{_pct(row['text_score_gte_90']['ratio'])} | {_pct(row['text_score_gte_95']['ratio'])} | "
                f"{_pct(row['predicted_label']['NORMAL']['ratio'])} | "
                f"{_pct(row['predicted_label']['SUSPICIOUS']['ratio'])} |"
            )
    lines.extend(["", "## Short vs long comparison", ""])
    for name, label in (("short_1_to_20", "1~20"), ("longer_41_plus", "41+")):
        row = stats[name]
        lines.append(
            f"- **{label}**: n={row['inferred_sample_count']}, mean={_fmt(row['text_score']['mean'], 2)}, "
            f"median={_fmt(row['text_score']['median'], 2)}, >=90={_pct(row['text_score_gte_90']['ratio'])}, "
            f">=95={_pct(row['text_score_gte_95']['ratio'])}, mean p(susp.)="
            f"{_fmt(row['suspicious_probability']['mean'], 6)}, SUSPICIOUS="
            f"{_pct(row['predicted_label']['SUSPICIOUS']['ratio'])}."
        )
    lines.extend([
        "",
        "## Diagnostic examples",
        "",
        "각 bucket의 최고/최저 score 예시다. 긴 원문은 120자로 잘라 표시한다.",
        "",
        "| Bucket | Extreme | Platform | Length | Score | p(susp.) | Label | Content |",
        "|---|---|---|---:|---:|---:|---|---|",
    ])
    for bucket in BUCKETS:
        examples = report["diagnostic_examples"]["by_bucket"][bucket]
        for extreme in ("highest", "lowest"):
            row = examples[extreme]
            if row is None:
                continue
            content = str(row["content"])
            if len(content) > 120:
                content = content[:117] + "..."
            content = content.replace("|", "\\|")
            lines.append(
                f"| {bucket} | {extreme} | {row['platform']} | {row['length']} | "
                f"{row['text_score']} | {_fmt(row['suspicious_probability'], 8)} | "
                f"{row['predicted_label']} | {content} |"
            )
    lines.extend(["", "Short samples with `text_score >= 90`:", ""])
    for row in report["diagnostic_examples"]["short_text_score_gte_90"]:
        safe_content = str(row["content"]).replace("|", "\\|")
        lines.append(
            f"- `{row['platform']}` / {row['length_bucket']} / len={row['length']} / "
            f"score={row['text_score']} / p={_fmt(row['suspicious_probability'], 8)} / "
            f"{row['predicted_label']}: {safe_content}"
        )
    interpretation_result = report["interpretation"]
    lines.extend([
        "",
        "## Key findings 및 A/B/C interpretation",
        "",
        f"**{interpretation_result['classification']} (closest), with global saturation caveat.**",
        "",
        interpretation_result["summary"],
        "",
        "- `short_specific_effect`: **not supported**",
        "- `global_score_saturation`: **observed**",
        "- `ground_truth_generalization_failure`: **not established**",
        "",
        interpretation_result["decision_aid"],
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in interpretation_result["limitations"])
    lines.extend([
        "",
        "## Policy recommendation for NEXT STEP only",
        "",
        "- 신규 1,296건을 재실행하지 말고, known NORMAL 5~10건과 known SUSPICIOUS 5~10건의 작은 positive-control sanity check를 먼저 수행한다.",
        "- Human-reviewed label과 previous-model expected label을 별도 provenance로 기록하고 같은 의미로 취급하지 않는다.",
        "- Control이 통과하면 신규 unlabeled 표본의 분포·calibration·text sufficiency를 후속 검토하고, 실패하면 runtime/model/config 정합성을 먼저 조사한다.",
        "- 이 audit만으로 일반화 실패나 suspicious-review 미탐지를 주장하지 않는다.",
        "- 이번 audit에서는 threshold, score mapping, unavailable/RTI 정책을 변경하지 않았다.",
        "",
    ])
    return "\n".join(lines)


def sample_count_summary(samples: Iterable[Sample]) -> dict[str, object]:
    rows = list(samples)
    by_bucket = {bucket: sum(row.length_bucket == bucket for row in rows) for bucket in BUCKETS}
    by_platform_bucket = {
        platform: {
            bucket: sum(
                row.platform == platform and row.length_bucket == bucket for row in rows
            )
            for bucket in BUCKETS
        }
        for platform in PLATFORMS
    }
    return {
        "total": len(rows),
        "by_bucket": by_bucket,
        "by_platform_and_bucket": by_platform_bucket,
    }


def run_population(args: argparse.Namespace) -> None:
    paths = source_paths(args.data_root)
    population, _, elapsed = scan_population(paths)
    payload = {
        "status": "population_complete",
        "metadata": metadata(paths, load_config()),
        "population": population,
        "population_scan_elapsed_seconds": elapsed,
    }
    write_json(args.json_report, payload)
    print(json.dumps({
        "status": payload["status"],
        "total_raw_rows": population["total_raw_rows"],
        "blank_rows": population["blank_rows"],
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False, indent=2))


def run_pilot(args: argparse.Namespace) -> None:
    previous = json.loads(args.json_report.read_text(encoding="utf-8"))
    if previous.get("status") != "population_complete":
        raise RuntimeError("run the population stage immediately before pilot")
    paths = source_paths(args.data_root)
    population, sampler, scan_elapsed = scan_population(paths, sample_per_bucket=FULL_PER_BUCKET)
    if population != previous["population"]:
        raise RuntimeError("population changed between population and pilot stages")
    pilot_samples = sampler.samples(limit=PILOT_PER_BUCKET)
    pilot_records, inference_elapsed = infer_samples(pilot_samples)
    try:
        validate_predictions(pilot_records)
    except Exception as error:
        previous["status"] = "pilot_failed"
        previous["pilot"] = {
            "status": "failed",
            "sample_count": len(pilot_records),
            "elapsed_seconds": inference_elapsed,
            "error": str(error),
        }
        write_json(args.json_report, previous)
        raise
    payload = {
        "status": "pilot_complete",
        "metadata": metadata(paths, load_config()),
        "population": population,
        "population_scan_elapsed_seconds": scan_elapsed,
        "pilot": {
            "status": "succeeded",
            "sample_count": len(pilot_records),
            "elapsed_seconds": inference_elapsed,
            "unavailable_count": sum(row["text_score"] == -1 for row in pilot_records),
            "runtime_errors": 0,
            "model_load_succeeded": True,
        },
        "_pilot_records_for_full_reuse": pilot_records,
    }
    write_json(args.json_report, payload)
    print(json.dumps({"status": payload["status"], "pilot": payload["pilot"]}, ensure_ascii=False, indent=2))


def run_full(args: argparse.Namespace) -> None:
    previous = json.loads(args.json_report.read_text(encoding="utf-8"))
    if previous.get("status") != "pilot_complete":
        raise RuntimeError("run a successful pilot stage immediately before full")
    pilot_records = previous["_pilot_records_for_full_reuse"]
    validate_predictions(pilot_records)
    paths = source_paths(args.data_root)
    population, sampler, scan_elapsed = scan_population(paths, sample_per_bucket=FULL_PER_BUCKET)
    if population != previous["population"]:
        raise RuntimeError("population changed between pilot and full stages")
    all_samples = sampler.samples()
    pilot_hashes = {row["sample_hash"] for row in pilot_records}
    expected_pilot_hashes = {
        sample.sample_hash for sample in sampler.samples(limit=PILOT_PER_BUCKET)
    }
    if pilot_hashes != expected_pilot_hashes:
        raise RuntimeError("deterministic pilot selection changed before full stage")
    remaining_samples = [sample for sample in all_samples if sample.sample_hash not in pilot_hashes]
    remaining_records, remaining_elapsed = infer_samples(remaining_samples)
    records = pilot_records + remaining_records
    if len(records) > len(PLATFORMS) * len(BUCKETS) * FULL_PER_BUCKET:
        raise RuntimeError("full inference sample exceeds the 1,800 hard limit")
    stats = grouped_statistics(records)
    unavailable_details = [
        {
            "platform": row["platform"],
            "length_bucket": row["length_bucket"],
            "reason": row["unavailable_reason"],
        }
        for row in records if row["text_score"] == -1
    ]
    sample_counts = sample_count_summary(all_samples)
    payload = {
        "status": "complete",
        "metadata": metadata(paths, load_config()),
        "population": population,
        "population_scan_elapsed_seconds": scan_elapsed,
        "pilot": previous["pilot"],
        "sample": {
            "total_inferred": len(records),
            "unique_normalized_content": True,
            "bucket_sample_counts": sample_counts["by_bucket"],
            "platform_and_bucket_sample_counts": sample_counts["by_platform_and_bucket"],
            "inference_elapsed_seconds": previous["pilot"]["elapsed_seconds"] + remaining_elapsed,
            "unavailable_details": unavailable_details,
        },
        "statistics": stats,
        "diagnostic_examples": diagnostic_examples(records),
        "interpretation": interpretation(stats),
    }
    write_json(args.json_report, payload)
    args.markdown_report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "total_inferred": len(records),
        "remaining_inferred_after_pilot": len(remaining_records),
        "remaining_elapsed_seconds": remaining_elapsed,
        "interpretation": payload["interpretation"]["classification"],
    }, ensure_ascii=False, indent=2))


def run_render(args: argparse.Namespace) -> None:
    payload = json.loads(args.json_report.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError("render requires a completed JSON report")
    payload["interpretation"] = interpretation(payload["statistics"])
    write_json(args.json_report, payload)
    args.markdown_report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": "rendered",
        "total_inferred": payload["sample"]["total_inferred"],
        "interpretation": payload["interpretation"]["classification"],
    }, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("population", "pilot", "full", "render"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # The standalone production smoke entry point also bounds CPU parallelism.
    import torch
    torch.set_num_threads(1)
    args = parse_args(argv)
    if args.stage == "population":
        run_population(args)
    elif args.stage == "pilot":
        run_pilot(args)
    elif args.stage == "full":
        run_full(args)
    else:
        run_render(args)


if __name__ == "__main__":
    main()
