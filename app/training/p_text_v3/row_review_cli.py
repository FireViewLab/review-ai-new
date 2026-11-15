"""Interactive single-reviewer CLI for AI-assisted validation-extension rows.

This tool displays private review text intentionally to the local reviewer. It
never auto-confirms a row, never fabricates per-row review timestamps, and its
outputs remain non-official and ineligible for training or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .data_contract import file_sha256
from .single_reviewer_experiment import RATIONALE_CODES, SIGNAL_TYPES
from .validation_extension_confirmation import STAGING_FIELDS


REVIEW_MODE = "SINGLE_REVIEWER_AI_ASSISTED"
try:
    SEOUL = ZoneInfo("Asia/Seoul")
except ZoneInfoNotFoundError:
    # Windows installations may lack the optional tzdata package. Korea has
    # observed UTC+09:00 without daylight-saving changes since 1988.
    SEOUL = timezone(timedelta(hours=9), name="Asia/Seoul")
CONFIRMED_FIELDS = (
    "annotation_id", "reviewer_id", "final_text_status", "final_signal_decision",
    "final_signal_types", "final_other_review_needed", "final_rationale_codes",
    "final_rationale_note", "reviewed_at", "review_mode", "confirmation_action",
    "proposal_origin", "official_human_ground_truth", "human_independent_review_verified",
    "eligible_for_training", "eligible_for_threshold_selection",
)
DEFERRED_FIELDS = ("annotation_id", "reviewer_id", "deferred_at", "reason")
BLIND_FIELDS = ("annotation_id", "display_order", "content")
ALLOWED_COMBINATIONS = {
    ("SUFFICIENT", "PRESENT"),
    ("SUFFICIENT", "ABSENT"),
    ("SUFFICIENT", "UNCERTAIN"),
    ("INSUFFICIENT", "NOT_ASSESSED"),
    ("INVALID", "NOT_ASSESSED"),
}


class RowReviewError(ValueError):
    """Stable error that never embeds review content or annotation IDs."""


def seoul_now() -> datetime:
    return datetime.now(SEOUL)


def _timestamp(now_fn: Callable[[], datetime]) -> str:
    value = now_fn()
    if value.tzinfo is None or value.utcoffset() is None:
        raise RowReviewError("REVIEW_TIMEZONE_REQUIRED")
    return value.astimezone(SEOUL).isoformat(timespec="seconds")


def _read_csv(path: str | Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise RowReviewError("INPUT_SCHEMA_INVALID")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RowReviewError("INPUT_READ_FAILED") from exc


def _unique(rows: Iterable[Mapping[str, str]], prefix: str) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        annotation_id = str(row.get("annotation_id", "")).strip()
        if not annotation_id:
            raise RowReviewError(prefix + "_ANNOTATION_ID_MISSING")
        if annotation_id in result:
            raise RowReviewError(prefix + "_ANNOTATION_ID_DUPLICATE")
        result[annotation_id] = row
    return result


def _pipe_values(value: object, allowed: frozenset[str], prefix: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value or "").split("|") if item.strip())
    if len(values) != len(set(values)):
        raise RowReviewError(prefix + "_DUPLICATE")
    if set(values).difference(allowed):
        raise RowReviewError(prefix + "_UNKNOWN")
    return values


def validate_label(
    *,
    text_status: str,
    signal_decision: str,
    signal_types: str,
    other_review_needed: str,
    rationale_codes: str,
    rationale_note: str,
) -> None:
    signals = _pipe_values(signal_types, SIGNAL_TYPES, "SIGNAL_TYPES")
    _pipe_values(rationale_codes, RATIONALE_CODES, "RATIONALE_CODES")
    if (text_status, signal_decision) not in ALLOWED_COMBINATIONS:
        raise RowReviewError("LABEL_COMBINATION_INVALID")
    if signal_decision == "PRESENT" and not signals:
        raise RowReviewError("PRESENT_SIGNAL_TYPES_REQUIRED")
    if signal_decision in {"ABSENT", "NOT_ASSESSED"} and signals:
        raise RowReviewError("SIGNAL_TYPES_MUST_BE_EMPTY")
    if other_review_needed not in {"YES", "NO"}:
        raise RowReviewError("OTHER_REVIEW_NEEDED_INVALID")
    if not rationale_note.strip():
        raise RowReviewError("RATIONALE_NOTE_REQUIRED")


def load_sources(
    packet_path: str | Path, staging_path: str | Path, *, expected_count: int = 194
) -> tuple[list[Mapping[str, str]], dict[str, Mapping[str, str]]]:
    if type(expected_count) is not int or expected_count < 1:
        raise RowReviewError("EXPECTED_COUNT_INVALID")
    packet_rows = _read_csv(packet_path, BLIND_FIELDS)
    staging_rows = _read_csv(staging_path, STAGING_FIELDS)
    packets = _unique(packet_rows, "PACKET")
    stagings = _unique(staging_rows, "STAGING")
    if len(packets) != expected_count or len(stagings) != expected_count:
        raise RowReviewError("SOURCE_COUNT_INVALID")
    if set(packets) != set(stagings):
        raise RowReviewError("SOURCE_ANNOTATION_IDS_MISMATCH")
    ordered = sorted(
        packet_rows,
        key=lambda row: (int(row["display_order"]), row["annotation_id"]),
    )
    for row in staging_rows:
        validate_label(
            text_status=row["text_status"], signal_decision=row["signal_decision"],
            signal_types=row["signal_types"], other_review_needed=row["other_review_needed"],
            rationale_codes=row["rationale_codes"], rationale_note=row["rationale_note"],
        )
        if row["proposal_origin"] != "AI_ASSISTED_DRAFT":
            raise RowReviewError("PROPOSAL_ORIGIN_INVALID")
        if row["official_human_ground_truth"] != "NO":
            raise RowReviewError("SOURCE_OFFICIAL_STATUS_INVALID")
    return ordered, stagings


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def _private_root() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts" / "p_text_v3_validation_extensions"


def _validate_output(output_dir: Path, private_root: Path) -> None:
    root = private_root.resolve()
    output = output_dir.resolve()
    if output == root or root not in output.parents:
        raise RowReviewError("OUTPUT_NOT_IN_PRIVATE_EXTENSION_ROOT")


def _load_progress(path: Path, fields: Sequence[str], prefix: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = _read_csv(path, fields)
    _unique(rows, prefix)
    return rows


def _validate_confirmed_row(row: Mapping[str, str]) -> None:
    validate_label(
        text_status=row["final_text_status"], signal_decision=row["final_signal_decision"],
        signal_types=row["final_signal_types"], other_review_needed=row["final_other_review_needed"],
        rationale_codes=row["final_rationale_codes"], rationale_note=row["final_rationale_note"],
    )
    if row["review_mode"] != REVIEW_MODE or row["confirmation_action"] not in {"AGREE", "MODIFY"}:
        raise RowReviewError("CONFIRMED_REVIEW_MODE_INVALID")
    if any(row[field] != "NO" for field in (
        "official_human_ground_truth", "human_independent_review_verified",
        "eligible_for_training", "eligible_for_threshold_selection",
    )):
        raise RowReviewError("CONFIRMED_POLICY_FLAG_INVALID")
    try:
        reviewed_at = datetime.fromisoformat(row["reviewed_at"])
    except ValueError as exc:
        raise RowReviewError("CONFIRMED_TIMESTAMP_INVALID") from exc
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise RowReviewError("CONFIRMED_TIMESTAMP_TIMEZONE_REQUIRED")
    if reviewed_at.utcoffset() != timedelta(hours=9):
        raise RowReviewError("CONFIRMED_TIMESTAMP_NOT_ASIA_SEOUL")


def _manifest(
    *, reviewer_id: str, packet_path: str | Path, staging_path: str | Path,
    expected_count: int, confirmed_count: int, deferred_count: int, updated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ptext-v3-single-reviewer-row-review-v1",
        "review_mode": REVIEW_MODE,
        "reviewer_id": reviewer_id,
        "source_sha256": {
            "blind_review_packet": file_sha256(packet_path),
            "ai_assisted_staging": file_sha256(staging_path),
        },
        "expected_count": expected_count,
        "confirmed_count": confirmed_count,
        "deferred_count": deferred_count,
        "remaining_count": expected_count - confirmed_count,
        "review_status": "COMPLETE" if confirmed_count == expected_count else "IN_PROGRESS",
        "updated_at": updated_at,
        "official_human_ground_truth": False,
        "human_independent_review_verified": False,
        "eligible_for_training": False,
        "eligible_for_threshold_selection": False,
        "dataset_publication_ready": False,
        "source_status": "AI_ASSISTED_DRAFT_REVIEWED_BY_ONE_PERSON",
    }


def _show_row(
    output_fn: Callable[[str], None], packet: Mapping[str, str], proposal: Mapping[str, str],
    position: int, expected_count: int,
) -> None:
    output_fn("\n" + "=" * 72)
    output_fn(f"[{position}/{expected_count}] review content")
    output_fn(packet["content"])
    output_fn("-" * 72)
    output_fn(f"AI text_status: {proposal['text_status']}")
    output_fn(f"AI signal_decision: {proposal['signal_decision']}")
    output_fn(f"AI signal_types: {proposal['signal_types'] or '(empty)'}")
    output_fn(f"AI other_review_needed: {proposal['other_review_needed']}")
    output_fn(f"AI rationale_codes: {proposal['rationale_codes'] or '(empty)'}")
    output_fn(f"AI rationale_note: {proposal['rationale_note']}")


def _prompt_choice(input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> str:
    aliases = {
        "a": "AGREE", "agree": "AGREE", "동의": "AGREE",
        "m": "MODIFY", "modify": "MODIFY", "수정": "MODIFY",
        "h": "HOLD", "hold": "HOLD", "보류": "HOLD",
        "q": "QUIT", "quit": "QUIT", "종료": "QUIT",
    }
    while True:
        value = input_fn("선택 [동의(a) / 수정(m) / 보류(h) / 종료(q)]: ").strip().casefold()
        if value in aliases:
            return aliases[value]
        output_fn("잘못된 선택입니다.")


def _prompt_modified(
    input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> dict[str, str]:
    while True:
        values = {
            "final_text_status": input_fn("text_status [SUFFICIENT/INSUFFICIENT/INVALID]: ").strip().upper(),
            "final_signal_decision": input_fn("signal_decision [PRESENT/ABSENT/UNCERTAIN/NOT_ASSESSED]: ").strip().upper(),
            "final_signal_types": input_fn("signal_types (| 구분, 없으면 빈 값): ").strip().upper(),
            "final_other_review_needed": input_fn("other_review_needed [YES/NO]: ").strip().upper(),
            "final_rationale_codes": input_fn("rationale_codes (| 구분, 없으면 빈 값): ").strip().upper(),
            "final_rationale_note": input_fn("사람 검수 근거(필수): ").strip(),
        }
        try:
            validate_label(
                text_status=values["final_text_status"],
                signal_decision=values["final_signal_decision"],
                signal_types=values["final_signal_types"],
                other_review_needed=values["final_other_review_needed"],
                rationale_codes=values["final_rationale_codes"],
                rationale_note=values["final_rationale_note"],
            )
            return values
        except RowReviewError as exc:
            output_fn(f"입력 오류: {exc}")


def run_review(
    *,
    packet_path: str | Path,
    staging_path: str | Path,
    output_dir: str | Path,
    reviewer_id: str,
    expected_count: int = 194,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    now_fn: Callable[[], datetime] = seoul_now,
    private_root: str | Path | None = None,
) -> dict[str, Any]:
    reviewer_id = reviewer_id.strip()
    if not reviewer_id or any(character.isspace() for character in reviewer_id):
        raise RowReviewError("REVIEWER_ID_INVALID")
    output = Path(output_dir)
    _validate_output(output, Path(private_root) if private_root else _private_root())
    packet_rows, staging_by_id = load_sources(packet_path, staging_path, expected_count=expected_count)
    output.mkdir(parents=True, exist_ok=True)
    confirmed_path = output / "confirmed_row_reviews.csv"
    deferred_path = output / "deferred_rows.csv"
    manifest_path = output / "review_session_manifest.json"
    allowed_files = {confirmed_path.name, deferred_path.name, manifest_path.name}
    unexpected = [path.name for path in output.iterdir() if path.name not in allowed_files]
    if unexpected:
        raise RowReviewError("OUTPUT_DIRECTORY_NOT_OWNED_BY_REVIEW_TOOL")
    confirmed_rows = _load_progress(confirmed_path, CONFIRMED_FIELDS, "CONFIRMED")
    deferred_rows = _load_progress(deferred_path, DEFERRED_FIELDS, "DEFERRED")
    confirmed_by_id = {row["annotation_id"]: row for row in confirmed_rows}
    deferred_by_id = {row["annotation_id"]: row for row in deferred_rows}
    source_ids = {row["annotation_id"] for row in packet_rows}
    if set(confirmed_by_id).difference(source_ids) or set(deferred_by_id).difference(source_ids):
        raise RowReviewError("PROGRESS_SOURCE_IDS_MISMATCH")
    if set(confirmed_by_id).intersection(deferred_by_id):
        raise RowReviewError("PROGRESS_STATUS_CONFLICT")
    for row in confirmed_rows:
        _validate_confirmed_row(row)
        if row["reviewer_id"] != reviewer_id:
            raise RowReviewError("PROGRESS_REVIEWER_MISMATCH")
    if any(row["reviewer_id"] != reviewer_id for row in deferred_rows):
        raise RowReviewError("PROGRESS_REVIEWER_MISMATCH")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "ptext-v3-single-reviewer-row-review-v1" or manifest.get("review_mode") != REVIEW_MODE:
            raise RowReviewError("SESSION_SCHEMA_INVALID")
        if manifest.get("reviewer_id") != reviewer_id:
            raise RowReviewError("SESSION_REVIEWER_MISMATCH")
        if manifest.get("source_sha256") != {
            "blind_review_packet": file_sha256(packet_path),
            "ai_assisted_staging": file_sha256(staging_path),
        }:
            raise RowReviewError("SESSION_SOURCE_SHA256_MISMATCH")
        if manifest.get("expected_count") != expected_count:
            raise RowReviewError("SESSION_EXPECTED_COUNT_MISMATCH")
        if manifest.get("confirmed_count") != len(confirmed_rows) or manifest.get("deferred_count") != len(deferred_rows):
            raise RowReviewError("SESSION_PROGRESS_COUNT_MISMATCH")

    seen_this_run: set[str] = set()
    stopped = False
    for position, packet in enumerate(packet_rows, 1):
        annotation_id = packet["annotation_id"]
        if annotation_id in confirmed_by_id or annotation_id in seen_this_run:
            continue
        proposal = staging_by_id[annotation_id]
        _show_row(output_fn, packet, proposal, position, expected_count)
        choice = _prompt_choice(input_fn, output_fn)
        if choice == "QUIT":
            stopped = True
            break
        seen_this_run.add(annotation_id)
        if choice == "HOLD":
            deferred_by_id[annotation_id] = {
                "annotation_id": annotation_id,
                "reviewer_id": reviewer_id,
                "deferred_at": _timestamp(now_fn),
                "reason": "USER_DEFERRED",
            }
        else:
            if choice == "AGREE":
                values = {
                    "final_text_status": proposal["text_status"],
                    "final_signal_decision": proposal["signal_decision"],
                    "final_signal_types": proposal["signal_types"],
                    "final_other_review_needed": proposal["other_review_needed"],
                    "final_rationale_codes": proposal["rationale_codes"],
                    "final_rationale_note": proposal["rationale_note"],
                }
            else:
                values = _prompt_modified(input_fn, output_fn)
            record = {
                "annotation_id": annotation_id,
                "reviewer_id": reviewer_id,
                **values,
                "reviewed_at": _timestamp(now_fn),
                "review_mode": REVIEW_MODE,
                "confirmation_action": choice,
                "proposal_origin": proposal["proposal_origin"],
                "official_human_ground_truth": "NO",
                "human_independent_review_verified": "NO",
                "eligible_for_training": "NO",
                "eligible_for_threshold_selection": "NO",
            }
            _validate_confirmed_row(record)
            confirmed_by_id[annotation_id] = record
            deferred_by_id.pop(annotation_id, None)
        confirmed_rows = [confirmed_by_id[key] for key in sorted(confirmed_by_id)]
        deferred_rows = [deferred_by_id[key] for key in sorted(deferred_by_id)]
        _atomic_csv(confirmed_path, CONFIRMED_FIELDS, confirmed_rows)
        _atomic_csv(deferred_path, DEFERRED_FIELDS, deferred_rows)
        updated_at = _timestamp(now_fn)
        _atomic_json(
            manifest_path,
            _manifest(
                reviewer_id=reviewer_id, packet_path=packet_path, staging_path=staging_path,
                expected_count=expected_count, confirmed_count=len(confirmed_rows),
                deferred_count=len(deferred_rows), updated_at=updated_at,
            ),
        )

    if not manifest_path.exists():
        _atomic_csv(confirmed_path, CONFIRMED_FIELDS, confirmed_rows)
        _atomic_csv(deferred_path, DEFERRED_FIELDS, deferred_rows)
        updated_at = _timestamp(now_fn)
        _atomic_json(
            manifest_path,
            _manifest(
                reviewer_id=reviewer_id, packet_path=packet_path, staging_path=staging_path,
                expected_count=expected_count, confirmed_count=len(confirmed_rows),
                deferred_count=len(deferred_rows), updated_at=updated_at,
            ),
        )
    complete = len(confirmed_by_id) == expected_count
    result = {
        "status": "STOPPED_BY_USER" if stopped else "REVIEW_COMPLETE" if complete else "PASS_FINISHED_WITH_PENDING",
        "confirmed_count": len(confirmed_by_id),
        "deferred_count": len(deferred_by_id),
        "remaining_count": expected_count - len(confirmed_by_id),
        "complete": complete,
        "official_human_ground_truth": False,
        "human_independent_review_verified": False,
        "eligible_for_training": False,
        "eligible_for_threshold_selection": False,
        "output_dir": str(output.resolve()),
    }
    output_fn(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--expected-count", type=int, default=194)
    args = parser.parse_args()
    try:
        run_review(
            packet_path=args.packet,
            staging_path=args.staging,
            output_dir=args.output_dir,
            reviewer_id=args.reviewer_id,
            expected_count=args.expected_count,
        )
    except RowReviewError as exc:
        print(json.dumps({"status": "FAILED", "failures": [str(exc)]}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
