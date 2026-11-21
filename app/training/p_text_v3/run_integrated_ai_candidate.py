"""CLI for the offline integrated AI candidate; never changes runtime config."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from app.analyzers.behavior import BehaviorInput
from app.analyzers.network import NetworkReview
from app.training.p_text_v3.integrated_ai_candidate import (
    EXPECTED_MODEL_SHA256,
    IntegratedAICandidate,
    IntegratedReviewInput,
)


def _datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("datetime values must be ISO-8601 strings or null")
    return datetime.fromisoformat(value)


def _behavior(value: Any) -> BehaviorInput | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("behavior must be an object or null")
    review_dates = value.get("user_review_dates")
    return BehaviorInput(
        review_date=_datetime(value.get("review_date")),
        user_id=value.get("user_id"),
        verified_purchase=value.get("verified_purchase"),
        account_created_at=_datetime(value.get("account_created_at")),
        user_review_dates=(
            tuple(_datetime(item) for item in review_dates)
            if review_dates is not None else None
        ),
    )


def review_from_json(value: dict[str, Any]) -> IntegratedReviewInput:
    return IntegratedReviewInput(
        platform=value["platform"],
        product_id=value["product_id"],
        review_id=value["review_id"],
        content=value["content"],
        rating=value.get("rating"),
        rating_scale_max=value.get("rating_scale_max"),
        rating_source=value.get("rating_source"),
        rating_source_verified=value.get("rating_source_verified", False),
        image_count=value.get("image_count"),
        image_count_source=value.get("image_count_source"),
        image_count_source_verified=value.get("image_count_source_verified", False),
        behavior=_behavior(value.get("behavior")),
    )


def synthetic_payload() -> dict[str, Any]:
    return {
        "reviews": [
            {
                "platform": "synthetic", "product_id": "product-demo",
                "review_id": "synthetic-001",
                "content": "2주 동안 직접 사용했고 오후에는 조금 건조했지만 발림성과 지속력은 좋았습니다.",
                "rating": 4, "rating_scale_max": 5,
                "rating_source": "synthetic_explicit", "rating_source_verified": True,
                "image_count": 1, "image_count_source": "synthetic_explicit",
                "image_count_source_verified": True,
            },
            {
                "platform": "synthetic", "product_id": "product-demo",
                "review_id": "synthetic-002", "content": "아주 좋아요",
                "rating": 5, "rating_scale_max": 5,
                "rating_source": "synthetic_explicit", "rating_source_verified": True,
                "image_count": 0, "image_count_source": "synthetic_explicit",
                "image_count_source_verified": True,
            },
            {
                "platform": "synthetic", "product_id": "product-demo",
                "review_id": "synthetic-003",
                "content": "혁신적인 기술력과 압도적인 성능으로 누구에게나 반드시 추천하며 완벽한 효과를 보장합니다.",
                "rating": None, "image_count": None,
            },
        ]
    }


def run_payload(analyzer: IntegratedAICandidate, payload: dict[str, Any]) -> dict[str, Any]:
    if "reviews" in payload:
        reviews = tuple(review_from_json(item) for item in payload["reviews"])
        results = analyzer.analyze_product_batch(reviews)
        return {"mode": "product_batch", "count": len(results),
                "results": [result.to_dict() for result in results]}
    review = review_from_json(payload["review"] if "review" in payload else payload)
    comparison_reviews = tuple(
        NetworkReview(item["review_id"], item["product_id"], item["content"])
        for item in payload.get("comparison_reviews", ())
    )
    result = analyzer.analyze_one(review, comparison_reviews=comparison_reviews)
    return {"mode": "single", "count": 1, "results": [result.to_dict()]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Re:view integrated AI candidate")
    parser.add_argument("--model-path", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--expected-model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    payload = (
        synthetic_payload() if args.synthetic else
        json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    )
    analyzer = IntegratedAICandidate(
        args.model_path,
        expected_model_sha256=args.expected_model_sha256,
        device="cpu",
    )
    output = run_payload(analyzer, payload)
    encoded = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(encoded, end="")
    else:
        if args.output_json.exists():
            raise FileExistsError("output JSON already exists")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
        print(json.dumps({"status": "OK", "count": output["count"],
                          "output": str(args.output_json.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
