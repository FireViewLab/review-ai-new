"""Build the isolated 30-pair synthetic expansion v1 annotation set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.p_text.contrastive_pilot import EXPORT_COLUMNS


OUTPUT_DIR = Path("data/ptext_improvement/synthetic_expansion_v1")
JSONL_PATH = OUTPUT_DIR / "annotation_synthetic_expansion_v1.jsonl"
CSV_PATH = OUTPUT_DIR / "annotation_synthetic_expansion_v1.csv"
CREATED_AT = "2026-10-03"
KEYWORD_BALANCE_REVISED_PAIRS = frozenset({1, 2, 4, 7, 10, 15, 26, 27, 28, 29, 30})
HUMAN_REVIEW_ROUND1_APPROVED_PAIRS = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
)
HUMAN_REVIEW_ROUND1_REVISED_PAIRS = frozenset({10, 12, 13, 24, 25, 26, 27, 28, 29, 30})
HUMAN_REVIEW_ROUND2_APPROVED_PAIRS = frozenset({10, 12, 13, 24, 25, 29, 30})
HUMAN_REVIEW_ROUND2_REVISED_PAIRS = frozenset({26, 27, 28})
HUMAN_REVIEW_ROUND3_APPROVED_PAIRS = frozenset({26, 27, 28})
HUMAN_REVIEW_ROUND4_APPROVED_PAIRS = frozenset({10, 26, 27, 29})
HUMAN_APPROVED_PAIRS = (
    HUMAN_REVIEW_ROUND1_APPROVED_PAIRS
    | HUMAN_REVIEW_ROUND2_APPROVED_PAIRS
    | HUMAN_REVIEW_ROUND3_APPROVED_PAIRS
    | HUMAN_REVIEW_ROUND4_APPROVED_PAIRS
)

# category, domain, difficulty, length, style, relation, suspicious, normal,
# suspicious praise/evidence, normal praise/evidence
PAIRS = (
    ("EXPERIENCE_FREE_CLAIM", "health_supplement", "easy", "medium", "polite", "unopened_vitamin_claim_vs_bounded_use",
     "비타민을 아직 뜯지도 않았지만 성분표가 좋아서 먹으면 아침 피로가 완전히 사라질 것 같습니다.",
     "비타민 성분을 확인한 뒤 석 달 복용해 보니 제 경우 아침 피로가 덜한 날이 있었지만 다른 생활 습관의 영향은 구분하기 어렵습니다.", "medium", "none", "medium", "sufficient"),
    ("EXPERIENCE_FREE_CLAIM", "pharma_otc", "medium", "medium", "conversational", "package_only_patch_claim_vs_observed_relief",
     "붙이는 제품은 포장 설명만 읽어 봤는데 이 정도 성분이면 뻐근함이 하루 종일 확실히 없어지겠네요.",
     "붙이는 제품의 성분과 사용법을 확인하고 저녁마다 사흘 써 보니 제 뻐근함은 잠깐 덜했어요. 하루 종일 유지되는지는 모르겠어요.", "medium", "none", "low", "limited"),
    ("EXPERIENCE_FREE_CLAIM", "electronics", "hard", "long", "polite", "spec_sheet_battery_claim_vs_measured_session",
     "제품은 배송 전이고 사양표의 배터리 용량만 확인했습니다. 이 수치라면 어떤 작업을 해도 충전 없이 종일 안정적으로 쓸 수 있겠습니다.",
     "제품을 문서 작업 위주로 닷새 사용했을 때는 저녁까지 배터리가 남았습니다. 영상 편집처럼 부하가 큰 작업 시간은 확인하지 못했습니다.", "medium", "none", "medium", "sufficient"),
    ("EXPERIENCE_FREE_CLAIM", "cosmetics", "medium", "medium", "polite", "ingredient_list_skin_claim_vs_two_week_observation",
     "아직 얼굴에 바르기 전이지만 진정 성분 구성을 보니 붉은 기가 금방 가라앉고 피부 장벽도 확실히 좋아질 것 같습니다.",
     "이 크림의 진정 성분을 확인하고 이 주 동안 밤에 발랐더니 제 피부의 당김은 줄었습니다. 붉은 기나 장벽 변화까지는 판단하기 어렵습니다.", "medium", "none", "medium", "sufficient"),
    ("EXPERIENCE_FREE_CLAIM", "food", "hard", "long", "conversational", "menu_description_taste_claim_vs_personal_tasting",
     "아직 봉지를 열진 않았고 원재료와 조리 설명만 봤어요. 먹어 보면 잡내 없이 깊은 맛이 나서 가족 모두 입맛에 꼭 맞을 거예요.",
     "직접 조리해 먹어 보니 제 입에는 잡내가 적고 맛이 진했어요. 아이는 향이 세다고 해서 가족 모두에게 맞는다고 하긴 어렵네요.", "medium", "none", "medium", "sufficient"),

    ("UNUSED_RECOMMENDATION", "health_supplement", "easy", "short", "concise", "unconsumed_probiotic_recommendation_vs_personal_trial",
     "유산균은 아직 먹기 전이지만 장이 불편한 분이라면 누구든 이 제품부터 드셔 보세요.",
     "유산균을 한 달 먹고 제 속은 전보다 편해서 비슷한 고민이 있는 분께 참고 정도로 권하고 싶어요.", "medium", "none", "medium", "sufficient"),
    ("UNUSED_RECOMMENDATION", "fashion", "medium", "medium", "conversational", "unworn_shoe_recommendation_vs_fit_limited_advice",
     "신발을 신어 보지는 않았는데 쿠션 모양만 봐도 발 아픈 사람에게 전부 편할 것 같아 꼭 추천해요.",
     "이 신발로 출퇴근을 일주일 해 보니 제 발볼에는 편해서 비슷한 발볼인 분께는 추천하고 싶어요. 다만 발 모양이 다르면 착화감도 달라질 수 있어요.", "medium", "none", "medium", "sufficient"),
    ("UNUSED_RECOMMENDATION", "pharma_otc", "hard", "medium", "polite", "unopened_spray_family_recommendation_vs_limited_use",
     "분사형 외용 제품은 아직 개봉하지 않았지만 설명에 적힌 용도를 보니 가족에게도 고민 없이 권해도 되겠습니다.",
     "분사형 외용 제품을 저는 이틀 써 보니 사용이 간편했습니다. 가족에게 맞는지는 각자 확인해야 할 것 같습니다.", "medium", "none", "low", "limited"),
    ("UNUSED_RECOMMENDATION", "household", "easy", "medium", "polite", "unassembled_shelf_recommendation_vs_loaded_use",
     "선반은 조립 전이지만 구조가 단단해 보여서 수납이 필요한 집이라면 무조건 들여놓아도 좋겠습니다.",
     "선반을 조립해 일주일간 가벼운 주방용품을 올려 두니 흔들림은 적었습니다. 무거운 물건에는 권하기 어렵습니다.", "medium", "none", "medium", "sufficient"),
    ("UNUSED_RECOMMENDATION", "health_supplement", "medium", "long", "polite", "label_only_omega_recommendation_vs_personal_course",
     "오메가3는 복용 전이고 함량 표시만 살펴봤습니다. 이 구성이라면 효과도 괜찮을 것 같아 건강을 챙기려는 분께 가장 먼저 추천할 만한 선택입니다.",
     "건강 관리를 위해 오메가3를 두 달 복용했고 제게 특별한 불편은 없었습니다. 체감 효과는 분명하지 않아 같은 복용 조건인 분께만 참고용으로 추천합니다.", "medium", "none", "low", "sufficient"),

    ("NATURAL_PROMOTIONAL", "cosmetics", "easy", "medium", "conversational", "sample_touch_to_all_day_promotion_vs_bounded_wear",
     "손등에 한 번 펴 본 정도인데 발림이 꽤 좋네요. 하루 종일 촉촉한 피부를 원한다면 망설이지 말고 골라도 될 것 같아요.",
     "얼굴에 열흘 써 보니 오전에는 촉촉했지만 오후엔 당김이 있었어요. 건성 피부라면 보습제를 덧바를 생각으로 골라야 해요.", "medium", "limited", "low", "sufficient"),
    ("NATURAL_PROMOTIONAL", "food", "medium", "medium", "polite", "aroma_only_purchase_nudge_vs_tasted_preference",
     "아직 먹어 보진 않고 포장을 열어 향만 맡았어요. 향이 괜찮아서 맛과 재료 품질도 분명 좋고 누구나 만족할 간식이니 바로 골라도 후회 없겠어요.",
     "직접 한 봉지를 먹어 보니 고소한 향은 좋았지만 제 입에는 단맛이 강했습니다. 덜 단 간식을 찾는 분은 참고하세요.", "medium", "limited", "medium", "sufficient"),
    ("NATURAL_PROMOTIONAL", "health_supplement", "hard", "long", "polite", "single_dose_routine_promotion_vs_short_observation",
     "홍삼을 오늘 처음 한 번 먹어 봤는데 몸이 조금 가벼운 느낌이었어요. 이 정도면 계속 먹는 동안 피로 걱정이 없을 것 같으니 바쁜 분들께도 꼭 권하고 싶어요.",
     "홍삼을 일주일 아침마다 먹었고 따뜻한 맛은 마음에 들었습니다. 활력 변화는 수면 상태와 겹쳐 아직 분명하게 말하기 어렵습니다.", "medium", "limited", "medium", "limited"),
    ("NATURAL_PROMOTIONAL", "electronics", "medium", "short", "concise", "display_only_productivity_pitch_vs_actual_setup",
     "전원만 켜 봤는데 화면이 선명하네요. 업무 효율을 확실히 높이고 싶다면 이 모니터가 답입니다.",
     "모니터를 사흘 문서 작업에 써 보니 글자는 선명했어요. 색 작업 정확도는 아직 확인하지 못했습니다.", "extreme", "limited", "medium", "limited"),
    ("NATURAL_PROMOTIONAL", "pharma_otc", "hard", "long", "conversational", "sealed_product_relief_pitch_vs_limited_application",
     "아직 밀봉 상태지만 사용법과 성분을 보니 불편한 부위를 빠르게 편하게 해 줄 구성이네요. 비슷한 고민이 있다면 미루지 말고 준비해 두세요.",
     "외용 제품의 성분과 사용법을 확인하고 작은 부위에 이틀 써 보니 바르기는 편했어요. 불편함이 줄어드는지와 넓은 부위 반응은 더 지켜보려고요.", "medium", "none", "low", "limited"),

    ("PROMOTION_EXPERIENCE_MIX", "health_supplement", "easy", "medium", "conversational", "two_day_mineral_claim_vs_bounded_two_day_note",
     "미네랄을 이틀 챙겼는데 아침부터 컨디션이 달라졌어요. 계속 먹으면 피로 걱정 없이 지낼 수 있으니 꼭 시작해 보세요.",
     "미네랄을 이틀 챙겼고 속 불편함은 없었어요. 피로나 컨디션 변화는 기간이 짧아 아직 모르겠어요.", "extreme", "limited", "low", "limited"),
    ("PROMOTION_EXPERIENCE_MIX", "fashion", "medium", "long", "polite", "brief_coat_tryon_to_season_claim_vs_observed_fit",
     "코트를 실내에서 잠깐 입어 보니 가볍고 움직이기 편했습니다. 이 정도면 추운 날에도 보온과 활동성을 모두 책임질 옷이라 누구에게나 권하고 싶습니다.",
     "코트를 실내에서 잠깐 입어 보니 제 어깨에는 잘 맞고 움직임도 편했습니다. 추운 날의 보온성과 장시간 착용감은 확인하지 못했습니다.", "extreme", "limited", "medium", "limited"),
    ("PROMOTION_EXPERIENCE_MIX", "household", "easy", "short", "concise", "single_wipe_durability_claim_vs_single_wipe_scope",
     "한 번 닦아 봤는데 얼룩이 잘 지워져요. 오래 써도 성능이 그대로일 완벽한 청소 도구네요.",
     "한 번 닦아 보니 가벼운 얼룩은 지워졌어요. 오래 쓸 때 내구성은 아직 모르겠어요.", "extreme", "limited", "low", "limited"),
    ("PROMOTION_EXPERIENCE_MIX", "cosmetics", "hard", "medium", "polite", "one_night_serum_universal_claim_vs_one_night_limit",
     "세럼을 어젯밤 처음 발랐는데 아침 피부가 매끈했습니다. 꾸준히 쓰면 모든 피부의 탄력 고민을 해결할 제품이라고 확신합니다.",
     "세럼을 어젯밤 처음 발랐고 아침에 제 피부가 덜 거칠게 느껴졌습니다. 탄력 변화나 다른 피부의 반응은 아직 판단할 수 없습니다.", "extreme", "limited", "medium", "limited"),
    ("PROMOTION_EXPERIENCE_MIX", "food", "medium", "medium", "conversational", "single_bite_family_claim_vs_taste_limit",
     "한입 먹자마자 재료가 좋다는 게 느껴졌어요. 남녀노소 매일 찾아 먹을 맛이니 쟁여 두면 틀림없어요.",
     "한입 먹어 보니 저는 담백해서 좋았어요. 향이 독특해 가족 취향에도 맞을지는 더 먹어 봐야겠어요.", "extreme", "limited", "medium", "limited"),

    ("FEATURE_AS_PERSONAL_EXPERIENCE", "health_supplement", "hard", "long", "polite", "nutrition_panel_as_personal_energy_vs_logged_course",
     "영양제의 비타민 함량과 배합만 확인했고 아직 복용하지 않았습니다. 이 구성 덕분에 제 오후 피로가 줄고 집중력이 오래 유지될 것입니다.",
     "영양제를 여섯 주 복용하며 기록해 보니 제 경우 오후에 처지는 날이 조금 줄었습니다. 집중력 변화는 일정하지 않았습니다.", "medium", "none", "medium", "sufficient"),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "pharma_otc", "medium", "medium", "polite", "applicator_feature_as_relief_vs_observed_application",
     "외용 제품의 얇은 도포구와 설명만 확인했고 피부에는 쓰지 않았습니다. 그래서 불편한 부위가 빠르게 편해지고 자극도 전혀 없을 것 같습니다.",
     "외용 제품을 작은 부위에 사흘 발라 보니 도포구는 편했고 제게 눈에 띄는 자극은 없었습니다. 불편함 변화는 크지 않았습니다.", "medium", "none", "low", "sufficient"),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "household", "medium", "medium", "conversational", "storage_layout_as_lived_order_vs_actual_loading",
     "수납함의 칸 구성과 손잡이만 살펴봤고 물건은 넣지 않았어요. 이 구조라면 집안 정리가 저절로 되고 매일 찾는 시간도 확 줄겠네요.",
     "수납함에 문구류를 일주일 넣어 보니 작은 물건은 찾기 쉬웠어요. 부피 큰 물건에는 칸이 좁았어요.", "medium", "none", "low", "sufficient"),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "electronics", "hard", "long", "polite", "specification_as_recorded_performance_vs_measured_use",
     "아직 들어 보진 않았고 드라이버 사양과 소음 차단 설명만 봤어요. 그래도 어디서 쓰든 음악은 분명 선명하고 주변 소음도 완전히 막아 줄 것 같아요.",
     "이어폰을 출퇴근길에 일주일 사용해 보니 지하철 안내음은 일부 들렸고 저음은 제 취향에 맞았습니다. 조용한 실내 통화는 시험하지 못했습니다.", "medium", "none", "medium", "sufficient"),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "fashion", "easy", "short", "concise", "fabric_tag_as_all_day_comfort_vs_timed_wear",
     "아직 입어 보진 않았지만 원단표를 보니 통기성이 좋아 보여요. 하루 종일 입어도 땀이 차지 않고 편한 옷일 게 분명해요.",
     "두 시간 입어 보니 원단은 부드러웠지만 소매가 조금 조였어요. 하루 종일 착용감은 모르겠어요.", "medium", "none", "low", "limited"),

    ("EXAGGERATED_PRAISE_WITHOUT_EVIDENCE", "health_supplement", "easy", "short", "concise", "sealed_omega_extreme_claim_vs_supported_high_praise",
     "오메가3는 아직 밀봉 상태지만 성분 구성이 아주 좋아 보여요. 복용하면 효과도 확실하고 누구에게나 잘 맞을 완벽한 제품일 것 같아서 기대가 큽니다.",
     "오메가3를 넉 달 복용했고 제 생활에는 거의 완벽할 만큼 먹기 편해 만족도가 최고였어요. 체감 효과는 개인마다 다를 수 있어요.", "extreme", "none", "extreme", "sufficient"),
    ("EXAGGERATED_PRAISE_WITHOUT_EVIDENCE", "health_supplement", "medium", "medium", "conversational", "one_sip_red_ginseng_praise_vs_long_use_praise",
     "홍삼을 한 모금 마셔 보니 맛은 꽤 만족스러웠어요. 이 정도 품질이면 꾸준히 먹을수록 활력 효과가 확실하고 누구에게나 잘 맞을 것 같아요.",
     "홍삼을 두 달 아침마다 먹었고 제 입맛과 생활 리듬에는 거의 완벽하게 맞아 최고로 만족해요. 다른 사람의 활력 효과까지 장담하진 못해요.", "extreme", "limited", "extreme", "sufficient"),
    ("EXAGGERATED_PRAISE_WITHOUT_EVIDENCE", "pharma_otc", "hard", "long", "polite", "box_only_extreme_relief_claim_vs_course_based_praise",
     "상자와 사용 설명만 봤는데 구성이 꽤 만족스러워요. 불편한 부위를 완벽하게 관리해 주고 집이나 밖 어디서든 가장 뛰어난 제품일 것 같아요.",
     "설명에 따라 일주일 사용했고 제 사용 조건에는 거의 완벽할 만큼 간편해 매우 만족했습니다. 모든 상황에 같은 결과가 난다고 말할 수는 없습니다.", "extreme", "none", "extreme", "sufficient"),
    ("EXAGGERATED_PRAISE_WITHOUT_EVIDENCE", "cosmetics", "hard", "long", "polite", "single_swab_total_skin_claim_vs_sustained_observation",
     "손목에 한 번 발라 봤는데 벌써 꽤 만족스러워요. 이 정도면 보습, 탄력, 진정 효과까지 완벽하고 어떤 피부에도 최고의 변화를 줄 것 같아요.",
     "얼굴에 두 달 사용했고 제 건조함 관리에는 완벽에 가깝다고 느낄 만큼 만족스러웠습니다. 탄력과 진정 효과는 분리해 확인하지 못했습니다.", "extreme", "limited", "extreme", "sufficient"),
    ("EXAGGERATED_PRAISE_WITHOUT_EVIDENCE", "food", "medium", "medium", "conversational", "appearance_only_best_taste_vs_repeated_tasting_praise",
     "아직 먹어 보진 않았지만 사진과 포장이 마음에 들어 벌써 만족스러워요. 맛과 신선도도 완벽해서 누구나 좋아할 간식일 게 분명해요.",
     "세 번 주문해 먹었고 제 취향에는 거의 완벽할 만큼 식감과 단맛의 균형이 좋아 계속 만족했어요. 단맛을 싫어하면 다르게 느낄 수 있어요.", "extreme", "none", "extreme", "sufficient"),
)


def _row(pair_number: int, label: str, item: tuple[str, ...]) -> dict[str, object]:
    category, domain, difficulty, length_bucket, style, relation, suspicious, normal, sp, se, np, ne = item
    pair_id = f"synthetic-expansion-v1-p{pair_number:03d}"
    is_suspicious = label == "SUSPICIOUS"
    human_approved = pair_number in HUMAN_APPROVED_PAIRS
    note_parts = ["codex_candidate", "human_review_required"]
    if pair_number in KEYWORD_BALANCE_REVISED_PAIRS:
        note_parts.append("revision_reason=keyword_balance")
    if pair_number in HUMAN_REVIEW_ROUND1_APPROVED_PAIRS:
        note_parts.extend(("human_review_round=1", "human_decision=APPROVE"))
    elif pair_number in HUMAN_REVIEW_ROUND1_REVISED_PAIRS:
        note_parts.extend(
            ("human_review_round=1", "human_decision=REVISE", "revision_reason=human_review_round1")
        )
    if pair_number in HUMAN_REVIEW_ROUND2_APPROVED_PAIRS:
        note_parts.extend(("human_review_round2=2", "human_review_round2_decision=APPROVE"))
    elif pair_number in HUMAN_REVIEW_ROUND2_REVISED_PAIRS:
        note_parts.extend(
            ("human_review_round2=2", "human_review_round2_decision=REVISE",
             "revision_reason=human_review_round2")
        )
    if pair_number in HUMAN_REVIEW_ROUND3_APPROVED_PAIRS:
        note_parts.extend(("human_review_round3=3", "human_review_round3_decision=APPROVE"))
    if pair_number in HUMAN_REVIEW_ROUND4_APPROVED_PAIRS:
        note_parts.extend(
            ("human_review_round4_pending=effect_keyword_shortcut_mitigation",
             "human_review_round4=4", "human_review_round4_decision=APPROVE")
        )
    return {
        "sample_id": f"{pair_id}-{'s' if is_suspicious else 'n'}",
        "content": suspicious if is_suspicious else normal,
        "label": label,
        "category": category,
        "secondary_categories": "",
        "domain": domain,
        "source_type": "synthetic",
        "parent_pair_id": pair_id,
        "generation_family_id": f"synthetic-expansion-v1-family-{pair_number:03d}",
        "difficulty": difficulty,
        "review_status": "DRAFT",
        "notes": "; ".join(note_parts),
        "author_id": "codex",
        "reviewer_ids": "",
        "split": "",
        "seed": "",
        "created_at": CREATED_AT,
        "source_reference": "",
        "creation_method": "codex_generated_candidate",
        "original_source_type": "",
        "is_llm_generated": True,
        "pair_relation": relation,
        "length_bucket": length_bucket,
        "style": style,
        "praise_intensity": sp if is_suspicious else np,
        "evidence_level": se if is_suspicious else ne,
        "human_reviewed": human_approved,
        "human_approved": human_approved,
    }


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_number, item in enumerate(PAIRS, start=1):
        rows.append(_row(pair_number, "SUSPICIOUS", item))
        rows.append(_row(pair_number, "NORMAL", item))
    return rows


def main() -> None:
    if JSONL_PATH.exists() or CSV_PATH.exists():
        raise FileExistsError("refusing to overwrite existing expansion annotation files")
    rows = build_rows()
    with JSONL_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with CSV_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
