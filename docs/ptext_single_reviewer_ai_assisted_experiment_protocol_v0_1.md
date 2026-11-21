# P_text v3 단일 검수자 + AI 보조 실험 프로토콜 v0.1

## 1. 상태와 적용 범위

이 프로토콜은 학교 프로젝트의 제한된 인력으로 P_text v3 학습 가능성을
탐색하기 위한 비공식 실험 경로다. 검수 인원은 1명이며 AI 제안과 사용자가
확인한 draft를 활용할 수 있다. 이 결과의 상태는 항상
`SINGLE_REVIEWER_AI_ASSISTED_EXPERIMENT`이며 공식 Human Ground Truth가 아니다.

공식 경로인 `ptext_labeling_guide_v0_3.md`, v0.3 계약, 두 명의 독립 블라인드
검수, adjudication 및 `build_dataset.py`의 fail-closed gate는 그대로 유지한다.
실험 결과를 공식 Reviewer A/B 결과나 adjudicated label로 변환하거나 표시하지
않는다.

## 2. 입력 증거와 역할 구분

- AI-assisted label은 후보 판단을 돕는 제안이며 사람의 독립 검수 결과가 아니다.
- 기존 사용자 확인 상태 `CONFIRMED_AS_DRAFT`는 실험용 label 선택에만 사용한다.
  공식 Reviewer A 제출, 두 명의 일치, adjudication 완료를 뜻하지 않는다.
- staging의 `annotator_id`와 `annotated_at`은 비워 둔다. 공식 사람 검수의 신원과
  시각을 허위로 만들지 않는다.
- provenance에는 `artifact_status=AI_ASSISTED_STAGING_NOT_FORMAL_REVIEW`,
  `user_confirmation_status=CONFIRMED_AS_DRAFT`, 후보 pack, 확인 원본과 400건
  확인 사실을 명시한다.
- 2026-11-13 기존 provenance처럼 명시 상태 필드가 생기기 전 파일은 정확한
  비공식 artifact 상태, `confirmed_source`, `confirmed_rows=400`, 공식 A 및 B
  완료=false가 모두 일치할 때만 이 실험 경로에서 `CONFIRMED_AS_DRAFT`의 legacy
  증거로 인정한다. 공식 경로에서는 이 호환 규칙을 사용하지 않는다.
- `SUFFICIENT/PRESENT`만 SUSPICIOUS, `SUFFICIENT/ABSENT`만 NORMAL로 매핑한다.
  `UNCERTAIN`, `NOT_ASSESSED`, `INSUFFICIENT`, `INVALID`는 이진 학습에서 제외한다.

## 3. 재검토 우선순위

단일 검수자는 다음 사례를 우선 재검토한다.

1. `UNCERTAIN` 또는 `NOT_ASSESSED` 사례
2. `other_review_needed=YES` 사례
3. `PROPOSED_BOUNDARY_*`, `PROPOSED_LOW_INFO_NOT_ASSESSED` 근거 사례
4. 신호가 여러 개이거나 rationale code와 메모가 충돌하는 사례
5. 저정보 리뷰, 상품 설명과 실제 사용 경험의 경계, 홍보·효능 표현의 경계
6. verified near-duplicate 관계 또는 동일 상품군에서 라벨이 일관되지 않은 사례

재검토를 하더라도 AI 제안을 먼저 본 이력이 있으므로 독립 블라인드 검수라고
보고하지 않는다. 변경 사항은 새 provenance와 새 출력 경로로 남기고 원본
staging을 수정하지 않는다.

## 4. 고정 Split과 Frozen Test 보호

- 역할은 라벨 확인 전에 정한 Train 220 / Validation 80 / Frozen Test 100을
  그대로 사용하며 재분할하지 않는다.
- Train만 학습에 사용하고 Validation만 checkpoint 선택, calibration 및 threshold
  선택에 사용한다.
- Frozen Test는 모델과 정책이 동결되기 전 실행하지 않는다. 모델 선택이나
  threshold 조정에 사용하지 않는다.
- Frozen Test label을 AI 보조 단계에서 이미 확인한 이력이 있으므로 완전한
  독립 블라인드 평가라고 주장하지 않는다.
- 최종 비교 보고서에는 `prior_ai_assisted_label_exposure=true`와 label 품질 한계를
  함께 표시한다.

## 5. 품질·누수 gate

실험 validator는 400 annotation ID의 정확한 일치와 유일성, 고정 split 수,
label/signal/rationale vocabulary, 허용 조합, provenance 상태를 확인한다. 상품,
정규화 본문 해시, duplicate cluster, verified near-duplicate/v2-lineage 관계가 split
사이에 걸치면 fail-closed한다. Pilot exclusion과 일치하는 행도 거부한다.

Dataset builder는 공식 builder와 별도 진입점 및 `experiment`가 포함된 새 출력
경로만 사용한다. 기존 출력은 덮어쓰지 않는다. workbook summary에 비공식 상태,
입력 SHA-256, 제외 수, 고정 split, Frozen Test 제한을 기록한다.

## 6. 결과 보고 원칙과 한계

보고서에는 다음을 반드시 명시한다.

- 검수 인원 1명과 AI 보조 사용 사실
- 공식 2인 독립 검수 및 adjudication 미실시
- 사용자 확인이 `CONFIRMED_AS_DRAFT`라는 사실
- 이진 적격/제외 건수와 split별 label 분포
- Frozen Test label 사전 노출로 인한 평가 독립성 한계
- 작은 표본, 검수자 편향, AI anchoring, 불확실 사례 제외로 인한 선택 편향

성능 수치는 실험 후보 간 비교 자료로만 사용한다. 공식 Ground Truth 성능이나
운영 일반화 성능으로 표현하지 않으며, 운영 활성화에는 별도 공식 검수와 배포
승인이 필요하다.

## 7. 공식 2인 검수와의 차이

| 항목 | 공식 v0.3 경로 | 이 실험 경로 |
|---|---|---|
| 검수자 | 2명 독립 블라인드 | 1명 + AI 보조 draft |
| AI 결과 | 공식 제출로 사용 금지 | 출처를 밝히고 실험에만 사용 |
| 불일치 | adjudication 필수 | 독립 두 번째 판단이 없어 측정 불가 |
| 최종 상태 | adjudicated Human Ground Truth | 비공식 실험 label |
| Frozen Test | 독립 평가 정책 | label 사전 노출 한계 명시 |
| Builder | `build_dataset.py` | `single_reviewer_experiment.py` |
