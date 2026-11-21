# Re:view P_text Human Review 라벨링 가이드 v0.3

- 문서 상태: 공식 검수용
- 발행일: 2026-11-13
- 가이드 버전: `ptext-labeling-guide-v0.3`
- 검수 계약: `ptext-human-review-contract-v0.3`
- 적용 범위: P_text v3 실제 리뷰 Reviewer A/B 독립 검수 및 adjudication
- 비적용 범위: 모델 학습, Runtime·RTI 변경, 외부 API 계약 변경

## 1. 목적과 v0.2 호환성

이 문서는 리뷰 텍스트에서 직접 관찰 가능한 신뢰 위험 신호를 두 검수자가 같은 기준으로 독립 판정하기 위한 공식 기준이다. P_text는 텍스트만으로 실제 조작 리뷰, 구매 사실, 작성자 신원, 캠페인 참여 또는 제품 효능의 사실 여부를 확정하지 않는다.

v0.3은 v0.2 Draft의 다음 정책을 변경하지 않는다.

- `text_status`와 `signal_decision`의 2축 라벨 구조
- 두 축의 허용 조합
- 다섯 가지 `signal_types`와 multi-label 원칙
- 정보 부족은 의심 신호가 아니라는 원칙
- 샘플링 provenance와 모델 출력을 정답으로 사용하지 않는 원칙
- Reviewer A/B 독립 blind review 및 원시 라벨 보존 원칙
- 불일치 adjudication과 `UNCERTAIN` 유지 가능성
- Result Contract v0.5와 현재 Runtime·RTI 비변경 원칙

v0.3의 추가 사항은 검수 근거 기록 의무의 명확화와 부록 A의 8개 `PROPOSED_` rationale code 정의다. 이 추가 사항은 라벨 또는 신호 vocabulary를 확장하지 않는다.

## 2. 고정 원칙

1. `CANDIDATE`, `CONTROL`, `DIVERSITY`는 샘플링 provenance이며 정답 라벨이 아니다.
2. 기존 모델 예측, heuristic score, 후보 source, 별점 및 사진 정보는 텍스트 라벨의 근거가 아니다.
3. 기존 v2 라벨을 신규 라벨로 자동 변환하지 않는다.
4. 검수자는 리뷰 원문에서 관찰 가능한 표현과 구조만 판정한다.
5. 글자 수만으로 `SUFFICIENT` 또는 `INSUFFICIENT`를 결정하지 않는다.
6. 정보가 부족하다는 사실은 `PRESENT` 또는 실제 조작의 증거가 아니다.
7. `ABSENT`는 다섯 신호가 관찰되지 않았다는 뜻이며 정상 구매나 사실성을 증명하지 않는다.
8. 리뷰 간 exact/near duplicate와 한 리뷰 내부 반복을 구분한다. 전자는 leakage/network 관리 대상이고 후자만 P_text 신호 후보다.
9. 사람 검수 원본, adjudication 결과, AI 보조 제안은 서로 구분하여 보존한다.
10. 이 가이드 발행으로 Runtime, RTI, Result Contract 또는 외부 reason code를 변경하지 않는다.

## 3. 공식 라벨 vocabulary

### 3.1 `text_status`

허용 값은 다음 세 개뿐이다.

| 값 | 정의 |
|---|---|
| `SUFFICIENT` | 텍스트만으로 정의된 신호의 존재 여부를 판단할 수 있다. |
| `INSUFFICIENT` | 유효한 리뷰지만 정의된 신호를 판단할 의미 정보가 부족하다. |
| `INVALID` | 리뷰 텍스트가 아니거나 손상·오입력 등으로 분석 대상이 아니다. |

### 3.2 `signal_decision`

허용 값은 다음 네 개뿐이다.

| 값 | 정의 |
|---|---|
| `PRESENT` | 하나 이상의 정의된 신호가 명확하게 관찰된다. |
| `ABSENT` | 충분한 텍스트를 검토했을 때 정의된 다섯 신호가 관찰되지 않는다. |
| `UNCERTAIN` | 텍스트는 충분하지만 PRESENT와 ABSENT 중 하나로 일관되게 확정하기 어렵다. |
| `NOT_ASSESSED` | 정보 부족 또는 입력 무효로 신호 판정을 수행하지 않았다. |

### 3.3 허용 조합

| `text_status` | 허용 `signal_decision` | `signal_types` 규칙 |
|---|---|---|
| `SUFFICIENT` | `PRESENT` | 하나 이상 필수 |
| `SUFFICIENT` | `ABSENT` | 비어 있어야 함 |
| `SUFFICIENT` | `UNCERTAIN` | 의심되는 후보 유형을 0개 이상 기록할 수 있으나 학습 positive로 사용하지 않음 |
| `INSUFFICIENT` | `NOT_ASSESSED` | 비어 있어야 함 |
| `INVALID` | `NOT_ASSESSED` | 비어 있어야 함 |

다른 조합은 스키마 오류다. 특히 `PRESENT`인데 신호 유형이 없거나, `ABSENT` 또는 `NOT_ASSESSED`인데 신호 유형이 있는 조합은 허용하지 않는다.

### 3.4 `signal_types`

허용 값은 다음 다섯 개뿐이며 명확한 경우 복수 선택할 수 있다.

1. `PRODUCT_DESCRIPTION_LIKE`
2. `NO_USE_ASSERTIVE_CLAIM`
3. `PROMOTIONAL_OR_SALES_LANGUAGE`
4. `HIGH_CERTAINTY_EFFICACY_CLAIM`
5. `WITHIN_REVIEW_REPETITION`

`OTHER_REVIEW_NEEDED`는 별도 검토 태그이며 `signal_types`가 아니다. 새로운 반복 패턴을 발견해도 검수자가 임의로 여섯 번째 신호를 만들지 않는다.

## 4. 정보 충분성

### 4.1 `SUFFICIENT`

다음과 같은 판단 가능한 정보가 하나 이상 있으면 짧더라도 `SUFFICIENT`일 수 있다.

- 구체적인 제품 속성 또는 사용 상황·기간·비교·결과
- 명확한 구매 촉구나 광고성 표현
- 미사용과 강한 효과 단정의 결합
- 절대적·의학적 고확신 주장
- 비정상적인 리뷰 내부 반복

### 4.2 `INSUFFICIENT`

일반 감탄, 배송·포장만의 짧은 언급, 의미 없는 이모지, 구체성이 없는 극단적 축약처럼 다섯 신호의 존재 여부를 판단할 정보가 부족한 경우다. `INSUFFICIENT`는 `NOT_ASSESSED`와만 결합한다.

“좋아요”, “배송 빨라요”처럼 신호가 보이지 않는다는 이유만으로 `ABSENT`를 부여하지 않는다. 반대로 짧더라도 판매 촉구나 비정상 반복이 명확하면 `SUFFICIENT / PRESENT`일 수 있다.

### 4.3 `INVALID`

빈값, 해석 불가능한 인코딩 손상, 시스템 식별자·HTML 조각만 있는 값, 리뷰가 아닌 오입력처럼 분석 가능한 본문이 아닌 경우다. 욕설, 오탈자, 비표준 문법 또는 외국어라는 이유만으로 `INVALID`로 판정하지 않는다.

## 5. 다섯 신호 판정 기준

### 5.1 `PRODUCT_DESCRIPTION_LIKE`

개인의 구매·사용 후기보다 제품 사양, 성분, 기능, 효능 또는 사용 대상을 설명하는 카탈로그형 문체가 본문을 지배하는 경우다.

- PRESENT 근거: 사양·기능의 연속 나열, 비개인적 소개 구조, 일반 사용자에게 동일하게 적용되는 효과 소개, 개인 경험보다 판매 페이지형 설명이 우세함.
- 제외 경계: 구체적인 사용 기간·상황·비교·장단점과 연결된 제품 정보, 단지 문장이 길거나 정돈된 경우.
- 설명과 개인 경험의 비중이 비슷해 중심을 정하기 어렵다면 `UNCERTAIN`을 고려한다.

### 5.2 `NO_USE_ASSERTIVE_CLAIM`

미사용·미개봉 또는 충분히 사용하지 못했음을 명시하면서 효과·성능·품질을 강하게 확정하는 경우다. 다음 두 조건을 모두 충족해야 한다.

1. 사용하지 않았거나 경험이 부족하다는 명시적 표현
2. 그럼에도 효과·성능·품질을 확정하는 표현

미사용 상태에서 배송·포장·외관만 평가하거나 효과를 모른다고 밝히는 경우, 단순 기대를 표현하는 경우는 이 신호가 아니다.

### 5.3 `PROMOTIONAL_OR_SALES_LANGUAGE`

개인 후기보다 구매 촉구, 긴급성, 혜택 강조 또는 판매 목적 표현이 본문을 지배하는 경우다.

- PRESENT 근거: 반복적인 직접 구매 촉구, 한정·마감·재고 압박, 혜택만을 중심으로 한 판매 유도, 슬로건형 문구가 본문 대부분을 차지함.
- 제외 경계: 개인 경험에 기반한 일반 추천, 재구매 의사, 본인의 할인 구매 경험.
- “추천한다” 한 번만으로 PRESENT를 부여하지 않는다.

### 5.4 `HIGH_CERTAINTY_EFFICACY_CLAIM`

효과를 모든 사용자에게 보장되는 것처럼 표현하거나 치료·예방·즉시 효과 등 결과를 높은 확신으로 단정하는 경우다. 검수자는 실제 효능이나 연구 존재 여부를 조사하지 않고 텍스트의 확신 강도만 본다.

- PRESENT 근거: 반드시·무조건·완벽하게 등의 절대 보장, 질환 치료·예방의 확정, 누구에게나 같은 결과, 근거 설명 없는 임상·통계 결과의 확정적 제시.
- 제외 경계: 개인 체감으로 한정하고 불확실성을 밝힌 표현, 희망·예상, 강한 만족만 있는 표현.

### 5.5 `WITHIN_REVIEW_REPETITION`

한 리뷰 안에서 동일하거나 매우 유사한 단어·구절·문장이 비정상적으로 반복되어 정보 전달을 방해하거나 복사·자동 생성과 유사한 구조를 보이는 경우다.

- PRESENT 근거: 동일 문장의 연속 반복, 일부만 바뀐 문장이 본문 대부분을 차지함, 긴 구간의 복사 중복, 반복으로 인한 큰 정보 손실.
- 제외 경계: 자연스러운 구어체 강조, 핵심 속성의 재언급, 서로 다른 정보를 전달하는 반복 단어.
- 다른 리뷰와의 중복은 이 신호가 아니다.

## 6. `signal_types`와 `rationale_codes`의 구분

`signal_types`는 PRESENT 또는 UNCERTAIN 판정에서 관찰된 정식 신호의 종류다. 모델 학습과 유형별 분석에 사용될 수 있다.

`rationale_codes`는 검수자가 왜 해당 판정 또는 경계 판단을 했는지 구조화하여 기록하는 근거다. 다음 원칙을 적용한다.

- rationale code는 새로운 신호 유형이 아니다.
- rationale code만으로 `PRESENT`, `ABSENT` 또는 `NOT_ASSESSED`가 자동 결정되지 않는다.
- `PROPOSED_` 접두사는 사전 제안 또는 경계 검토 항목임을 뜻하며 ground truth가 아니다.
- rationale code를 `signal_types` 필드에 넣지 않는다.
- 같은 rationale code가 서로 다른 최종 판정에서 사용될 수 있다. 최종 판정은 본문과 본 가이드 전체를 적용해 사람이 결정한다.
- 공식 검수자가 제안에 동의하지 않으면 해당 코드를 유지할 의무가 없다. 필요한 경우 `rationale_note`에 기각 또는 대체 근거를 짧게 기록한다.

## 7. Reviewer A/B 공식 검수 절차

### 7.1 독립 blind review

1. Reviewer A와 Reviewer B는 원본 blind packet을 각각 독립 검수한다.
2. 검수 종료 전 상대 검수자의 라벨, 근거 코드 또는 note를 확인하지 않는다.
3. sampling group, selection category, heuristic score, 후보 source, 모델·AI 예측 및 split 역할을 판단 화면에 노출하지 않는다.
4. 두 reviewer의 표본 순서는 독립적으로 무작위화된 기존 packet 순서를 유지한다.
5. AI-assisted staging을 참고한 검수는 독립적인 사람 blind review로 표시하지 않는다. AI 제안은 사람 원본 라벨과 별도 provenance로 관리한다.

### 7.2 필수 기록

각 reviewer는 모든 행에 다음을 기록한다.

- `annotator_id`
- `text_status`
- `signal_decision`
- 허용 조합에 필요한 `signal_types`
- `other_review_needed`
- 판정 근거를 설명하는 `rationale_codes` 또는 짧은 `rationale_note`
- `guideline_version=ptext-labeling-guide-v0.3`
- `annotation_round`
- 시간대가 포함된 `annotated_at`

PRESENT, UNCERTAIN, 정보 부족 경계, 두 신호 사이 경계 사례는 최소 하나의 구조화된 근거 코드 또는 구체적인 note를 남긴다. ABSENT도 단순 기본값으로 채우지 않고 충분한 텍스트를 검토했다는 근거가 드러나야 한다.

### 7.3 판단 순서

1. 리뷰 본문이 유효한지 판단한다.
2. 다섯 신호를 판단할 정보가 충분한지 판단한다.
3. 다섯 신호를 각각 검토한다.
4. 명확한 신호를 모두 `signal_types`에 기록한다.
5. 허용 조합에 맞게 전체 판정을 기록한다.
6. 경계·정보 부족·복수 신호 근거를 rationale에 기록한다.

## 8. 불일치와 adjudication

1. Reviewer A/B의 원시 라벨과 근거를 수정하거나 덮어쓰지 않는다.
2. `text_status`, `signal_decision`, `signal_types`, 대표 근거 및 INVALID/INSUFFICIENT 경계 불일치를 구분한다.
3. 두 reviewer가 각자 근거를 기록한 뒤 가이드 조항을 기준으로 재검토한다.
4. 합의되지 않으면 별도 adjudicator가 `final_*` 필드에 최종 판정을 기록한다.
5. 다수결, 평균, 모델 점수 또는 AI 제안으로 final label을 만들지 않는다.
6. 정책적으로 모호한 사례는 억지로 이진 확정하지 않고 `UNCERTAIN`을 유지할 수 있다.
7. 반복되는 정책 경계는 adjudication에서 임의 확정하지 않고 가이드 개정 검토 항목으로 별도 기록한다.

## 9. 자동 검증 규칙

검수 결과 통합 전 최소한 다음을 검사한다.

1. 허용되지 않은 `text_status × signal_decision` 조합이 0건인지 확인
2. PRESENT의 `signal_types`가 하나 이상인지 확인
3. ABSENT와 NOT_ASSESSED의 `signal_types`가 비어 있는지 확인
4. label·signal vocabulary가 본 문서의 고정 목록과 일치하는지 확인
5. `rationale_codes`가 `signal_types`에 섞이지 않았는지 확인
6. reviewer ID, 가이드 버전, 검수 차수, 검수 시각이 채워졌는지 확인
7. 두 reviewer의 annotation ID 집합이 동일한지 확인
8. 사람 원시 라벨과 AI-assisted staging provenance가 분리됐는지 확인
9. adjudication이 원시 reviewer 파일을 수정하지 않았는지 확인

## 10. AI 보조 제안 취급

AI가 미리 채운 라벨, 신호 또는 `PROPOSED_` 코드는 공식 사람 검수 결과가 아니다. 다음을 지킨다.

- AI-assisted staging은 reviewer 원본 packet 및 사람 annotation과 별도 파일로 유지한다.
- 빈 `annotator_id`, 미완료 시각 또는 AI provenance가 있는 행을 사람 독립 검수 완료로 집계하지 않는다.
- 사람이 각 본문을 직접 검토하고 자신의 라벨과 근거를 기록해야 한다.
- AI 제안과의 일치 여부는 검수 완료 후 별도 분석할 수 있으나, reviewer 간 agreement에 AI 결과를 포함하지 않는다.
- 정책적 판단이 필요한 제안은 adjudication 또는 가이드 검토 항목으로 남기며 자동 확정하지 않는다.

## 부록 A. `PROPOSED_` 판단 근거 코드

아래 8개 코드는 `rationale_codes`에만 사용할 수 있는 경계·검토 근거다. 모두 신호 유형이나 정답 라벨이 아니다.

### A.1 `PROPOSED_BOUNDARY_EFFICACY_CLAIM`

효과·성능 주장은 있으나 개인 체감, 예상, 제한 표현과 고확신 단정의 경계가 모호해 `HIGH_CERTAINTY_EFFICACY_CLAIM` 여부를 검토해야 함을 뜻한다. 절대성, 적용 대상의 일반화, 불확실성 표현을 함께 확인한다.

### A.2 `PROPOSED_REPEAT_WITHIN_TEXT`

동일·유사 표현의 반복이 관찰되어 `WITHIN_REVIEW_REPETITION` 후보를 검토해야 함을 뜻한다. 자연스러운 강조인지, 반복이 본문을 지배하고 정보 손실을 만드는지 사람이 확인한다.

### A.3 `PROPOSED_BOUNDARY_PROMOTIONAL_LANGUAGE`

추천·혜택·구매 관련 표현이 있지만 개인 경험 기반 추천과 판매 목적 언어의 경계가 모호함을 뜻한다. 직접 구매 촉구, 긴급성, 반복성과 본문 내 지배 비중을 확인한다.

### A.4 `PROPOSED_LOW_INFO_NOT_ASSESSED`

본문 정보가 적어 `INSUFFICIENT / NOT_ASSESSED` 후보를 검토해야 함을 뜻한다. 짧다는 이유만으로 확정하지 않고, 다섯 신호 중 하나라도 판단 가능한 표현 구조가 있는지 확인한다.

### A.5 `PROPOSED_BOUNDARY_CONTEXT_UNCLEAR`

대상, 화자, 인용 여부 또는 문맥이 불명확해 PRESENT·ABSENT를 안정적으로 결정하기 어려움을 뜻한다. 정보 자체가 부족하면 INSUFFICIENT를, 텍스트는 충분하나 경계가 남으면 UNCERTAIN을 검토한다.

### A.6 `PROPOSED_PRODUCT_INFO_DOMINANT`

제품 사양·기능·효능 설명이 개인 경험보다 본문을 지배하는 것으로 보인다는 근거다. 사람이 전체 비중과 개인 사용 맥락을 확인한 뒤 `PRODUCT_DESCRIPTION_LIKE` 여부를 결정한다.

### A.7 `PROPOSED_BOUNDARY_PROMO_EFFICACY`

하나의 표현이 판매 유도와 고확신 효능 주장 양쪽에 걸칠 수 있어 두 신호의 복수 선택 여부를 검토해야 함을 뜻한다. 각 신호의 요건을 독립적으로 충족하는 경우에만 둘 다 선택한다.

### A.8 `PROPOSED_BOUNDARY_PRODUCT_DESCRIPTION`

제품 정보가 포함돼 있으나 정상 사용 후기와 카탈로그형 설명의 경계가 모호함을 뜻한다. 제품 정보의 존재 자체가 아니라 설명이 본문을 지배하는지, 개인 관찰과 연결되는지를 확인한다.

## 부록 B. 공식 고정 vocabulary

```text
text_status:
  SUFFICIENT
  INSUFFICIENT
  INVALID

signal_decision:
  PRESENT
  ABSENT
  UNCERTAIN
  NOT_ASSESSED

signal_types:
  PRODUCT_DESCRIPTION_LIKE
  NO_USE_ASSERTIVE_CLAIM
  PROMOTIONAL_OR_SALES_LANGUAGE
  HIGH_CERTAINTY_EFFICACY_CLAIM
  WITHIN_REVIEW_REPETITION
```

이 목록은 v0.2에서 변경되지 않았다. `PROPOSED_` rationale code와 `OTHER_REVIEW_NEEDED`는 이 목록에 포함되지 않는다.
