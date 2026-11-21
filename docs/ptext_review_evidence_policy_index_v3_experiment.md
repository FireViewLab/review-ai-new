# Re:view review_evidence_tier_v3 정책 지수 실험

## 상태와 목적

이 문서의 점수는 `review_evidence_tier_v3`가 먼저 결정한 SAFE/WARN/DANGER를
표시하기 위한 **순서형 정책 지수**다. 실제 리뷰 진위 확률, 신뢰도, 모델 정확도,
calibration 결과가 아니며 운영 RTI를 대체하지 않는다. 기존 RTI는 별도 값으로
그대로 보존한다.

등급 구간은 SAFE 70~100, WARN 40~69, DANGER 0~39로 고정한다. 등급을 점수에서
추론하여 다시 결정하지 않고, v3 등급 안에서만 점수를 계산한다. 이 tier-first
방식 때문에 39→40 및 69→70에는 의도적인 구간 단절이 있다.

## 공통 산식

`R`은 보존된 기존 RTI(0~100), `L`은 의미 문자 수다.

```text
position = R / 100
base = anchor_low + (anchor_high - anchor_low) * position
length_deficit = max(0, 30 - L) / 30
length_penalty = maximum_length_penalty * length_deficit
policy_index = clamp(base - length_penalty, v3 등급의 점수 구간)
```

같은 v3 등급과 같은 길이에서는 기존 RTI 순서를 선형 보존한다. 같은 다른 조건에서
본문이 짧아질수록 지수가 낮아지며, 30자 이상은 길이 감점이 없다. 사진이 있는
초단문 SAFE도 SAFE 구간 안에서 낮은 값을 갖기 때문에 등급과 길이 신호가 동시에
표현된다.

## 비교 후보

| 후보 | DANGER 기준점 | WARN 기준점 | SAFE 기준점 | 최대 길이 감점 |
|---|---:|---:|---:|---:|
| GENTLE_WIDE_V1 | 3~37 | 43~68 | 73~98 | 4 |
| BALANCED_INTERIOR_V1 | 5~35 | 45~65 | 75~95 | 6 |
| CONSERVATIVE_LENGTH_V1 | 7~35 | 47~65 | 77~95 | 8 |

- GENTLE은 기존 RTI 순서가 표현되는 폭이 넓고 길이 감점이 작지만 구간 경계에
  상대적으로 가깝다.
- BALANCED는 경계 여유, 원 RTI 반영 폭, 길이 감점 사이의 중간 후보다.
- CONSERVATIVE_LENGTH는 짧은 본문 차이를 크게 표현하지만 경계 clamp와 낮은 점수
  집중이 늘어날 수 있다.

`BALANCED_INTERIOR_V1`은 팀 검토를 위한 우선 후보일 뿐 운영 기준으로 확정된 값이
아니다. 기준점과 감점 폭은 검증된 outcome에서 학습된 수치가 아니므로 독립 검토와
UX 검증 없이는 운영에 적용할 수 없다.

## 필수 해석 제한

- 신규 DANGER 후보의 기존 RTI가 54여도 정책 지수는 DANGER 구간에 위치할 수 있다.
  이는 v3 정책 등급의 표현이지 기존 RTI를 39 이하로 수정한 것이 아니다.
- 서로 다른 등급 사이 점수 차이를 연속적인 확률 차이로 해석할 수 없다.
- 100점 또는 구간 경계 포화, 점수·등급 충돌, 동일 조건 단조성은 실험 산출물에서
  별도로 감사한다.
- 독립 평가 150건과 Frozen Test는 이 설계 및 실험에 사용하지 않는다.
- 운영 Meta-Scorer, API, 모델, Network/Behavior 계산은 변경하지 않는다.
