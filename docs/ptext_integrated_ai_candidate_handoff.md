# Re:view 통합 AI v3 후보 전달 안내

## 상태

이 패키지는 팀 검토용 `OFFLINE_TEAM_REVIEW_CANDIDATE_NOT_RUNTIME_DEFAULT`다.
운영 API, Meta-Scorer, 기본 P_text 모델 경로 또는 RTI 정책을 변경하지 않는다.

- P_text: v3.1 B (`model.safetensors` SHA-256
  `012b65174cdbc2e1fb36d75519438043bebd4317a69ea9997325b66bb436474b`)
- P_network/P_behavior: 기존 baseline
- original RTI: 기존 Meta-Scorer
- 후보 등급: `review_evidence_tier_v3`
- 정책 지수: `BALANCED_INTERIOR_V1`
- 지수 해석: `ORDINAL_V3_POLICY_INDEX_NOT_PROBABILITY`

모델은 기존 별도 ZIP
`ptext-v3.1-b-experimental-seed20261115-share-20261116.zip`을 재사용한다.
모델 ZIP SHA-256은
`f1fc7abe2248c32a9d310834ca38687e17d72d4eb24a528045be345836e368b9`다.

## 설치 및 실행

Python 3.12 CPU 환경에서 검증했다.

```powershell
python -m venv venv
venv\Scripts\python -m pip install -r requirements-ai.txt
venv\Scripts\python -m pytest -q tests
venv\Scripts\python -m app.training.p_text_v3.run_integrated_ai_candidate `
  --model-path C:\path\to\extracted-model\model `
  --synthetic
```

실제 입력 파일 실행:

```powershell
venv\Scripts\python -m app.training.p_text_v3.run_integrated_ai_candidate `
  --model-path C:\path\to\extracted-model\model `
  --input-json examples\single_input.json `
  --output-json result.json
```

모델 파일은 로컬에서만 읽으며 네트워크 다운로드를 사용하지 않는다. 시작 시
`model.safetensors` SHA를 한 번 검증한 뒤 predictor를 생성한다.

## Python 호출 계약

```python
from app.training.p_text_v3.integrated_ai_candidate import (
    IntegratedAICandidate, IntegratedReviewInput,
)

analyzer = IntegratedAICandidate(r"C:\path\to\model", device="cpu")
result = analyzer.analyze_one(IntegratedReviewInput(
    platform="oliveyoung",
    product_id="product-example",
    review_id="review-example",
    content="2주 동안 사용했고 오후에는 조금 건조했지만 지속력은 좋았습니다.",
    rating=4,
    rating_scale_max=5,
    rating_source="crawler_raw_verified",
    rating_source_verified=True,
    image_count=1,
    image_count_source="crawler_raw_verified",
    image_count_source_verified=True,
))
payload = result.to_dict()
```

같은 상품 리뷰는 `analyze_product_batch()`로 함께 전달한다. 서로 다른 product_id 또는
중복 review_id가 포함되면 fail-closed한다. 단일 리뷰는 Network peer가 없는 정상적인
unavailable 결과가 된다.

## nullable 및 provenance 계약

- 확인되지 않은 `rating`, `image_count`는 `null`로 전달한다.
- `null`을 별점 5점 또는 사진 0장으로 대체하지 않는다.
- rating 값이 있으면 `rating_scale_max=5`, 비어 있지 않은 `rating_source`,
  `rating_source_verified=true`가 모두 필요하다.
- image_count 값이 있으면 비어 있지 않은 `image_count_source`와
  `image_count_source_verified=true`가 모두 필요하다.
- 닉네임을 user_id로 사용하지 않는다. Behavior 관측값이 없으면 `behavior=null`이다.

## 주요 출력

```json
{
  "original_rti": 54.0,
  "original_level": "WARN",
  "text_score": 54.0,
  "behavior_score": null,
  "network_score": null,
  "evidence_level_v3": "SAFE",
  "policy_index_v3": 85.8,
  "policy_index_type": "ORDINAL_V3_POLICY_INDEX",
  "short_text_penalty": 0.0,
  "experience_evidence": "YES",
  "promotional_evidence": "NO",
  "available_signals": ["text"],
  "unavailable_signals": ["behavior", "network"]
}
```

`original_rti/original_level`과 후보 등급·지수는 별도 결과다. 정책 지수는 실제 리뷰가
진실일 확률이나 모델 confidence가 아니다.

## 운영 기본값과 차이 및 복귀

운영 경로는 기존 P_text v1과 기존 RTI `level`을 사용한다. 이 패키지는 별도 Python
호출에서만 v3.1 B와 후보 정책을 계산한다. 복귀하려면 호출 측에서 이 어댑터 사용을
중단하고 기존 AI API 응답을 계속 사용하면 된다. 운영 코드나 모델 파일을 되돌릴
변경 사항은 없다.

## 제한 사항

- 공식 2인 Human Ground Truth 및 독립 평가 150건이 완료되지 않았다.
- Frozen Test를 이 패키지 검증에 사용하지 않았다.
- 정책 지수 기준점과 감점 폭은 outcome에서 학습된 값이 아니다.
- 등급 우선 지수라 39/40 및 69/70 사이에 의도적인 단절이 있다.
- Network는 전달된 동일 상품 리뷰 묶음 안의 관계만 분석한다.
- Behavior는 실제 관측값이 있을 때만 available이다.
- 603건 회귀 결과는 코드 재현 감사이며 실제 정확도 평가가 아니다.
