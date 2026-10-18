# P_text Short / Low-information Review Audit

## 목적

짧거나 정보량이 적은 리뷰가 의심 신호 부재만으로 높은 `text_score`에 수렴하는지 진단한다. 짧은 리뷰를 fake 또는 suspicious로 가정하지 않는다.

## 현재 P_text 기준

- Model: `ptext-koelectra-v1-2epoch-20260929`
- Runtime: `app.analyzers.p_text.predict_text_score`
- Inference config: `{"threshold": 0.5, "max_length": 256, "score_formula": "round((1-p_suspicious)*100)", "dataset_sha256": "cd3812b26dd7c782e01ca919c0ff12e3a8bf57e81ecc218522c7e476a7490673"}`
- `text_score`에만 temperature T=2가 적용되고 label은 raw probability/threshold를 따른다.

## 데이터 범위

- 6개 플랫폼, raw rows: 1,151,943
- Blank normalized content: 0
- 원본 CSV는 UTF-8, streaming, read-only 방식으로 처리했다.

## Population length distribution

| Bucket | Raw rows | Raw ratio | Unique normalized content |
|---|---:|---:|---:|
| 1~5 | 436 | 0.04% | 193 |
| 6~10 | 25,829 | 2.24% | 18,496 |
| 11~20 | 242,014 | 21.01% | 211,588 |
| 21~40 | 324,261 | 28.15% | 300,860 |
| 41~80 | 287,609 | 24.97% | 257,776 |
| 81+ | 271,794 | 23.59% | 249,523 |

## Sampling methodology

- Per platform x length bucket, select the lexicographically smallest SHA-256 digests of platform + normalized content; duplicate normalized content is excluded. A separate full row-key SHA-256 chooses canonical metadata deterministically.
- Sampling unit: unique normalized content within each platform × bucket.
- Raw rows, unique contents, and inferred samples are reported separately.

## Pilot result

- Status: succeeded
- Samples: 284
- Runtime: 37.79s
- Unavailable: 0
- Model load/output validation succeeded before full expansion.

## Full audit sample size

- Total inferred: 1296
- Total inference runtime (pilot included): 159.25s

## Length-bucket result

| Bucket | N | Mean score | Median | Min | Max | Mean p(susp.) | Median p(susp.) | >=90 | >=95 | NORMAL | SUSPICIOUS | Unavailable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1~5 | 77 | 93.69 | 94.00 | 75 | 95 | 0.005632 | 0.003402 | 97.40% | 44.16% | 100.00% | 0.00% | 0.00% |
| 6~10 | 122 | 94.83 | 95.00 | 92 | 95 | 0.002983 | 0.002736 | 100.00% | 87.70% | 100.00% | 0.00% | 0.00% |
| 11~20 | 208 | 94.87 | 95.00 | 91 | 95 | 0.002744 | 0.002490 | 100.00% | 94.71% | 100.00% | 0.00% | 0.00% |
| 21~40 | 289 | 95.25 | 95.00 | 95 | 96 | 0.002298 | 0.002265 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| 41~80 | 300 | 95.75 | 96.00 | 95 | 96 | 0.002199 | 0.002184 | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| 81+ | 300 | 95.73 | 96.00 | 92 | 96 | 0.002222 | 0.002180 | 100.00% | 99.67% | 100.00% | 0.00% | 0.00% |

## Platform × bucket observations

- **elevenst**: raw=171,052, blank=0, 1~20 ratio=13.07%, inferred=300.
- **kurly**: raw=463,843, blank=0, 1~20 ratio=52.97%, inferred=275.
- **musinsa**: raw=118,192, blank=0, 1~20 ratio=0.06%, inferred=206.
- **naver**: raw=1,286, blank=0, 1~20 ratio=12.67%, inferred=218.
- **ohouse**: raw=6,828, blank=0, 1~20 ratio=0.01%, inferred=140.
- **oliveyoung**: raw=390,742, blank=0, 1~20 ratio=0.00%, inferred=157.

| Platform | Bucket | N | Mean score | Median | >=90 | >=95 | NORMAL | SUSPICIOUS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| elevenst | 1~5 | 50 | 94.02 | 94.00 | 100.00% | 46.00% | 100.00% | 0.00% |
| elevenst | 6~10 | 50 | 94.94 | 95.00 | 100.00% | 94.00% | 100.00% | 0.00% |
| elevenst | 11~20 | 50 | 95.00 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| elevenst | 21~40 | 50 | 95.26 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| elevenst | 41~80 | 50 | 95.80 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| elevenst | 81+ | 50 | 95.70 | 96.00 | 100.00% | 98.00% | 100.00% | 0.00% |
| kurly | 1~5 | 25 | 93.00 | 94.00 | 92.00% | 44.00% | 100.00% | 0.00% |
| kurly | 6~10 | 50 | 94.78 | 95.00 | 100.00% | 86.00% | 100.00% | 0.00% |
| kurly | 11~20 | 50 | 95.00 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| kurly | 21~40 | 50 | 95.18 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| kurly | 41~80 | 50 | 95.62 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| kurly | 81+ | 50 | 95.70 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| musinsa | 1~5 | 2 | 94.00 | 94.00 | 100.00% | 0.00% | 100.00% | 0.00% |
| musinsa | 6~10 | 4 | 93.75 | 93.50 | 100.00% | 25.00% | 100.00% | 0.00% |
| musinsa | 11~20 | 50 | 94.70 | 95.00 | 100.00% | 86.00% | 100.00% | 0.00% |
| musinsa | 21~40 | 50 | 95.24 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| musinsa | 41~80 | 50 | 95.76 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| musinsa | 81+ | 50 | 95.74 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| naver | 1~5 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% |
| naver | 6~10 | 18 | 94.89 | 95.00 | 100.00% | 88.89% | 100.00% | 0.00% |
| naver | 11~20 | 50 | 95.00 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| naver | 21~40 | 50 | 95.20 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| naver | 41~80 | 50 | 95.82 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| naver | 81+ | 50 | 95.68 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| ohouse | 1~5 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% |
| ohouse | 6~10 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% |
| ohouse | 11~20 | 1 | 95.00 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| ohouse | 21~40 | 39 | 95.31 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| ohouse | 41~80 | 50 | 95.86 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| ohouse | 81+ | 50 | 95.90 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| oliveyoung | 1~5 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% |
| oliveyoung | 6~10 | 0 | N/A | N/A | 0.00% | 0.00% | 0.00% | 0.00% |
| oliveyoung | 11~20 | 7 | 93.29 | 92.00 | 100.00% | 42.86% | 100.00% | 0.00% |
| oliveyoung | 21~40 | 50 | 95.34 | 95.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| oliveyoung | 41~80 | 50 | 95.64 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |
| oliveyoung | 81+ | 50 | 95.66 | 96.00 | 100.00% | 100.00% | 100.00% | 0.00% |

## Short vs long comparison

- **1~20**: n=407, mean=94.63, median=95.00, >=90=99.51%, >=95=83.05%, mean p(susp.)=0.003362, SUSPICIOUS=0.00%.
- **41+**: n=600, mean=95.74, median=96.00, >=90=100.00%, >=95=99.83%, mean p(susp.)=0.002210, SUSPICIOUS=0.00%.

## Diagnostic examples

각 bucket의 최고/최저 score 예시다. 긴 원문은 120자로 잘라 표시한다.

| Bucket | Extreme | Platform | Length | Score | p(susp.) | Label | Content |
|---|---|---|---:|---:|---:|---|---|
| 1~5 | highest | elevenst | 5 | 95 | 0.00309233 | NORMAL | 빨리왔어요 |
| 1~5 | lowest | kurly | 5 | 75 | 0.09836496 | NORMAL | 커리 합격 |
| 6~10 | highest | kurly | 9 | 95 | 0.00282315 | NORMAL | 만족합니다 ㅎㅎ! |
| 6~10 | lowest | kurly | 8 | 92 | 0.00741252 | NORMAL | 냉동실에 보관증 |
| 11~20 | highest | kurly | 11 | 95 | 0.00259863 | NORMAL | 조아하는 샐러드에요. |
| 11~20 | lowest | musinsa | 20 | 91 | 0.00983385 | NORMAL | 심플123456789000000000 |
| 21~40 | highest | musinsa | 40 | 96 | 0.00215640 | NORMAL | 뱃살이 많아서 원하는 핏은 안 나오지만 가볍고 뒤에 주머니 있어서 좋아요 |
| 21~40 | lowest | kurly | 24 | 95 | 0.00223860 | NORMAL | 맛있어요 삼삼한게 느끼하지고 않고 좋습미다. |
| 41~80 | highest | kurly | 43 | 96 | 0.00218179 | NORMAL | 꼬깔콘중 이 맛이 제일 맛있어요 자극적이지도않고 고소하고 막 집어먹기 짱이에요 |
| 41~80 | lowest | musinsa | 63 | 95 | 0.00227509 | NORMAL | 꾸준히 사용하는 마스크로 마데카소사이드 함량이 다른 제품보다 높으며 피부가 매끈해져요. 무신사가 가장 저렴했어요. |
| 81+ | highest | oliveyoung | 280 | 96 | 0.00219473 | NORMAL | 남성애게 좋은 영양제라고 해서 아빠 선물하려고 구매했어요 남성건강을 위한 비오틴, 아연이 강화되어 있다고 해요 특히 오메가3와 마그네슘, 비타민B로 구성되어 있어 종합 건강관리에 도움을 줄 수 있어요 개별 포장... |
| 81+ | lowest | elevenst | 162 | 92 | 0.00712224 | NORMAL | 좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것같아요좋은것... |

Short samples with `text_score >= 90`:

- `kurly` / 11~20 / len=11 / score=95 / p=0.00259863 / NORMAL: 조아하는 샐러드에요.
- `kurly` / 11~20 / len=13 / score=95 / p=0.00246880 / NORMAL: 생연어회 좋아요 맛있어요
- `kurly` / 11~20 / len=16 / score=95 / p=0.00284803 / NORMAL: 떨어지지 않게 매번 구매합니다
- `elevenst` / 11~20 / len=17 / score=95 / p=0.00233598 / NORMAL: 레이어드되어 있어서 더 예쁘네요
- `kurly` / 11~20 / len=12 / score=95 / p=0.00262893 / NORMAL: 늘 주문합니다 좋습니다

## Key findings 및 A/B/C interpretation

**C (closest), with global saturation caveat.**

C is the closest category because sampled 1~20 character reviews broadly saturate at >=90. However, the 41+ group saturates at least as strongly. Across this unlabeled new-data sample, P_text outputs concentrate globally in the NORMAL direction, mostly around scores 94~96; this does not support a short-specific effect and does not establish ground-truth generalization failure.

- `short_specific_effect`: **not supported**
- `global_score_saturation`: **observed**
- `ground_truth_generalization_failure`: **not established**

Diagnostic heuristic: C when every 1~20 bucket has >=70% scores >=90; B when either 1~5 or 6~10 does; otherwise A. This is not a policy threshold.

## Limitations

- Sampled inference, not inference over all raw rows.
- The new real-world sample is unlabeled; this is not a ground-truth performance evaluation, and no Accuracy, Recall, or FPR is estimated.
- NORMAL output concentration does not establish model generalization failure or prove that suspicious reviews cannot be detected.
- Short reviews are not assumed to be fake or suspicious.
- Writing style may differ by platform.
- Unique-content sampling differs from actual traffic weighting.
- Weak suspicious evidence is not equivalent to sufficient trustworthy evidence.
- A separate positive-control inference has not yet been run.

## Policy recommendation for NEXT STEP only

- 신규 1,296건을 재실행하지 말고, known NORMAL 5~10건과 known SUSPICIOUS 5~10건의 작은 positive-control sanity check를 먼저 수행한다.
- Human-reviewed label과 previous-model expected label을 별도 provenance로 기록하고 같은 의미로 취급하지 않는다.
- Control이 통과하면 신규 unlabeled 표본의 분포·calibration·text sufficiency를 후속 검토하고, 실패하면 runtime/model/config 정합성을 먼저 조사한다.
- 이 audit만으로 일반화 실패나 suspicious-review 미탐지를 주장하지 않는다.
- 이번 audit에서는 threshold, score mapping, unavailable/RTI 정책을 변경하지 않았다.
