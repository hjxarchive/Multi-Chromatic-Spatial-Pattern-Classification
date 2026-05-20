# Six-Pack Individual Barcode Contribution Analysis

**데이터**: Sixpack_Rips — 61200D raw PI 벡터 (preprocessing 없음)  
**평가**: StandardScaler → PCA(100) → LinearSVC(C=1) + RF(200), StratifiedKFold(5)  
**날짜**: 2026-05-21

---

## 바코드 구성

| 바코드 | 설명 | 차원 (per barcode) |
|---|---|---|
| image | Im(f*) | 2 dirs × (100+5000) = 10200D |
| kernel | Ker(f*) | 10200D |
| cokernel | Cok(f*) | 10200D |
| domain | H*(L) = H*(Rips(A)) | 10200D |
| codomain | H*(K) = H*(Rips(A∪B)) | 10200D |
| relative | H*(K,L) | 10200D |
| **Full six-pack** | 위 6개 합산 | **61200D** |

---

## 실험 1. 단독 바코드 기여도 (10200D raw → PCA100)

| Barcode | LIN Soft | LIN Strict | RF Soft | RF Strict |
|---|---|---|---|---|
| **relative** | **88.30%** | 53.72% | **99.42%** | 76.56% |
| kernel | 84.77% | 55.87% | **97.65%** | 71.28% |
| domain | 84.39% | 51.57% | 97.07% | 75.20% |
| image | 84.58% | 55.09% | 96.10% | 70.70% |
| cokernel | 84.00% | 46.89% | 95.52% | 67.19% |
| codomain | 78.90% | 40.04% | 92.38% | 61.71% |

### 요약 순위 (RF Soft 기준)

1. **relative** — RF Soft=99.42%, RF Strict=76.56%, LIN Soft=88.30%
2. kernel — RF Soft=97.65%, RF Strict=71.28%, LIN Soft=84.77%
3. domain — RF Soft=97.07%, RF Strict=75.20%, LIN Soft=84.39%
4. image — RF Soft=96.10%, RF Strict=70.70%, LIN Soft=84.58%
5. cokernel — RF Soft=95.52%, RF Strict=67.19%, LIN Soft=84.00%
6. codomain — RF Soft=92.38%, RF Strict=61.71%, LIN Soft=78.90%

---

## 실험 2. Full Six-Pack (61200D raw → PCA100)

| Classifier | Soft | Strict |
|---|---|---|
| LinearSVM | 84.18% | 53.14% |
| RF(200) | 97.47% | 77.55% |

> **주목**: Full six-pack RF Soft(97.47%) < relative 단독(99.42%)  
> 나머지 바코드들이 PCA100 공간에서 noise로 작용

---

## 실험 3. Leave-One-Out (50200D raw → PCA100)

| 제거 바코드 | LIN ΔSoft | LIN ΔStrict | RF ΔSoft | RF ΔStrict |
|---|---|---|---|---|
| w/o image | +1.18%p | +2.35%p | +1.56%p | +0.97%p |
| w/o kernel | -2.92%p | -0.39%p | +0.97%p | -0.79%p |
| w/o cokernel | -0.58%p | +0.99%p | +0.39%p | -0.39%p |
| w/o domain | -1.17%p | -0.20%p | +1.36%p | +1.17%p |
| w/o codomain | +0.19%p | +3.51%p | -0.19%p | -0.59%p |
| w/o relative | -2.93%p | +2.91%p | +0.19%p | +0.00%p |

### 요약 순위 (RF ΔSoft 기준, 손해 큰 순)

| 제거 바코드 | RF ΔSoft | RF ΔStrict | LIN ΔSoft |
|---|---|---|---|
| w/o codomain | -0.19%p | -0.59%p | +0.19%p |
| w/o relative | +0.19%p | +0.00%p | -2.93%p |
| w/o cokernel | +0.39%p | -0.39%p | -0.58%p |
| w/o kernel | +0.97%p | -0.79%p | -2.92%p |
| w/o domain | +1.36%p | +1.17%p | -1.17%p |
| w/o image | +1.56%p | +0.97%p | +1.18%p |

> **해석**: LOO에서 모든 Δ가 ±3% 이내 → PCA100 압축으로 개별 기여도가 희석됨  
> LinearSVM에서는 relative(-2.93%), kernel(-2.92%) 제거가 가장 손해  
> RF에서는 상대적으로 차이가 적음 (비선형 결합 효과)

---

## 실험 4. 의미 기반 그룹핑 (raw → PCA100)

| Group | Dim | LIN Soft | RF Soft | RF Strict |
|---|---|---|---|---|
| Relative only | 10200D | 87.13% | **99.42%** | 76.56% |
| IKC + relative | 40800D | 85.36% | 98.05% | 77.74% |
| Ordinary + relative | 30600D | 85.75% | 98.83% | 75.20% |
| Six-pack core (img+ker+cok) | 30600D | 84.39% | 97.47% | 77.35% |
| IKC + ordinary (w/o relative) | 51000D | 80.86% | 97.66% | 77.55% |
| Full (all 6) | 61200D | 83.60% | 97.47% | 77.55% |
| Ordinary (domain+codomain) | 20400D | 82.43% | 97.07% | 73.82% |

> **상위권 모두 relative 포함** → relative barcode의 독보적 분류 기여도 확인

---

## 실험 5. Pairwise Combinations (raw → PCA100, RF Soft 기준)

| Combination | LIN Soft | RF Soft | RF Strict |
|---|---|---|---|
| **kernel + relative** | 85.74% | **99.03%** | 79.10% |
| cokernel + relative | 87.12% | 98.83% | 75.39% |
| codomain + relative | 88.88% | 98.83% | 75.20% |
| domain + relative | 86.92% | 98.63% | 78.91% |
| image + relative | 85.94% | 98.44% | 76.17% |
| kernel + cokernel | 83.41% | 98.24% | 77.15% |
| kernel + domain | 84.57% | 97.46% | 76.56% |
| image + codomain | 83.60% | 97.27% | 69.73% |
| image + domain | 83.41% | 97.08% | 77.35% |
| cokernel + domain | 85.16% | 97.08% | 73.83% |
| image + kernel | 85.16% | 97.07% | 76.18% |
| domain + codomain | 83.61% | 97.07% | 73.82% |
| image + cokernel | 84.98% | 96.49% | 72.47% |
| kernel + codomain | 84.38% | 96.49% | 73.25% |
| cokernel + codomain | 82.82% | 94.34% | 64.06% |

> **상위 5개 모두 relative 포함** — relative가 포함된 쌍이 그렇지 않은 쌍을 모두 상회

---

## 종합 분석

### 1. Relative 바코드의 압도적 기여

H*(K,L) (relative homology)이 단독으로 99.42% (RF Soft)를 달성하며 full six-pack(97.47%)을 상회한다. 이는 relative barcode가 두 공간 L⊆K의 **차이(difference)**를 직접 인코딩하기 때문이다 — 포함 사상 f: L→K에서 어떤 위상 구조가 추가/소멸하는지를 나타내며, 이것이 패턴 분류에 가장 discriminative한 정보다.

### 2. Full Six-Pack이 Relative 단독보다 낮은 이유

PCA100 압축 공간에서 나머지 5개 바코드(61200→10200D)의 정보가 relative의 분류 방향과 직교하거나 noise로 작용한다. PCA가 분산 기준으로 축을 선택하므로 relative의 분류 관련 분산이 희석된다.

### 3. Leave-One-Out의 작은 Δ

모든 LOO Δ가 ±3% 이내인 것은 PCA100 압축의 결과다. 50200D→PCA100에서 정보 손실이 커서 개별 바코드의 한계적 기여가 측정 불가 수준으로 희석된다. PCA 없이 평가하거나 더 높은 PCA 차원을 쓰면 차이가 더 명확하게 나타날 수 있다.

### 4. Codomain 최하위

H*(K) = H*(Rips(A∪B))는 두 공간 합집합의 절대 위상만 담으며, image/kernel/cokernel이 이미 codomain의 정보를 포함한다. rank-nullity 정리(dim domain = dim image + dim kernel)와 제1동형정리(codomain/image ≅ cokernel)에 의한 대수적 중복으로 codomain의 독자적 기여가 가장 낮다.

### 5. LinearSVM vs RF 불일치

| | relative 단독 | LOO w/o relative |
|---|---|---|
| LIN Soft | 88.30% | -2.93%p |
| RF Soft | 99.42% | +0.19%p |

LinearSVM에서는 relative 제거가 -2.93%p 손해이나 RF에서는 거의 0 — RF는 비선형 결합으로 나머지 바코드에서 relative 정보를 복원할 수 있음을 시사한다.

---

## 코드

실험 코드: `barcode_contribution.py`
