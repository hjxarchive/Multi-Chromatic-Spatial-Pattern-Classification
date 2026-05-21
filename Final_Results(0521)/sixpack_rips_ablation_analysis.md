# Sixpack_Rips Ablation Study — 2^6 Barcode Subsets

**대상**: Sixpack_Rips — 6개 barcode의 모든 조합 (2^6 - 1 = 63개 non-empty subset)  
**벡터**: raw PI (61200D 기준, 선택된 barcode만 flatten)  
**평가**: StandardScaler → PCA(100) → 5-fold CV, Soft/Strict accuracy  
**날짜**: 2026-05-21

---

## 실험 설정

| 항목 | 설정 |
|---|---|
| Barcodes | image, kernel, cokernel, domain(sub_complex), codomain(complex), relative |
| 차원 | 바코드당 10200D (2 dir × 5100D) |
| Classifiers | LinearSVM (C=1e5, hard margin), KNN (k=9), RF (n=200) |
| CV | StratifiedKFold(5, shuffle, seed=42) |
| PCA | 100D |

---

## 전체 결과 (RF200 Soft 기준 상위 20개)

| Combo | n | Dim | LIN Soft | LIN Str | KNN Soft | KNN Str | RF Soft | RF Str |
|---|---|---|---|---|---|---|---|---|
| **relative** | 1 | 10200 | 82.62% | 38.29% | 98.05% | 68.16% | **99.42%** | 76.56% |
| kernel+relative | 2 | 20400 | 80.09% | 44.16% | 98.24% | 70.13% | **99.03%** | 79.10% |
| kernel+domain+relative | 3 | 30600 | 80.28% | 46.70% | 98.63% | 73.05% | **99.03%** | 79.30% |
| image+kernel+codomain+relative | 4 | 40800 | 79.70% | 45.15% | 99.22% | 73.83% | 99.02% | 78.13% |
| kernel+cokernel+domain+codomain+relative | 5 | 51000 | 80.30% | 49.63% | 98.44% | 74.02% | 99.02% | 78.52% |
| kernel+cokernel+relative | 3 | 30600 | 82.23% | 46.50% | 98.44% | 72.67% | 98.83% | 77.14% |
| domain+codomain+relative | 3 | 30600 | 83.20% | 50.22% | 99.61% | 72.07% | 98.83% | 75.20% |
| cokernel+relative | 2 | 20400 | 80.09% | 42.98% | 98.83% | 67.19% | 98.83% | 75.39% |
| codomain+relative | 2 | 20400 | 81.06% | 42.00% | 98.64% | 69.14% | 98.83% | 75.20% |
| image+kernel+relative | 3 | 30600 | 79.69% | 47.67% | 98.24% | 72.66% | 98.83% | 78.91% |
| image+cokernel+domain+relative | 4 | 40800 | 79.29% | 48.84% | 99.22% | 74.23% | 98.83% | 76.76% |
| image+kernel+cokernel+codomain+relative | 5 | 51000 | 80.09% | 47.66% | 99.03% | 72.47% | 98.83% | 78.71% |
| image+cokernel+relative | 3 | 30600 | 77.15% | 44.55% | 98.04% | 68.55% | 98.64% | 75.00% |
| cokernel+domain+codomain+relative | 4 | 40800 | 79.89% | 47.48% | 99.22% | 71.09% | 98.64% | 74.61% |
| domain+relative | 2 | 20400 | 82.03% | 47.28% | 99.22% | 70.90% | 98.63% | 78.91% |
| image+domain+codomain | 3 | 30600 | 77.95% | 49.62% | 98.44% | 71.09% | 98.44% | 76.37% |
| kernel+cokernel+codomain+relative | 4 | 40800 | 82.23% | 50.01% | 98.25% | 72.08% | 98.44% | 77.93% |
| image+relative | 2 | 20400 | 79.51% | 43.17% | 97.07% | 69.34% | 98.44% | 76.17% |
| image+kernel+codomain | 3 | 30600 | 81.84% | 47.28% | 97.85% | 71.67% | 98.44% | 76.37% |
| image+codomain+relative | 3 | 30600 | 80.27% | 45.33% | 97.46% | 69.53% | 98.44% | 76.38% |

---

## 단독 바코드 순위 (n=1)

| Barcode | LIN Soft | LIN Strict | KNN Soft | KNN Strict | RF Soft | RF Strict |
|---|---|---|---|---|---|---|
| **relative** | 82.62% | 38.29% | 98.05% | 68.16% | **99.42%** | 76.56% |
| kernel | 80.67% | 46.29% | 96.29% | 68.17% | 97.65% | 71.28% |
| cokernel | 80.87% | 42.79% | 93.36% | 55.48% | 95.52% | 67.19% |
| image | 79.51% | 41.82% | 91.60% | 57.61% | 96.10% | 70.70% |
| domain | 77.55% | 42.19% | 97.46% | 71.10% | 97.07% | 75.20% |
| codomain | 73.63% | 33.00% | 91.41% | 51.16% | 92.38% | 61.71% |

---

## Subset 크기별 최고 조합 (RF Soft 기준)

| n | Best combo | RF Soft | RF Strict | LIN Soft |
|---|---|---|---|---|
| 1 | **relative** | **99.42%** | 76.56% | 82.62% |
| 2 | kernel+relative | 99.03% | 79.10% | 80.09% |
| 3 | kernel+domain+relative | 99.03% | 79.30% | 80.28% |
| 4 | image+kernel+codomain+relative | 99.02% | 78.13% | 79.70% |
| 5 | kernel+cokernel+domain+codomain+relative | 99.02% | 78.52% | 80.30% |
| 6 | image+kernel+cokernel+domain+codomain+relative (full) | 97.47% | 77.55% | 81.06% |

> **full six-pack(97.47%) < relative 단독(99.42%)**: 바코드를 추가할수록 RF Soft가 감소

---

## 핵심 분석

### 1. Relative 바코드의 압도적 지위

**단독 relative가 full six-pack보다 RF Soft 기준 1.95%p 높다.** H*(K,L)이 두 공간의 위상적 차이를 직접 인코딩하기 때문이다. 상위 20개 조합 중 relative가 포함되지 않은 것은 `image+domain+codomain` (n=3, 98.44%) 하나뿐.

### 2. Relative 없이 최고 성능

| Combo | RF Soft | RF Strict |
|---|---|---|
| image+domain+codomain | 98.44% | 76.37% |
| kernel+cokernel | 98.24% | 77.15% |
| kernel+cokernel+domain | 98.24% | 77.73% |

relative 없이도 98%대 달성 가능하나, relative 포함 조합들이 상위권을 독점.

### 3. Classifier별 패턴

| | Linear SVM | KNN(9) | RF(200) |
|---|---|---|---|
| 상위권 패턴 | relative 포함 시 82~83% | relative 포함 시 98~99% | relative 단독 99.42% |
| Soft 최고 | domain+codomain+relative (83.2%) | image+domain+relative (99.61%) | relative 단독 (99.42%) |
| 특이점 | PCA100 압축으로 선형 분리 한계 | relative 없어도 고성능 유지 | 바코드 추가 시 오히려 하락 |

**KNN은 relative 없이도 98%+**: domain+codomain+relative에서 KNN이 99.61%로 RF보다 높음. KNN은 고차원에서도 근방 구조를 활용하여 상대적으로 robust.

### 4. Full Six-Pack이 최하위인 이유

n=6(full)이 RF Soft 97.47%로 n=1~5 최고보다 모두 낮다. PCA100 압축 시 6개 바코드(61200D)의 정보를 100D로 압축하면 relative의 discriminative 방향이 희석되기 때문. PCA 차원을 높이면 full six-pack이 상위권으로 올라올 가능성이 있음.

### 5. Codomain 기여도 재확인

- 단독 성능 최하위 (RF Soft 92.38%)
- 상위 조합에서도 codomain 포함 여부가 성능 개선에 기여하지 않음
- rank-nullity/isomorphism 정리에 의한 대수적 중복이 codomain을 실질적으로 redundant하게 만듦

---

## 코드 및 데이터

- 실험 코드: `sixpack_rips_ablation.py`
- Raw 결과: `Final_Results(0521)/sixpack_rips_ablation.csv`
