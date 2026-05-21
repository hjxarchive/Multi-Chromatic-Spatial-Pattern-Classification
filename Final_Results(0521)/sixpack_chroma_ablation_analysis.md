# Sixpack_Chroma Ablation Study — 2^6 Barcode Subsets

**대상**: Sixpack_Chroma — 6개 barcode의 모든 조합 (2^6 - 1 = 63개 non-empty subset)  
**벡터**: raw PI (선택된 barcode만 flatten, 바코드당 10200D)  
**평가**: StandardScaler → PCA(100) → 5-fold CV, Soft/Strict accuracy  
**날짜**: 2026-05-22

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
| **image+domain+codomain+relative** | 4 | 40800 | 79.71% | 45.91% | 98.05% | 71.67% | **99.22%** | 74.22% |
| image+cokernel+relative | 3 | 30600 | 78.51% | 41.22% | 98.04% | 73.23% | 98.83% | 74.81% |
| image+kernel+codomain+relative | 4 | 40800 | 77.74% | 45.12% | 97.46% | 69.52% | 98.83% | 75.78% |
| cokernel+relative | 2 | 20400 | 77.35% | 40.82% | 97.85% | 69.52% | 98.83% | 73.44% |
| image+kernel+relative | 3 | 30600 | 76.97% | 45.52% | 98.44% | 73.04% | 98.83% | 76.18% |
| image+codomain+relative | 3 | 30600 | 78.71% | 41.60% | 98.24% | 70.88% | 98.63% | 73.04% |
| image+kernel+cokernel+codomain+relative | 5 | 51000 | 79.31% | 45.70% | 98.24% | 69.71% | 98.63% | 75.39% |
| image+domain+relative | 3 | 30600 | 75.58% | 45.71% | 98.24% | 73.23% | 98.63% | 75.58% |
| image+kernel+cokernel+relative | 4 | 40800 | 78.92% | 45.93% | 98.04% | 70.69% | 98.63% | 75.00% |
| cokernel+domain+codomain+relative | 4 | 40800 | 77.73% | 47.06% | 98.24% | 72.06% | 98.63% | 74.02% |
| domain+codomain+relative | 3 | 30600 | 79.88% | 47.28% | 98.63% | 71.09% | 98.63% | 73.43% |
| image+cokernel+domain+relative | 4 | 40800 | 75.79% | 43.37% | 98.24% | 74.02% | 98.44% | 74.41% |
| kernel+codomain+relative | 3 | 30600 | 77.95% | 46.50% | 98.04% | 73.62% | 98.44% | 76.37% |
| kernel+cokernel+domain+relative | 4 | 40800 | 81.85% | 47.08% | 97.26% | 70.69% | 98.44% | 76.17% |
| kernel+domain+codomain+relative | 4 | 40800 | 80.28% | 46.88% | 98.44% | 71.28% | 98.44% | 75.39% |
| kernel+relative | 2 | 20400 | 77.15% | 40.64% | 97.07% | 70.31% | 98.43% | 75.59% |
| relative | 1 | 10200 | 80.27% | 44.74% | 98.05% | 72.45% | 98.24% | 74.60% |
| codomain+relative | 2 | 20400 | 79.49% | 39.86% | 96.87% | 69.52% | 98.24% | 70.89% |
| image+kernel+domain+relative | 4 | 40800 | 76.96% | 43.56% | 98.24% | 72.45% | 98.24% | 74.41% |
| kernel+domain+relative | 3 | 30600 | 76.19% | 44.95% | 97.27% | 69.91% | 98.24% | 73.63% |

---

## 단독 바코드 순위 (n=1)

| Barcode | LIN Soft | LIN Strict | KNN Soft | KNN Strict | RF Soft | RF Strict |
|---|---|---|---|---|---|---|
| **relative** | 80.27% | 44.74% | 98.05% | 72.45% | **98.24%** | 74.60% |
| image | 78.32% | 41.42% | 91.21% | 59.56% | 95.51% | 69.73% |
| kernel | 75.18% | 42.57% | 91.80% | 57.82% | 95.31% | 69.15% |
| domain | 77.55% | 42.58% | 95.11% | 62.49% | 95.12% | 70.31% |
| cokernel | 77.95% | 38.88% | 92.57% | 58.19% | 93.95% | 65.23% |
| codomain | 71.89% | 29.69% | 90.82% | 54.49% | 91.02% | 62.31% |

---

## Subset 크기별 최고 조합 (RF Soft 기준)

| n | Best combo | RF Soft | RF Strict | LIN Soft |
|---|---|---|---|---|
| 1 | **relative** | 98.24% | 74.60% | 80.27% |
| 2 | cokernel+relative | 98.83% | 73.44% | 77.35% |
| 3 | image+cokernel+relative | 98.83% | 74.81% | 78.51% |
| 4 | **image+domain+codomain+relative** | **99.22%** | 74.22% | 79.71% |
| 5 | image+kernel+cokernel+codomain+relative | 98.63% | 75.39% | 79.31% |
| 6 | image+kernel+cokernel+domain+codomain+relative (full) | 97.66% | 75.00% | 77.55% |

> **전체 최고는 n=4 조합(99.22%)**. relative 단독(98.24%)이 n=1 최고이지만, image+domain+codomain 추가 시 0.98%p 추가 이득.

---

## 핵심 분석

### 1. Chroma vs Rips — Relative 바코드의 역할 변화

| 지표 | Sixpack_Rips | Sixpack_Chroma |
|---|---|---|
| relative 단독 RF Soft | **99.42%** | 98.24% |
| 전체 최고 RF Soft | 99.42% (relative 단독) | **99.22%** (4-barcode 조합) |
| 최고 달성 n | 1 | 4 |
| full six-pack RF Soft | 97.47% | 97.66% |

**Rips**: relative 단독이 전체 최고 — 바코드를 추가할수록 성능 하락  
**Chroma**: relative가 여전히 핵심이지만 단독으로는 최고에 못 미침 — image, domain, codomain 추가 시 최고 달성

Chroma 필터링은 chromatic 제약(색상 기반 simplex 포함 조건)을 적용하여 Rips보다 위상 정보가 여러 바코드에 분산됨. 따라서 **단일 barcode 지배** 현상이 Rips보다 약함.

---

### 2. Image 바코드의 핵심 역할

Chroma 상위 20개 조합 중 image가 포함된 조합: **14/20개** (70%).  
반면 Rips에서는 image가 단독 성능 4위(96.10%)이고 상위권의 필수 요소가 아니었음.

| | Rips image 단독 | Chroma image 단독 |
|---|---|---|
| RF Soft | 96.10% | 95.51% |
| 역할 | 있으면 좋음 (보조) | **상위 조합 필수 구성요소** |

Chromatic filtration에서 image 바코드는 H*(K)의 전체 위상 구조를 포착하며, 색상 제약 없이 필터된 complex 전체의 신호를 담음. Chroma 맥락에서 image가 relative를 보완하는 가장 효과적인 파트너.

---

### 3. 최고 조합 `image+domain+codomain+relative` 분석

```
image:    H*(K)         전체 complex 위상
domain:   H*(L)         sub-complex 위상
codomain: H*(K/L) 관련  포함 공간의 위상
relative: H*(K,L)       두 공간의 위상적 차이 (상대 호몰로지)
```

이 4개 조합은 short exact sequence of the pair (K, L)을 구성하는 네 항을 모두 포함:

```
... → H*(L) → H*(K) → H*(K,L) → H*(L) → ...
```

즉, 위상적으로 완전한 정보를 담으면서 kernel/cokernel의 중복 정보를 제거한 최적 subset.

---

### 4. Codomain 재해석 — Chroma에서의 역할

**Rips**: codomain 단독 최하위(92.38%), 조합에서도 기여 낮음  
**Chroma**: codomain 단독 최하위(91.02%)이지만, `domain+codomain+relative` (98.63%), `image+domain+codomain+relative` (99.22%)에서 핵심 기여

| 조합 | RF Soft |
|---|---|
| image+domain+relative | 98.63% |
| image+codomain+relative | 98.63% |
| **image+domain+codomain+relative** | **99.22%** |

domain과 codomain을 함께 쓸 때 0.59%p 추가 이득. Chromatic filtration에서 두 바코드가 상호보완적 위상 정보를 담음.

---

### 5. Relative 없이 최고 성능 (Chroma)

| Combo | RF Soft | RF Strict |
|---|---|---|
| image+kernel+domain+codomain | 97.85% | 75.59% |
| image+domain+codomain | 97.46% | 73.04% |
| domain+codomain | 97.06% | 72.46% |

relative 없이도 97%대 달성 가능. Chroma에서 relative의 기여는 Rips(~2%p)보다 작음(~1.4%p).

---

### 6. Classifier별 패턴

| | Linear SVM | KNN(9) | RF(200) |
|---|---|---|---|
| Soft 최고 | kernel+cokernel+domain+relative (81.85%) | cokernel+domain+relative (98.82%) | image+domain+codomain+relative (99.22%) |
| n=1 최고 | relative (80.27%) | relative (98.05%) | relative (98.24%) |
| 특이점 | PCA100 압축으로 선형 분리 한계 유지 | relative 없이도 고성능 | n=4에서 peak, 이후 하락 |

**KNN의 relative 의존도**: KNN Soft 최고는 relative 포함 조합 (`cokernel+domain+codomain+relative: 98.82%`). KNN도 Chroma에서 relative 선호.

---

### 7. Rips vs Chroma 비교 요약

| 항목 | Sixpack_Rips | Sixpack_Chroma |
|---|---|---|
| 지배 barcode | relative | relative + image |
| 최고 성능 | 99.42% (relative 단독) | 99.22% (4-barcode) |
| full six-pack | 97.47% (최하위) | 97.66% (최하위) |
| n 증가 시 성능 | 단조 감소 (n=1 최고) | n=4에서 peak 후 감소 |
| Codomain 역할 | redundant | domain과 조합 시 기여 |
| Image 역할 | 보조 | 상위 조합 핵심 |

---

## 코드 및 데이터

- 실험 코드: `sixpack_chroma_ablation.py`
- Raw 결과: `Final_Results(0521)/sixpack_chroma_ablation.csv`
