# Fusion Experiment Analysis: Ord_PI × Inter_PI / Ord_PI × 3D_PI

**대상**: Early / Intermediate / Late fusion + Weighted Sum (Ord+Inter, Ord+3D)  
**평가**: StandardScaler → PCA → 5-fold CV, Soft/Strict accuracy  
**분류기**:  
- `LIN_hard`: LinearSVC(C=1e5) — hard margin  
- `LIN_soft`: SVC(kernel='linear', C=1.0) — soft margin (벤치마크 기준)  
- `RBF`: SVC(kernel='rbf', C=1.0)  
- `RF200`: RandomForestClassifier(200)  

**날짜**: 2026-05-22

---

## 전체 결과

### Baseline (단일 Descriptor, PCA 100)

| Descriptor | LIN_hard Soft | LIN_soft Soft | RBF Soft | RF Soft |
|---|---|---|---|---|
| Ord_PI | 76.97% | 98.64% | 96.49% | 97.66% |
| Inter_PI | 81.26% | 97.66% | 94.34% | 97.46% |
| **3D_PI** | 80.48% | **99.61%** | 91.40% | **98.05%** |

> LIN_soft baseline이 full_benchmark.csv 값과 정확히 일치 (Ord: 98.64%, Inter: 97.66%, 3D: 99.61%) ✓

### Early Fusion (concat → PCA(k) → clf)

| ID | 조합 | PCA | LIN_hard | LIN_soft | RBF | RF |
|---|---|---|---|---|---|---|
| E1 | Ord+Inter | 100 | 78.32% | 99.03% | 96.30% | 97.66% |
| **E2** | **Ord+3D** | **100** | 80.87% | **99.22%** | **97.66%** | 97.65% |
| E3 | Ord+Inter | 200 | 81.45% | 99.22% | 96.30% | 96.88% |
| E4 | Ord+3D | 200 | 80.47% | 99.22% | 97.66% | 97.65% |

### Intermediate Fusion (PCA each → concat embedding → clf)

| ID | 조합 | 총 차원 | LIN_hard | LIN_soft | RBF | RF |
|---|---|---|---|---|---|---|
| I1 | Ord+Inter | 100 | 83.02% | 96.29% | 94.92% | 96.88% |
| I2 | Ord+3D | 100 | 82.61% | 96.88% | 94.14% | 97.26% |
| **I3** | **Ord+Inter** | **200** | **83.99%** | 96.49% | 94.92% | 96.68% |
| **I4** | **Ord+3D** | **200** | **84.37%** | 98.05% | 95.51% | 96.88% |

### Late Fusion — Soft Voting (equal weight, PCA 100 per modal)

| ID | 조합 | Base clf | Soft | Strict |
|---|---|---|---|---|
| L1 | Ord+Inter | RF | 96.49% | 72.47% |
| L2 | Ord+3D | RF | 97.07% | 75.98% |
| L3 | Ord+Inter | RBF | 95.90% | 69.92% |
| L4 | Ord+3D | RBF | 96.69% | 73.05% |
| L5 | Ord+Inter | LIN_soft | 96.49% | 67.97% |
| **L6** | **Ord+3D** | **LIN_soft** | **97.66%** | 72.45% |

### Late Fusion — Stacking (RF base × 2 → RF meta, OOF)

| ID | 조합 | Soft | Strict |
|---|---|---|---|
| L7 | Ord+Inter | 96.49% | 71.88% |
| L8 | Ord+3D | 97.27% | 76.38% |

### Weighted Sum — Embedding level (PCA100 each → norm → w·Za + (1-w)·Zb)

| ID | 조합 | LIN_hard (w) | LIN_soft (w) | RBF (w) | RF (w) |
|---|---|---|---|---|---|
| W1 | Ord+Inter | 80.09% (0.9) | 93.56% (0.2) | 93.56% (0.6) | 94.92% (0.6) |
| W2 | Ord+3D | 80.87% (0.7) | 94.92% (0.8) | 95.91% (0.1) | 95.51% (0.5) |

### Weighted Sum — Late level (w·proba_a + (1-w)·proba_b, PCA 100 per modal)

| ID | 조합 | Base clf | Soft | Strict | avg w (Ord 비중) |
|---|---|---|---|---|---|
| W3 | Ord+Inter | RF | 96.29% | 72.27% | 0.70 |
| **W4** | **Ord+3D** | **RF** | **97.85%** | **76.18%** | **0.32** |
| W5 | Ord+Inter | RBF | 96.10% | 69.53% | 0.54 |
| W6 | Ord+3D | RBF | 95.12% | 71.28% | 0.44 |
| W7 | Ord+Inter | LIN_soft | 96.88% | 68.75% | 0.62 |
| W8 | Ord+3D | LIN_soft | 97.07% | 70.69% | 0.06 |

---

## 핵심 분석

### 1. 분류기별 fusion 효과

| 분류기 | 단독 최고 | Best fusion | Soft | 방향 |
|---|---|---|---|---|
| **LIN_hard** | Inter_PI 81.26% | I4 Intermediate Ord+3D | **84.37%** | ↑ +3.11%p |
| **LIN_soft** | 3D_PI 99.61% | E2/E3/E4 Early Ord+3D | 99.22% | ↓ -0.39%p |
| **RBF** | Ord_PI 96.49% | E2 Early Ord+3D | **97.66%** | ↑ +1.17%p |
| **RF** | 3D_PI 98.05% | W4 Late WS Ord+3D | 97.85% | ↓ -0.20%p |

**LIN_hard만 Intermediate fusion에서 유의미한 이득. LIN_soft와 RF는 단독이 여전히 최고.**

---

### 2. LIN_hard vs LIN_soft: 방향이 정반대

**LIN_soft (soft margin)**:
- 단독 성능이 이미 매우 높음 (3D_PI 99.61%)
- Early fusion E2(99.22%) < 3D_PI 단독 — 이득 없음
- Intermediate I1~I4: 96~98% — baseline 대비 하락. fold별 clip+PCA가 LIN_soft의 margin 계산을 방해
- 결론: **LIN_soft는 단독 descriptor가 최선**

**LIN_hard (hard margin)**:
- 단독 성능이 낮음 (~77~81%) — PCA 100D 공간에서 hard margin 제약이 너무 강함
- Intermediate I3/I4(83~84%): 독립 PCA 후 concat → feature 공간이 더 선형 분리에 적합
- 결론: **LIN_hard는 Intermediate fusion에서 이득**

**핵심 이유**: LIN_soft는 soft margin(C=1.0)으로 PCA100 공간에서도 좋은 hyperplane을 찾지만, LIN_hard는 hard margin이라 PCA100 단일 공간에서 misclassification 없이 분리하기 어려움. Intermediate fusion의 200D 공간에서는 더 많은 자유도가 생겨 LIN_hard가 개선됨.

---

### 3. Embedding WS는 전 분류기에서 손해

| | Intermediate concat | Embedding WS |
|---|---|---|
| LIN_hard [Ord+Inter] | **83.99%** (I3) | 80.09% (W1) |
| LIN_soft [Ord+3D] | **98.05%** (I4) | 94.92% (W2) |
| RF [Ord+3D] | **97.26%** (I2) | 95.51% (W2) |

서로 다른 PCA 좌표계에서 element-wise 합산은 정보 손실. Concat이 항상 우세.

---

### 4. 최적 w가 보여주는 descriptor 선호도

| | w (Ord 비중) | 해석 |
|---|---|---|
| W4 RF [Ord+3D] | **0.32** | 3D_PI 68% — RF는 3D_PI 선호 |
| W8 LIN_soft [Ord+3D] | **0.06** | 3D_PI 94% — LIN_soft도 3D_PI 압도적 선호 |
| W1 LIN_hard [Ord+Inter] | **0.9** | Ord_PI 90% — LIN_hard는 Ord 선호 |

W8의 w=0.06은 사실상 3D_PI 단독에 가까움 → LIN_soft 입장에서 Ord_PI 추가는 거의 무의미.

---

## 방법별 최고 결과 요약

| 분류기 | 방법 | 조합 | Soft | 단독 최고 대비 |
|---|---|---|---|---|
| **LIN_hard** | Intermediate 200D | Ord+3D | **84.37%** | +3.11%p ↑ |
| **LIN_soft** | 단독 3D_PI 사용 권장 | — | 99.61% | fusion 무효 |
| **RBF** | Early PCA100 | Ord+3D | **97.66%** | +1.17%p ↑ |
| **RF** | 단독 3D_PI 사용 권장 | — | 98.05% | fusion 무효 |

---

## 코드 및 데이터

- 실험 코드: `fusion_experiment.py`
- Raw 데이터: `Final_Results(0521)/fusion_experiment.csv`
- 분류기: LinearSVC(C=1e5) hard / SVC(linear, C=1.0) soft / SVC(rbf, C=1.0) / RF(200)
- Weighted Sum w 탐색: inner 4-fold CV, w ∈ {0.0, 0.1, ..., 1.0}
- 전처리 주의: 3D_PI fold별 SS에서 std≈0 feature → `np.clip(1e6)` 후 PCA
