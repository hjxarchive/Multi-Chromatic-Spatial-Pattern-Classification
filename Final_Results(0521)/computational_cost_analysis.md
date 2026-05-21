# Computational Cost Analysis

**대상**: Ord_PI / Inter_PI / 3D_PI / Sixpack_Rips / Sixpack_Chroma  
**측정**: wall time (sec) + peak memory (MB) per sample  
**입력**: `Data/ParamSweep_input/` 중 5개 샘플링 (N≈200 points, |A|≈80~87, |B|≈113~121)  
**환경**: Apple Silicon (macOS), single process, `.venv/bin/python`  
**날짜**: 2026-05-21  
**파이프라인**: `Vectorization/` 노트북 코드 그대로 사용 (notebook-accurate)

---

## 실험 설계

```
ParamSweep_N_Output/
  Pos_*.dat   → 2D 좌표
  Types_*.dat → type 1(A) / type 2(B)

각 Descriptor별 전체 파이프라인 측정:
  data load → complex 구축 → PH 계산 → PI 벡터화

측정 도구:
  wall time : time.perf_counter()
  peak mem  : tracemalloc (Python heap peak, MB)
```

### Descriptor 파이프라인

| Descriptor | 파이프라인 | 벡터 차원 |
|---|---|---|
| Ord_PI | Rips(A∪B) → GUDHI PH → PI | 10,200D |
| Inter_PI | Mixup barcode Wagner'24 (pure Python) → Interaction PI | 10,200D |
| 3D_PI | Mixup barcode Wagner'24 (pure Python) → 3D KDE | 16,800D |
| Sixpack_Rips | L=Rips(A)⊆K=Rips(A∪B), _reduce_with_V 5-step → PI | 61,200D |
| Sixpack_Chroma | chromatic_tda Alpha complex → 6-pack → PI | 61,200D |

---

## Per-Sample 결과

| Sample | \|A\| | \|B\| | Ord_PI | Inter_PI | 3D_PI | Sixpack_Rips | Sixpack_Chroma |
|---|---|---|---|---|---|---|---|
| ParamSweep_141 | 79 | 121 | 0.34s / 0.44MB | 83.2s / 2248MB | 82.4s / 2248MB | 374.1s / 10356MB | 4.53s / 10.87MB |
| ParamSweep_2   | 80 | 120 | 0.12s / 0.40MB | 19.9s / 626MB  | 20.4s / 626MB  | 67.2s / 2553MB   | 4.77s / 12.08MB |
| ParamSweep_302 | 84 | 116 | 0.45s / 0.42MB | 149.1s / 4069MB | 153.8s / 4069MB | 974.1s / 25905MB | 4.79s / 11.89MB |
| ParamSweep_400 | 84 | 116 | 0.18s / 0.41MB | 34.2s / 1014MB | 33.8s / 1014MB | 121.0s / 4181MB  | 4.68s / 11.08MB |
| ParamSweep_455 | 87 | 113 | 0.23s / 0.41MB | 48.2s / 1372MB | 48.2s / 1372MB | 166.4s / 6555MB  | 4.10s / 10.31MB |

---

## 집계 결과 (N=5)

### Wall Time

| Descriptor | Mean | Std | Min | Max |
|---|---|---|---|---|
| **Ord_PI** | **0.26s** | 0.12s | 0.12s | 0.45s |
| Inter_PI | 66.93s | 46.17s | 19.9s | 149.1s |
| 3D_PI | 67.71s | 47.75s | 20.4s | 153.8s |
| **Sixpack_Rips** | **340.53s** | 333.40s | 67.2s | 974.1s |
| **Sixpack_Chroma** | **4.57s** | 0.25s | 4.10s | 4.79s |

### Peak Memory

| Descriptor | Mean | Std | Min | Max |
|---|---|---|---|---|
| **Ord_PI** | **0.41 MB** | 0.01 MB | 0.40 MB | 0.44 MB |
| Inter_PI | 1,865.8 MB | 1,225.4 MB | 626 MB | 4,069 MB |
| 3D_PI | 1,865.8 MB | 1,225.4 MB | 626 MB | 4,069 MB |
| **Sixpack_Rips** | **9,910.1 MB** | 8,417.3 MB | 2,553 MB | 25,905 MB |
| **Sixpack_Chroma** | **11.25 MB** | 0.66 MB | 10.3 MB | 12.1 MB |

### Ord_PI 대비 배율

| Descriptor | Time 배율 | Memory 배율 |
|---|---|---|
| Ord_PI | ×1.0 | ×1.0 |
| Inter_PI | ×253.8 | ×4,517 |
| 3D_PI | ×256.7 | ×4,517 |
| **Sixpack_Rips** | **×1,291** | **×23,991** |
| **Sixpack_Chroma** | **×17.3** | **×27.2** |

---

## 분석

### 1. Ord_PI vs 나머지: 파이프라인 구조가 핵심

Ord_PI는 GUDHI의 C++ native persistence 계산(`RipsComplex → SimplexTree.persistence()`)을 그대로 사용. 나머지 Descriptor 중 Inter_PI / 3D_PI / Sixpack_Rips는 **pure Python column reduction** (boundary matrix reduction)을 구현. Python column reduction의 복잡도는 최악 O(m³) (m = simplex 수).

### 2. Sixpack_Chroma: 비용 효율 최상

`chromatic_tda`는 C++/Cython 기반 Alpha complex를 사용하므로 Sixpack_Rips보다 **74.5배 빠르고** 메모리는 **881배 절약**. 동일한 61,200D 벡터를 생성하면서도 평균 4.57s / 11.25 MB — Ord_PI 대비 ×17 수준.

tracemalloc은 Python heap만 측정하므로 chromatic_tda의 C++ 내부 메모리는 포함되지 않음. 실제 메모리는 다소 높을 수 있으나 scale 자체가 다름.

### 3. Sixpack_Rips: 가장 비싼 Descriptor

5-step `_reduce_with_V` 알고리즘은 V(transformation matrix)를 추적하므로 단순 column reduction 대비 메모리가 약 2배 이상 증가. 최악 샘플(ParamSweep_302)에서 974초 / 25.9 GB — 동일 N=200이지만 위상 구조에 따라 배율이 크게 달라짐 (ParamSweep_2 대비 ~14.5배 차이).

### 4. Inter_PI ≈ 3D_PI (메모리·시간 동일)

두 Descriptor 모두 **Mixup barcode 계산을 공유**하며, 이 단계가 전체 비용의 대부분. PI 벡터화 단계(Interaction PI vs 3D KDE)의 차이는 무시 가능. 동시 필요 시 Mixup barcode 1회 계산 후 공유하면 비용 절반.

### 5. 분산이 매우 큰 이유

시간과 메모리 모두 std/mean > 0.9 — 위상 복잡도(matchable barcode 수)가 입력 포인트 배열에 크게 의존. 동일 N=200이지만 ParamSweep_302(974s / 25.9GB) vs ParamSweep_2(67s / 2.6GB)는 ~14배 차이.

---

## Descriptor 선택 가이드

| 상황 | 권장 |
|---|---|
| 빠른 프로토타입 / 대규모 sweep | Ord_PI (0.26s / 0.41MB) |
| 성능 + 비용 균형 | Sixpack_Chroma (4.57s / 11MB, 61200D) |
| 분류 성능 최우선 (비용 감수) | Sixpack_Rips (340s / 9.9GB, 61200D) |
| Inter_PI + 3D_PI 동시 필요 | Mixup barcode 1회 계산 후 공유 |
| 전체 512 샘플 Sixpack_Rips 재계산 | 512 × 340s → 병렬화 필수 (단일 프로세스 불가) |

---

## 코드 및 데이터

- 실험 코드: `computational_cost.py`  
- Raw 데이터: `Final_Results(0521)/computational_cost_raw.csv`  
- 파이프라인 기준: `Vectorization/` 노트북 (Ordinary PI, Six-pack (Rips), Mixup barcode (1), Six-pack (chromatic_tda) (1))
