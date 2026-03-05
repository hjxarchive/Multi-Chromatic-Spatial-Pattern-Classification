# 연구 데이터 파이프라인

## 전체 흐름 개요

```mermaid
flowchart TD
    classDef data fill:#fce4ec,stroke:#c62828,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef vector fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef eval fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    SIM[("🔬 시뮬레이션 데이터\n512 samples\nPos_*.dat / Types_*.dat")]:::data

    SIM --> SPLIT["Type 분리\nA = Type 1 (Red)\nB = Type 2 (Green)"]:::process

    SPLIT --> OP_PIPE
    SPLIT --> TDA_PIPE
    SPLIT --> SIXPACK_PIPE

    subgraph OP_PIPE ["📐 Order Parameter Pipeline"]
        direction TB
        ANG["Angular Distribution\nΘ, Θ_B, Θ_O"]:::process
        RAD["Radial Distribution\nR, R_B, R_O"]:::process
        ANG --> OP_VEC["OP Vector"]:::vector
        RAD --> OP_VEC
    end

    subgraph TDA_PIPE ["🔵 Ordinary TDA Pipeline"]
        direction TB
        VR["Vietoris-Rips Filtration\n(Green / Red / All)"]:::process
        PH["Persistent Homology\nH0, H1, H2"]:::process
        PI["Persistence Image\nσ=0.05, res=0.1"]:::process
        VR --> PH --> PI
        PI --> TDA_VEC["Ord PI Vector\nH0: 100D, H1: 5000D"]:::vector
    end

    subgraph SIXPACK_PIPE ["📦 Six-pack Pipeline"]
        direction TB

        subgraph FILT ["Filtration 선택"]
            RIPS_F["Rips Complex"]:::process
            CECH_F["Čech Complex"]:::process
            CHROMA_F["Chromatic Alpha\nComplex"]:::process
        end

        BARCODE["Six-pack Barcode\n(image / kernel / cokernel\n+ complex / sub / relative)"]:::process
        FILT --> BARCODE

        SP_PI["PI 벡터화\n양방향: A→B, B→A"]:::process
        BARCODE --> SP_PI

        SP_PI --> SP_VEC["Six-pack Vector\n~61,200D (Rips)"]:::vector
    end

    OP_VEC --> EVAL
    TDA_VEC --> EVAL
    SP_VEC --> EVAL

    subgraph EVAL ["📊 평가 파이프라인"]
        direction TB
        SCALE["StandardScaler"]:::process
        PCA_R["PCA 차원 축소\n20D / 50D / 100D"]:::process
        CLF["분류기 앙상블\nSVM (RBF/Linear)\nKNN / Random Forest"]:::eval
        CV["5-fold Stratified CV"]:::eval
        METRICS["Soft Acc / Strict Acc / F1"]:::eval
        SCALE --> PCA_R --> CLF --> CV --> METRICS
    end

    METRICS --> RESULT[("📈 최종 결과\n방법별 정확도 비교")]:::data
```

---

## 파이프라인 상세

### 1️⃣ 데이터 입력

```mermaid
flowchart LR
    classDef file fill:#fff9c4,stroke:#f9a825

    P["ParamSweep_{1..512}_Output/"]:::file
    P --> POS["Pos_RR_RG_GG.dat\n(x, y 좌표)"]:::file
    P --> TYP["Types_RR_RG_GG.dat\n(1 or 2)"]:::file

    POS & TYP --> LOAD["np.loadtxt\n→ A (Type1), B (Type2)"]
```

| 항목 | 값 |
|------|-----|
| 총 샘플 수 | 512개 (8×8×8 파라미터 조합) |
| 파라미터 | RR, RG, GG ∈ {0.0, 0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25} |
| 클래스 수 | 12개 (Phase 0–13, 일부 미사용) |
| 저장 위치 | Google Drive `/URP/ParamSweep_*_Output/` |

---

### 2️⃣ Six-pack Barcode 계산 (핵심)

```mermaid
flowchart TD
    classDef bar fill:#e8eaf6,stroke:#283593

    A["Point Cloud A"] & B["Point Cloud B"]
    AB["A ∪ B"]

    A & B --> AB

    subgraph INCLUSION ["Inclusion L ⊆ K"]
        L["L = Filtration(A)"]
        K["K = Filtration(A ∪ B)"]
        L -->|"⊆"| K
    end

    A --> L
    AB --> K

    INCLUSION --> MAT["Boundary Matrix\nReduction (Z/2)"]

    MAT --> IMG["Image Barcode\n(L에서 태어나 K에서도 생존)"]:::bar
    MAT --> KER["Kernel Barcode\n(L에서 죽지만 K에서 생존)"]:::bar
    MAT --> COK["Cokernel Barcode\n(K\\L에서 태어남)"]:::bar

    A --> ORD_A["Ordinary(A)\n= sub_complex"]:::bar
    AB --> ORD_AB["Ordinary(A∪B)\n= complex"]:::bar
    B --> ORD_B["Ordinary(B)\n= relative"]:::bar

    IMG & KER & COK & ORD_A & ORD_AB & ORD_B --> SIX["Complete Six-pack\n(6 barcode types × 2 directions)"]
```

#### Filtration 방식별 비교

| Filtration | 노트북 | `max_radius` | 성능 (Soft Acc) | 비고 |
|-----------|--------|-------------|----------------|------|
| **Vietoris-Rips** | `Sixpack_Rips.ipynb` | 10 | **~93%** ⭐ | |
| **Čech** | `Phase5_Sixpack_Cech.ipynb` | 5 | 실험 예정 | Drel로 H*(K,L) 직접 계산 |
| **Chromatic Alpha** | `Six-pack (chromatic_tda).ipynb` | 10 | ~77% | |

---

### 3️⃣ PI 벡터화

```mermaid
flowchart LR
    classDef pi fill:#fce4ec,stroke:#880e4f

    BD["Barcode\n{(b₁,d₁), (b₂,d₂), ...}"]
    TR["Transform\n(b,d) → (b, d-b)"]
    WT["Weight\nw = persistence"]
    GK["Gaussian Kernel\nσ = 0.05"]
    GRID["Pixel Grid\nres = 0.1"]

    BD --> TR --> WT --> GK --> GRID

    GRID --> H0["H0 PI\nbirth∈[0,1]\npers∈[0,10]\n→ mean(axis=0)\n→ 100D"]:::pi

    GRID --> H1["H1 PI\nbirth∈[0,10]\npers∈[0,5]\nskew=True\n→ flatten\n→ 5000D"]:::pi

    H0 & H1 --> CAT["Concatenate\n→ 5100D / barcode type"]
```

---

### 4️⃣ 평가 파이프라인

```mermaid
flowchart TD
    classDef best fill:#c8e6c9,stroke:#1b5e20

    X["Feature Vector X\n(512 × D)"]
    Y["Labels y\n(512,) — 12 classes"]

    X --> SS["StandardScaler\nzero mean, unit var"]
    SS --> PCA["PCA\n→ 20D / 50D / Full"]

    PCA --> FOLD["StratifiedKFold\nk=5, shuffle=True"]

    FOLD --> C1["SVM RBF\nC=0.5, 1.0, 2.0"]
    FOLD --> C2["SVM Linear\nC=1.0"]
    FOLD --> C3["KNN\nk=3, 12"]
    FOLD --> C4["Random Forest\nn=100"]

    C1 & C2 & C3 & C4 --> PRED["y_pred"]

    PRED --> SA["Soft Accuracy\n인접 phase 허용"]:::best
    PRED --> HA["Strict Accuracy\n정확 일치만"]
    PRED --> F1["Weighted F1"]

    SA & HA & F1 --> REPORT["Mean ± Std\n(5-fold)"]
```

#### Adjacent Phase 관계

```mermaid
graph LR
    0 --- 1 & 2
    1 --- 3 & 4
    2 --- 3 & 4
    3 --- 4
    4 --- 5 & 8 & 11
    5 --- 6 & 7 & 11
    6 --- 7 & 9 & 10
    7 --- 9 & 10
    8 --- 9 & 10
```

> Soft Accuracy에서는 위 그래프에서 **간선으로 연결된 phase 간 오분류를 정답으로 간주**합니다.

---

### 5️⃣ 파일 저장 구조

```
Google Drive/URP/
├── ParamSweep_{1..512}_Output/     ← 원본 시뮬레이션
│   ├── Pos_RR_RG_GG.dat
│   └── Types_RR_RG_GG.dat
│
└── 1224_Vectors/                   ← 벡터화 결과
    ├── Sixpack_Rips/
    │   └── Sixpack_Rips_{1..512}.npz
    ├── Sixpack_Chroma*/
    │   └── Sixpack_Chroma_{1..512}.npz
    └── Sixpack_Cech/               ← Phase 5 (신규)
        └── Sixpack_Cech_{1..512}.npz
```

각 `.npz` 파일 구조:
- `arr_0` = `PI_A_to_B` (dict: barcode_type → {0: H0_vec, 1: H1_vec})
- `arr_1` = `PI_B_to_A` (dict: 동일 구조)

---

## 6️⃣ 7가지 Descriptor Vector 상세

```mermaid
flowchart LR
    classDef ord fill:#e3f2fd,stroke:#1565c0
    classDef mix fill:#fce4ec,stroke:#c62828
    classDef six fill:#e8f5e9,stroke:#2e7d32
    classDef combo fill:#fff3e0,stroke:#e65100

    subgraph ORDINARY ["Ordinary PH 기반"]
        ORD["① Ord_PI"]:::ord
    end

    subgraph MIXUP ["Mixup Barcode 기반"]
        INTER["② Inter_PI"]:::mix
        THREED["③ 3D_PI"]:::mix
    end

    subgraph SIXPACK ["Six-pack 기반"]
        RIPS["④ Sixpack_Rips"]:::six
        CHROMA["⑤ Sixpack_Chroma"]:::six
    end

    subgraph COMBINED ["결합 벡터"]
        IO["⑥ Inter+Ord"]:::combo
        DO["⑦ 3D+Ord"]:::combo
    end
```

---

### ① Ord_PI (Ordinary Persistence Image)

**기본적인 TDA 벡터.** 각 sub-population에 대해 독립적으로 VR filtration → PH → PI 수행.

```mermaid
flowchart LR
    A["Red cells"] --> VR_A["VR(Red)"] --> PH_A["H0, H1"] --> PI_A["PI(Red)"]
    B["Green cells"] --> VR_B["VR(Green)"] --> PH_B["H0, H1"] --> PI_B["PI(Green)"]
    C["All cells"] --> VR_C["VR(All)"] --> PH_C["H0, H1"] --> PI_C["PI(All)"]
    PI_A & PI_B & PI_C --> CAT["Concat → Ord_PI"]
```

| 항목 | 값 |
|------|-----|
| Filtration | Vietoris-Rips |
| Sub-populations | Red / Green / All (3개) |
| Homology 차원 | H0, H1 |
| 벡터 크기 | 3 × (H0: 100D + H1: 5000D) = **15,300D** |
| PI 파라미터 | σ=0.05, res=0.1, weight=persistence |
| 성능 | 중상위 |

---

### ② Inter_PI (Interaction Persistence Image)

**Mixup barcode에서 추출한 interaction 정보.** A와 B 사이의 상호작용을 topology로 포착.

```mermaid
flowchart LR
    A["Red"] & B["Green"]
    A & B --> MIX["Mixup Barcode\n(Wagner et al. 2024)"]
    MIX --> TRIPLE["Triple (b_im, d_im, d_dom)"]
    TRIPLE --> PROJ["2D Projection\n(d_dom, d_im)"]
    PROJ --> PI_I["PI with weight = mixup\n→ Inter_PI"]
```

| 항목 | 값 |
|------|-----|
| Barcode 원천 | Mixup barcode (canonical matching) |
| 좌표 | $(d_{dom}, d_{im})$ — domain death × image death |
| Weight function | **mixup** (interaction 강도 반영) |
| 성능 | Ord_PI 대비 **+1~2%** 향상 |
| 의의 | A, B 간 상호작용 정보를 직접 포착 |

---

### ③ 3D_PI (3D Persistence Image)

**Mixup barcode의 전체 triple을 3D PI로 벡터화.** 더 풍부한 정보를 담지만 차원이 높음.

```mermaid
flowchart LR
    MIX["Mixup Barcode"]
    MIX --> TRIPLE["Triple (b_im, d_im, d_dom)"]
    TRIPLE --> PI3D["3D Persistence Image\nweight = 1"]
    PI3D --> VEC["3D_PI Vector"]
```

| 항목 | 값 |
|------|-----|
| Barcode 원천 | Mixup barcode |
| 좌표 | $(b_{im}, d_{im}, d_{dom})$ — 3차원 전체 활용 |
| Weight function | **1** (균등 가중) |
| 성능 | **하위** — weight=1이 저조한 성능의 주요 원인 |
| 보완 계획 | weight=mixup 또는 weight=mixup+persistence 실험 예정 (Phase 3) |

---

### ④ Sixpack_Rips (Six-pack from Vietoris-Rips)

**최고 성능 벡터.** Rips complex 기반으로 L⊆K inclusion의 6가지 barcode를 모두 활용.

```mermaid
flowchart LR
    A["Red"] & B["Green"]
    A --> L["L = Rips(A)"]
    A & B --> K["K = Rips(A∪B)"]
    L & K --> SP["Six-pack\n6 barcode types"]
    SP --> PI_SP["PI × 6 × 2 방향\n→ Sixpack_Rips"]
```

| 항목 | 값 |
|------|-----|
| Filtration | Vietoris-Rips (max_edge=10) |
| Barcode 종류 | image, kernel, cokernel, complex, sub_complex, relative (6개) |
| 방향 | A→B + B→A (양방향) |
| 벡터 크기 | 6 × 2 × (H0:100D + H1:5000D) = **61,200D** |
| 성능 | **⭐ 최고** — Soft 93.0%, Strict 83.98% |

---

### ⑤ Sixpack_Chroma (Six-pack from Chromatic Alpha)

**Chromatic Alpha Complex 기반 six-pack.** 색상 정보를 filtration에 직접 반영하나, 성능은 최하위.

```mermaid
flowchart LR
    PTS["Points + Labels"] --> CAC["ChromaticAlphaComplex\nmax_alpha=10"]
    CAC --> SC["SimplicialComplex\nsub='0', full='all'"]
    SC --> SP["bars_six_pack()"]
    SP --> PI_SP["PI 벡터화\n→ Sixpack_Chroma"]
```

| 항목 | 값 |
|------|-----|
| Filtration | Chromatic Alpha Complex (`chromatic_tda` 라이브러리) |
| 파라미터 | max_alpha=10, sub_complex='0', full_complex='all' |
| 벡터 크기 | **61,200D** (Rips와 동일 구조) |
| 성능 | **최하위** ~77% — 파라미터 튜닝으로도 개선 불가 (구조적 한계) |
| H0 birth_range | (0, 0.01) — Rips 대비 매우 좁음 |

---

### ⑥ Inter+Ord (Interaction PI + Ordinary PI)

**Inter_PI와 Ord_PI를 단순 연결(concatenation)한 결합 벡터.**

```mermaid
flowchart LR
    INTER["Inter_PI"] --> CAT["Concatenate"]
    ORD["Ord_PI"] --> CAT
    CAT --> VEC["Inter+Ord Vector"]
```

| 항목 | 값 |
|------|-----|
| 구성 | Inter_PI ⊕ Ord_PI |
| 의도 | Interaction 정보 + Ordinary topology 정보 결합 |
| 성능 | Ord_PI보다 소폭 향상 (Inter_PI의 기여) |

---

### ⑦ 3D+Ord (3D PI + Ordinary PI)

**3D_PI와 Ord_PI를 단순 연결한 결합 벡터.**

```mermaid
flowchart LR
    THREED["3D_PI"] --> CAT["Concatenate"]
    ORD["Ord_PI"] --> CAT
    CAT --> VEC["3D+Ord Vector"]
```

| 항목 | 값 |
|------|-----|
| 구성 | 3D_PI ⊕ Ord_PI |
| 성능 | **Ord_PI보다 저조** — 3D_PI의 낮은 품질이 Ord_PI를 희석 |
| 원인 | 단순 concat 시 고차원 노이즈가 PCA 축소 과정에서 유용한 정보를 가림 |
| 보완 계획 | 개별 normalization → concat, weight 기반 결합 등 실험 예정 (Phase 4) |

---

### 성능 종합 비교

| 순위 | Vector | Soft Acc (%) | Strict Acc (%) | 핵심 특성 |
|:---:|--------|:-----------:|:-------------:|----------|
| 🥇 | **Sixpack_Rips** | ~93 | ~84 | Rips 기반 six-pack 전체 활용 |
| 🥈 | **Inter+Ord** | 중상위 | 중상위 | Interaction + Ordinary 결합 |
| 🥉 | **Inter_PI** | 중상위 | 중상위 | Mixup barcode의 interaction 정보 |
| 4 | **Ord_PI** | 중상위 | 중상위 | 기본 TDA 벡터 (baseline) |
| 5 | **3D+Ord** | 하위 | 하위 | 3D_PI가 Ord_PI를 희석 |
| 6 | **3D_PI** | 하위 | 하위 | weight=1의 한계 |
| 7 | **Sixpack_Chroma** | ~77 | ~68 | Chromatic Alpha의 구조적 한계 |
