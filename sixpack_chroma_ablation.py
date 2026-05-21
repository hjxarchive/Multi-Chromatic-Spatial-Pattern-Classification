"""
Sixpack_Chroma Ablation Study — 2^6 = 64 barcode subsets
Descriptors: image, kernel, cokernel, domain, codomain, relative

Classifiers:
  LinearSVM (hard margin, C=1e5)
  KNN (k=9)
  RF (n=200)

Evaluation:
  StandardScaler → PCA(100) → classifier
  StratifiedKFold(5), Soft / Strict accuracy
"""

import os, glob, warnings, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
warnings.filterwarnings('ignore')

BASE       = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"
CHROMA_DIR = os.path.join(BASE, "Final_Vector", "Sixpack_Chroma")

# ── Ground Truth ──────────────────────────────────────────────────────────────
M1=[[0,0,1,1,1,1,1,1],[0,0,1,1,1,1,1,1],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3]]
M2=[[0,0,1,1,1,1,1,1],[0,0,1,1,1,1,1,1],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,4],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,4,4],[2,2,3,3,3,3,3,3]]
M3=[[6,6,7,7,7,7,7,7],[6,6,6,7,7,7,7,7],[9,6,3,3,3,3,3,3],[9,10,3,4,4,3,3,4],[9,10,3,3,4,4,3,4],[9,10,3,4,4,4,4,4],[9,10,3,4,3,4,4,4],[9,10,3,4,3,4,4,4]]
M4=[[6,6,12,12,7,7,7,7],[6,6,12,12,7,7,7,7],[9,6,6,11,7,7,4,4],[9,9,6,3,3,4,4,4],[9,9,10,3,3,4,4,4],[9,9,10,3,3,4,4,4],[9,9,10,4,4,4,4,4],[9,9,10,4,4,4,4,4]]
M5=[[6,6,12,12,12,12,7,7],[6,6,12,12,12,12,12,7],[9,9,6,11,11,11,12,11],[9,9,6,11,11,11,4,4],[9,9,13,13,4,4,4,4],[9,9,13,10,4,4,4,4],[9,9,13,10,4,4,4,4],[9,9,10,10,4,4,4,4]]
M6=[[6,12,12,12,12,12,12,12],[6,6,12,12,12,12,12,12],[9,6,6,11,11,11,11,11],[9,9,6,11,11,11,11,11],[9,9,6,6,6,13,4,4],[9,9,6,13,13,4,4,4],[9,9,6,13,4,4,4,4],[9,9,6,13,4,4,4,4]]
M7=[[6,6,12,12,12,12,12,12],[9,6,12,12,12,12,12,12],[9,6,6,11,11,11,11,12],[9,6,6,11,11,11,11,11],[9,9,6,6,11,11,11,11],[9,9,6,6,11,11,11,4],[9,9,6,6,13,13,4,4],[9,9,6,13,13,4,4,4]]
M8=[[6,12,12,12,12,12,12,12],[6,6,12,12,12,12,12,12],[9,6,6,6,11,11,11,11],[9,6,6,6,11,11,11,11],[9,9,6,6,11,11,11,11],[9,9,6,6,6,11,11,11],[9,9,6,6,13,13,11,11],[9,9,6,6,13,13,11,4]]
GT = np.asarray([M1,M2,M3,M4,M5,M6,M7,M8])

def get_label(task_id):
    idx = task_id - 1
    return GT[(idx%64)//8][idx//64][idx%8]

def build_adj(M_list):
    adj = {}
    for M in M_list:
        M = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                cur = int(M[i,j])
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj=i+di,j+dj
                    if 0<=ni<M.shape[0] and 0<=nj<M.shape[1]:
                        nb=int(M[ni,nj])
                        if nb!=cur:
                            adj.setdefault(cur,[])
                            if nb not in adj[cur]: adj[cur].append(nb)
    return adj

ADJ = build_adj(GT)
def soft_acc(yt, yp):
    return np.mean([t==p or p in ADJ.get(t,[]) or t in ADJ.get(p,[]) for t,p in zip(yt,yp)])

# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
ALL_KEYS = ["image", "kernel", "cokernel", "sub_complex", "complex", "relative"]
ALIAS    = {"sub_complex": "domain", "complex": "codomain"}

def load_selected(selected_keys):
    files = sorted(glob.glob(os.path.join(CHROMA_DIR, "Sixpack_Chroma_*.npz")))
    X_list, y_list = [], []
    for fp in files:
        sim_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        data = np.load(fp, allow_pickle=True)
        arr_0 = data["arr_0"].item()
        arr_1 = data["arr_1"].item()
        feats = []
        for sp in [arr_0, arr_1]:
            for k in ALL_KEYS:
                if k not in selected_keys:
                    continue
                val = sp[k]
                for dim in sorted(val.keys()):
                    feats.extend(np.asarray(val[dim]).flatten())
        X_list.append(feats)
        y_list.append(get_label(sim_idx))
    return np.nan_to_num(np.array(X_list, dtype=float)), np.array(y_list)

# ── 평가 ─────────────────────────────────────────────────────────────────────
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

CLFS = {
    "LinSVM":  LinearSVC(C=1e5, max_iter=10000, dual=True),
    "KNN9":    KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    "RF200":   RandomForestClassifier(200, random_state=42, n_jobs=-1),
}

def evaluate(X, y, pca_dim=100):
    X_s = StandardScaler().fit_transform(X)
    if pca_dim and X_s.shape[1] > pca_dim:
        X_s = PCA(n_components=pca_dim, random_state=42).fit_transform(X_s)
    results = {}
    for name, clf_tmpl in CLFS.items():
        soft_l, strict_l = [], []
        for tr, te in SKF.split(X_s, y):
            c = clone(clf_tmpl)
            c.fit(X_s[tr], y[tr])
            yp = c.predict(X_s[te])
            soft_l.append(soft_acc(y[te], yp))
            strict_l.append(accuracy_score(y[te], yp))
        results[name] = (np.mean(soft_l)*100, np.mean(strict_l)*100)
    return results

# ── 2^6 ablation ─────────────────────────────────────────────────────────────
all_combos = []
for r in range(1, len(ALL_KEYS)+1):
    for combo in itertools.combinations(ALL_KEYS, r):
        all_combos.append(list(combo))

print(f"총 {len(all_combos)}개 subset (공집합 제외)")
print(f"Classifiers: {list(CLFS.keys())}")
print(f"Evaluation: StandardScaler → PCA(100) → 5-fold CV, Soft/Strict\n")

rows = []
for i, keys in enumerate(all_combos):
    X, y = load_selected(keys)
    res  = evaluate(X, y, pca_dim=100)
    dim  = X.shape[1]
    label = "+".join(ALIAS.get(k,k) for k in keys)
    n_comp = len(keys)
    rows.append({
        'combo':    label,
        'n':        n_comp,
        'dim':      dim,
        'keys':     keys,
        **{f'{clf}_{m}': v
           for clf, (soft, strict) in res.items()
           for m, v in [('soft', soft), ('strict', strict)]}
    })
    if (i+1) % 10 == 0 or (i+1) == len(all_combos):
        print(f"  [{i+1:2d}/{len(all_combos)}] {label:<50}  "
              f"RF soft={res['RF200'][0]:.1f}%  strict={res['RF200'][1]:.1f}%")

# ── 결과 정렬 및 출력 ─────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("전체 결과 — RF200 Soft 기준 내림차순")
print(f"  {'Combo':<48}  {'n':>2}  {'Dim':>6}  "
      f"{'LIN Soft':>9}  {'LIN Str':>8}  {'KNN Soft':>9}  {'KNN Str':>8}  "
      f"{'RF Soft':>8}  {'RF Str':>7}")
print("  " + "-"*108)

for row in sorted(rows, key=lambda r: -r['RF200_soft']):
    print(f"  {row['combo']:<48}  {row['n']:>2}  {row['dim']:>6}  "
          f"  {row['LinSVM_soft']:>8.2f}%  {row['LinSVM_strict']:>7.2f}%"
          f"  {row['KNN9_soft']:>8.2f}%  {row['KNN9_strict']:>7.2f}%"
          f"  {row['RF200_soft']:>7.2f}%  {row['RF200_strict']:>6.2f}%")

# ── 요약: 각 크기별 최고 조합 ─────────────────────────────────────────────────
print(f"\n{'='*90}")
print("요약: 각 subset 크기별 최고 조합 (RF Soft 기준)")
print(f"  {'n':>2}  {'Best combo':<48}  {'RF Soft':>8}  {'RF Strict':>9}  {'LIN Soft':>9}")
print("  " + "-"*82)
for n in range(1, 7):
    subset = [r for r in rows if r['n'] == n]
    best = max(subset, key=lambda r: r['RF200_soft'])
    print(f"  {n:>2}  {best['combo']:<48}  {best['RF200_soft']:>7.2f}%  "
          f"{best['RF200_strict']:>8.2f}%  {best['LinSVM_soft']:>8.2f}%")

# ── CSV 저장 ──────────────────────────────────────────────────────────────────
import csv
out_csv = os.path.join(BASE, "Final_Results(0521)", "sixpack_chroma_ablation.csv")
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['combo','n_barcodes','dim',
                'LinSVM_soft','LinSVM_strict',
                'KNN9_soft','KNN9_strict',
                'RF200_soft','RF200_strict'])
    for row in sorted(rows, key=lambda r: -r['RF200_soft']):
        w.writerow([row['combo'], row['n'], row['dim'],
                    f"{row['LinSVM_soft']:.4f}", f"{row['LinSVM_strict']:.4f}",
                    f"{row['KNN9_soft']:.4f}",   f"{row['KNN9_strict']:.4f}",
                    f"{row['RF200_soft']:.4f}",  f"{row['RF200_strict']:.4f}"])
print(f"\nCSV 저장: {out_csv}")
