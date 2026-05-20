"""
Feature space의 선형 분리 가능성 실험적 입증
속도를 위해 PCA로 먼저 축소 후 실험
"""
import os, glob, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.base import clone
warnings.filterwarnings('ignore')

BASE = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"
RIPS_DIR = os.path.join(BASE, "Final_Vector", "Sixpack_Rips")

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
                    ni,nj = i+di,j+dj
                    if 0<=ni<M.shape[0] and 0<=nj<M.shape[1] and int(M[ni,nj])!=cur:
                        adj.setdefault(cur,[])
                        nb = int(M[ni,nj])
                        if nb not in adj[cur]: adj[cur].append(nb)
    return adj

ADJ = build_adj(GT)

def soft_acc(yt, yp):
    return np.mean([t==p or p in ADJ.get(t,[]) or t in ADJ.get(p,[]) for t,p in zip(yt,yp)])

# ── 데이터 로드 ─────────────────────────────────────────────────────────────
print("데이터 로딩...", end=" ", flush=True)
files = sorted(glob.glob(os.path.join(RIPS_DIR, "Sixpack_Rips_*.npz")))
X_list, y_list = [], []
for fp in files:
    sim_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
    data = np.load(fp, allow_pickle=True)
    feats = []
    for k in sorted(data.keys()):
        arr = data[k]
        if hasattr(arr,'item') and arr.ndim==0: arr = arr.item()
        if isinstance(arr, dict):
            for v in arr.values():
                if isinstance(v, dict):
                    for vv in v.values(): feats.extend(np.asarray(vv).flatten())
                else: feats.extend(np.asarray(v).flatten())
        else: feats.extend(arr.flatten())
    X_list.append(feats)
    y_list.append(get_label(sim_idx))

X_raw = np.nan_to_num(np.array(X_list, dtype=float))
y = np.array(y_list)
print(f"완료: {X_raw.shape}")

# StandardScaler → PCA 200D (속도용)
print("StandardScaler + PCA 200D...", end=" ", flush=True)
X_sc = StandardScaler().fit_transform(X_raw)
X200 = PCA(n_components=200, random_state=42).fit_transform(X_sc)
print("완료")

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv(clf, Xin, y):
    soft_l, strict_l = [], []
    for tr, te in SKF.split(Xin, y):
        c = clone(clf); c.fit(Xin[tr], y[tr])
        yp = c.predict(Xin[te])
        soft_l.append(soft_acc(y[te], yp))
        strict_l.append(accuracy_score(y[te], yp))
    c_tr = clone(clf); c_tr.fit(Xin, y)
    tr_soft = soft_acc(y, c_tr.predict(Xin))
    return np.mean(soft_l)*100, np.mean(strict_l)*100, tr_soft*100

SEP = "=" * 60

# ── 실험 1. LinearSVC C sweep (PCA200) ────────────────────────────────────
print(f"\n{SEP}")
print("실험 1. LinearSVC C sweep  (C 작을수록 large-margin)")
print(f"  {'C':>8}  {'Soft CV':>9}  {'Strict CV':>10}  {'Soft Train':>11}")
print("  " + "-"*45)
for C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
    s, st, tr = cv(LinearSVC(C=C, max_iter=5000, dual=True), X200, y)
    marker = " ← large margin" if C <= 0.001 else ""
    print(f"  {C:>8.4f}  {s:>8.2f}%  {st:>9.2f}%  {tr:>10.2f}%{marker}")

# ── 실험 2. Hard-margin 근사 ────────────────────────────────────────────────
print(f"\n{SEP}")
print("실험 2. Hard-margin 근사 (C=1e5)")
print("  Train Soft=100% → 선형 초평면이 실제로 존재")
s, st, tr = cv(LinearSVC(C=1e5, max_iter=10000, dual=True), X200, y)
print(f"  Soft CV={s:.2f}%  Strict CV={st:.2f}%  Soft Train={tr:.2f}%")

# ── 실험 3. 다양한 선형 분류기 ──────────────────────────────────────────────
print(f"\n{SEP}")
print("실험 3. 다양한 선형 분류기 (모두 일관되게 고성능?)")
print(f"  {'Classifier':<28}  {'Soft CV':>9}  {'Strict CV':>10}  {'Soft Train':>11}")
print("  " + "-"*60)
linear_clfs = {
    "LinearSVC (C=1)":            LinearSVC(C=1, max_iter=5000),
    "LogisticRegression (C=1)":   LogisticRegression(C=1, max_iter=2000, random_state=42),
    "LogisticRegression (C=0.01)":LogisticRegression(C=0.01, max_iter=2000, random_state=42),
    "LDA":                        LinearDiscriminantAnalysis(),
    "Perceptron":                 Perceptron(max_iter=2000, random_state=42),
}
for name, clf in linear_clfs.items():
    s, st, tr = cv(clf, X200, y)
    print(f"  {name:<28}  {s:>8.2f}%  {st:>9.2f}%  {tr:>10.2f}%")

# ── 실험 4. Polynomial degree sweep ──────────────────────────────────────
print(f"\n{SEP}")
print("실험 4. Kernel degree sweep (d=1 선형이 최고면 비선형 불필요)")
print(f"  {'Kernel':<22}  {'Soft CV':>9}  {'Strict CV':>10}")
print("  " + "-"*44)
for d in [1, 2, 3]:
    s, st, _ = cv(SVC(kernel='poly', degree=d, C=1.0, coef0=1), X200, y)
    tag = " ← linear" if d==1 else ""
    print(f"  {'poly (degree='+str(d)+')':.<22}  {s:>8.2f}%  {st:>9.2f}%{tag}")
s, st, _ = cv(SVC(kernel='rbf', C=1.0), X200, y)
print(f"  {'RBF (C=1)':.<22}  {s:>8.2f}%  {st:>9.2f}%")

# ── 실험 5. PCA 차원 sweep + LinearSVC ─────────────────────────────────────
print(f"\n{SEP}")
print("실험 5. PCA 차원 sweep + LinearSVC (몇 차원부터 100% 도달?)")
print(f"  {'PCA dim':>9}  {'Soft CV':>9}  {'Strict CV':>10}")
print("  " + "-"*32)
for dim in [2, 5, 10, 20, 50, 100, 200]:
    Xp = PCA(n_components=dim, random_state=42).fit_transform(X_sc)
    s, st, _ = cv(LinearSVC(C=1, max_iter=5000), Xp, y)
    tag = " ★" if s >= 99.9 else ""
    print(f"  {dim:>9}  {s:>8.2f}%  {st:>9.2f}%{tag}")

print(f"\n{SEP}")
