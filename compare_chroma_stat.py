"""
Sixpack_Chroma에 동일한 stat features 추출 (288D) 적용 후
Sixpack_Rips (288D) 와 성능 비교
"""
import os, glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"
RIPS_DIR   = os.path.join(BASE, "Final_Vector", "Sixpack_Rips")
CHROMA_DIR = os.path.join(BASE, "Final_Vector", "Sixpack_Chroma")

# ── Ground Truth ───────────────────────────────────────────────────────────────
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
    return GT[(idx % 64) // 8][idx // 64][idx % 8]

def extract_adjacent_phases(matrices):
    adj = {}
    for M in matrices:
        M = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                cur = int(M[i,j])
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj = i+di,j+dj
                    if 0<=ni<M.shape[0] and 0<=nj<M.shape[1] and M[ni,nj]!=cur:
                        nb = int(M[ni,nj])
                        adj.setdefault(cur,[])
                        if nb not in adj[cur]: adj[cur].append(nb)
    return adj

ADJ = extract_adjacent_phases(GT)

def soft_acc(y_true, y_pred):
    n = len(y_true)
    correct = sum(1 for t,p in zip(y_true,y_pred)
                  if t==p or p in ADJ.get(t,[]) or t in ADJ.get(p,[]))
    return correct / n

# ── Feature extraction ─────────────────────────────────────────────────────────
BARCODE_TYPES_RIPS   = ["domain","codomain","relative","image","kernel","cokernel"]
BARCODE_TYPES_CHROMA = ["sub_complex","complex","relative","image","kernel","cokernel"]

def extract_statistical_features(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return np.zeros(12)
    if arr.ndim == 1:
        if len(arr) % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            arr = arr[:len(arr)//2*2].reshape(-1,2) if len(arr) > 2 else np.array([[0.,0.]])
    if arr.ndim == 1 or arr.shape[1] < 2:
        return np.zeros(12)
    ls = arr[:,1] - arr[:,0]
    b, d = arr[:,0], arr[:,1]
    feats = [len(arr), np.mean(ls), np.std(ls), np.max(ls), np.min(ls), np.sum(ls),
             np.mean(b), np.std(b), np.mean(d), np.std(d), np.median(ls)]
    p = ls / np.sum(ls) if np.sum(ls) > 0 else ls
    p = p[p > 0]
    feats.append(-np.sum(p * np.log(p + 1e-10)) if len(p) > 0 else 0)
    return np.array(feats)

def load_stat_features(data_dir, prefix, barcode_types):
    """PI 벡터 → stat features (288D) 추출 (Rips/Chroma 공통)"""
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.npz")))
    X_list, y_list = [], []
    key_map = {"domain": "sub_complex", "codomain": "complex"}  # Rips 키명 → Chroma 키명

    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
            label = get_label(sim_idx)
            data = np.load(fp, allow_pickle=True)
            arr_0 = data["arr_0"].item()
            arr_1 = data["arr_1"].item()
            feats = []
            for sp_dict in [arr_0, arr_1]:
                for bt in barcode_types:
                    # Rips는 domain/codomain, Chroma는 sub_complex/complex
                    actual_key = key_map.get(bt, bt) if bt in key_map and bt not in sp_dict else bt
                    for dim in [0, 1]:
                        if actual_key in sp_dict and dim in sp_dict[actual_key]:
                            feats.extend(extract_statistical_features(sp_dict[actual_key][dim]))
                        else:
                            feats.extend(np.zeros(12))
            X_list.append(feats)
            y_list.append(label)
        except Exception as e:
            print(f"  Error {fp}: {e}")
    return np.nan_to_num(np.array(X_list)), np.array(y_list)

# ── 평가 ───────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS = 5
REDUCTION_DIM = 20

def evaluate(X, y, pca_dim=REDUCTION_DIM):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if X_s.shape[1] > pca_dim:
        X_r = PCA(n_components=pca_dim, random_state=RANDOM_STATE).fit_transform(X_s)
        print(f"  PCA: {X_s.shape[1]}D → {pca_dim}D")
    else:
        X_r = X_s
        print(f"  No PCA needed ({X_s.shape[1]}D)")

    clfs = {
        "KNN (k=3)":     KNeighborsClassifier(3),
        "KNN (k=12)":    KNeighborsClassifier(12),
        "SVM (RBF)":     SVC(kernel="rbf", C=1.0, gamma="scale"),
        "SVM (Linear)":  SVC(kernel="linear", C=1.0),
        "Random Forest": RandomForestClassifier(100, random_state=RANDOM_STATE),
    }
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, tmpl in clfs.items():
        soft_list, strict_list = [], []
        for tr, te in skf.split(X_r, y):
            c = clone(tmpl)
            c.fit(X_r[tr], y[tr])
            yp = c.predict(X_r[te])
            soft_list.append(soft_acc(y[te], yp))
            strict_list.append(accuracy_score(y[te], yp))
        results[name] = {
            "soft":   (np.mean(soft_list)*100, np.std(soft_list)*100),
            "strict": (np.mean(strict_list)*100, np.std(strict_list)*100),
        }
    return results

# ── 실행 ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("데이터 로딩...")

print("\n[Sixpack_Rips] stat features 로딩...")
X_rips, y_rips = load_stat_features(RIPS_DIR, "Sixpack_Rips", BARCODE_TYPES_RIPS)
print(f"  shape: {X_rips.shape}")

print("\n[Sixpack_Chroma] stat features 로딩...")
X_chroma, y_chroma = load_stat_features(CHROMA_DIR, "Sixpack_Chroma", BARCODE_TYPES_CHROMA)
print(f"  shape: {X_chroma.shape}")

print("\n" + "=" * 70)
datasets = {
    "Sixpack_Rips  (288D, stat)":   (X_rips,   y_rips),
    "Sixpack_Chroma (288D, stat)":  (X_chroma, y_chroma),
}

all_res = {}
for name, (X, y) in datasets.items():
    print(f"\n▶ [{name}]")
    res = evaluate(X, y)
    all_res[name] = res
    best = max(res, key=lambda k: res[k]["soft"][0])
    r = res[best]
    print(f"  Best: {best} → Soft {r['soft'][0]:.2f}±{r['soft'][1]:.2f}%  |  Strict {r['strict'][0]:.2f}±{r['strict'][1]:.2f}%")

print("\n" + "=" * 70)
print(f"{'Method':<30} {'Classifier':<20} {'Soft':>12} {'Strict':>12}")
print("-" * 70)
for name, res in all_res.items():
    for clf, r in res.items():
        print(f"{name:<30} {clf:<20} {r['soft'][0]:>6.2f}±{r['soft'][1]:.2f}%  {r['strict'][0]:>6.2f}±{r['strict'][1]:.2f}%")
    print()
