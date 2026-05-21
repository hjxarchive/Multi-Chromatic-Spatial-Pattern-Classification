"""
Top-5 Family Benchmark (179-classifier paper 기반)
Families: RF, SVM, NNET, BST, BAG — 5개씩 총 25개 classifier
Descriptors: Ord_PI, Inter_PI, 3D_PI
Pipeline: StandardScaler → PCA(100) → 5-fold CV, Soft/Strict
"""

import os, glob, warnings
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# RF
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, AdaBoostClassifier,
                               BaggingClassifier, HistGradientBoostingClassifier)
# SVM
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
# NNET
from sklearn.neural_network import MLPClassifier
# BAG base
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# XGB / LGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

BASE       = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"
VECTOR_DIR = os.path.join(BASE, "Final_Vector")

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
                    ni,nj = i+di, j+dj
                    if 0<=ni<M.shape[0] and 0<=nj<M.shape[1]:
                        nb = int(M[ni,nj])
                        if nb != cur:
                            adj.setdefault(cur, [])
                            if nb not in adj[cur]: adj[cur].append(nb)
    return adj

ADJ = build_adj(GT)
def soft_acc(yt, yp):
    return np.mean([t==p or p in ADJ.get(t,[]) or t in ADJ.get(p,[])
                    for t,p in zip(yt,yp)])

# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_pi(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f'{prefix}_*.npz')))
    X_list, y_list = [], []
    for fp in files:
        sim_idx = int(os.path.basename(fp).split('_')[-1].split('.')[0])
        data = np.load(fp, allow_pickle=True)
        features = []
        for key in sorted(data.keys()):
            arr = data[key]
            if hasattr(arr, 'item') and arr.ndim == 0: arr = arr.item()
            if isinstance(arr, dict):
                for k in sorted(arr.keys()):
                    val = arr[k]
                    if isinstance(val, dict):
                        for dk in sorted(val.keys()): features.extend(np.asarray(val[dk]).flatten())
                    else: features.extend(np.asarray(val).flatten())
            else: features.extend(np.asarray(arr).flatten())
        X_list.append(features)
        y_list.append(get_label(sim_idx))
    return np.nan_to_num(np.array(X_list, dtype=float)), np.array(y_list)

print("Loading descriptors...")
_raw = {}
for name in ['Ord_PI', 'Inter_PI', '3D_PI', 'Sixpack_Rips', 'Sixpack_Chroma']:
    X, y = load_pi(os.path.join(VECTOR_DIR, name), name)
    _raw[name] = (X, y)
    print(f"  {name}: {X.shape}")

datasets = dict(_raw)

# Early fusion: Inter+Ord, 3D+Ord
X_inter, y_ = _raw['Inter_PI']
X_ord,   _  = _raw['Ord_PI']
X_3d,    _  = _raw['3D_PI']
datasets['Inter+Ord'] = (np.hstack([X_inter, X_ord]), y_)
datasets['3D+Ord']    = (np.hstack([X_3d,   X_ord]), y_)
print(f"  Inter+Ord: {datasets['Inter+Ord'][0].shape}")
print(f"  3D+Ord:    {datasets['3D+Ord'][0].shape}")

# ── Classifiers ───────────────────────────────────────────────────────────────
N_CLASSES = len(np.unique(list(datasets.values())[0][1]))

CLASSIFIERS = {
    # RF family
    "RF_n200":          RandomForestClassifier(200, random_state=42, n_jobs=-1),
    "RF_n500":          RandomForestClassifier(500, random_state=42, n_jobs=-1),
    "ET_n200":          ExtraTreesClassifier(200, random_state=42, n_jobs=-1),
    "ET_n500":          ExtraTreesClassifier(500, random_state=42, n_jobs=-1),
    "RF_log2":          RandomForestClassifier(200, max_features='log2', random_state=42, n_jobs=-1),

    # SVM family
    "SVM_lin":          SVC(kernel='linear', C=1.0),
    "SVM_rbf":          SVC(kernel='rbf',    C=1.0),
    "SVM_rbf_C2":       SVC(kernel='rbf',    C=2.0),
    "SVM_poly3":        SVC(kernel='poly',   degree=3, C=1.0),
    "SVM_sigmoid":      SVC(kernel='sigmoid', C=1.0),

    # NNET family
    "MLP_128_64":       MLPClassifier((128,64),  max_iter=500, early_stopping=True, random_state=42),
    "MLP_256_128_64":   MLPClassifier((256,128,64), max_iter=500, early_stopping=True, random_state=42),
    "MLP_512":          MLPClassifier((512,),    max_iter=500, early_stopping=True, random_state=42),
    "MLP_tanh":         MLPClassifier((128,64),  activation='tanh', max_iter=500, early_stopping=True, random_state=42),
    "MLP_sgd":          MLPClassifier((128,64),  solver='sgd', momentum=0.9, max_iter=500, early_stopping=True, random_state=42),

    # BST family
    "AdaBoost":         AdaBoostClassifier(n_estimators=200, random_state=42),
    "GBM":              GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "HistGBM":          HistGradientBoostingClassifier(max_iter=200, random_state=42),
    "XGB":              XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42,
                                      eval_metric='mlogloss', verbosity=0, n_jobs=-1),
    "LGBM":             LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42,
                                       n_jobs=-1, verbosity=-1),

    # BAG family
    "Bag_DT":           BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=42, n_jobs=-1),
    "Bag_KNN":          BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=20, random_state=42, n_jobs=-1),
    "Bag_SVM":          BaggingClassifier(SVC(kernel='rbf', C=1.0), n_estimators=10, random_state=42, n_jobs=-1),
    "Bag_MLP":          BaggingClassifier(MLPClassifier((128,64), max_iter=500, random_state=42), n_estimators=10, random_state=42, n_jobs=1),
    "Bag_LR":           BaggingClassifier(LogisticRegression(max_iter=1000, C=1.0), n_estimators=30, random_state=42, n_jobs=-1),
}

FAMILY = {
    "RF_n200": "RF", "RF_n500": "RF", "ET_n200": "RF", "ET_n500": "RF", "RF_log2": "RF",
    "SVM_lin": "SVM", "SVM_rbf": "SVM", "SVM_rbf_C2": "SVM", "SVM_poly3": "SVM", "SVM_sigmoid": "SVM",
    "MLP_128_64": "NNET", "MLP_256_128_64": "NNET", "MLP_512": "NNET", "MLP_tanh": "NNET", "MLP_sgd": "NNET",
    "AdaBoost": "BST", "GBM": "BST", "HistGBM": "BST", "XGB": "BST", "LGBM": "BST",
    "Bag_DT": "BAG", "Bag_KNN": "BAG", "Bag_SVM": "BAG", "Bag_MLP": "BAG", "Bag_LR": "BAG",
}

# ── 평가 ─────────────────────────────────────────────────────────────────────
SKF    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_CLIP  = 1e6
PCA_DIMS = [10, 20, 30, 50, 75, 100]

rows = []
n_total = len(datasets) * len(PCA_DIMS) * len(CLASSIFIERS)
done = 0

for desc_name, (X_raw, y) in datasets.items():
    print(f"\n{'='*60}")
    print(f"Descriptor: {desc_name}  shape={X_raw.shape}")
    print(f"{'='*60}")

    # XGB/LGBM용 연속 레이블 인코딩
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    for pca_dim in PCA_DIMS:
        for clf_name, clf_tmpl in CLASSIFIERS.items():
            soft_l, strict_l = [], []
            for tr, te in SKF.split(X_raw, y):
                ss    = StandardScaler()
                Xs_tr = np.clip(ss.fit_transform(X_raw[tr]), -_CLIP, _CLIP)
                Xs_te = np.clip(ss.transform(X_raw[te]),     -_CLIP, _CLIP)

                pca   = PCA(n_components=pca_dim, random_state=42)
                Z_tr  = pca.fit_transform(Xs_tr)
                Z_te  = pca.transform(Xs_te)

                clf = clone(clf_tmpl)
                try:
                    clf.fit(Z_tr, y_enc[tr])
                    yp_enc = clf.predict(Z_te)
                    yp     = le.inverse_transform(yp_enc.astype(int))
                    soft_l.append(soft_acc(y[te], yp))
                    strict_l.append(accuracy_score(y[te], yp))
                except Exception as e:
                    # fold 실패 시 skip (결과에서 해당 fold 제외)
                    print(f"    [WARN] fold skip: {e}")

            s  = np.mean(soft_l)   * 100 if soft_l   else float('nan')
            st = np.mean(strict_l) * 100 if strict_l else float('nan')
            done += 1
            print(f"  [{done:>4d}/{n_total}] {desc_name:<14} PCA{pca_dim:<4} "
                  f"{clf_name:<22} Soft={s:.2f}%  Strict={st:.2f}%")
            rows.append({
                'descriptor': desc_name,
                'pca':        pca_dim,
                'family':     FAMILY[clf_name],
                'classifier': clf_name,
                'soft':       s,
                'strict':     st,
            })

# ── 결과 출력 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("결과 — Descriptor × Classifier (Soft 기준)")
print(f"{'='*70}")

for desc_name in ['Ord_PI', 'Inter_PI', '3D_PI', 'Sixpack_Rips', 'Sixpack_Chroma', 'Inter+Ord', '3D+Ord']:
    sub = [r for r in rows if r['descriptor'] == desc_name]
    # best PCA per (classifier) 기준 top 출력
    by_clf = {}
    for r in sub:
        key = r['classifier']
        if key not in by_clf or r['soft'] > by_clf[key]['soft']:
            by_clf[key] = r
    sub_sorted = sorted(by_clf.values(), key=lambda r: -r['soft'])
    print(f"\n[{desc_name}]  (best PCA per classifier)")
    print(f"  {'Rank':<5} {'Family':<6} {'Classifier':<22} {'PCA':>5} {'Soft':>8}  {'Strict':>8}")
    print(f"  {'-'*60}")
    for rank, r in enumerate(sub_sorted, 1):
        print(f"  {rank:<5} {r['family']:<6} {r['classifier']:<22} {r['pca']:>5} {r['soft']:>7.2f}%  {r['strict']:>7.2f}%")

# ── 계열별 최고 ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("계열별 최고 (7 descriptor × best PCA 평균 Soft)")
print(f"{'='*70}")
for family in ['RF', 'SVM', 'NNET', 'BST', 'BAG']:
    sub = [r for r in rows if r['family'] == family]
    by_clf = {}
    for r in sub:
        key = (r['descriptor'], r['classifier'])
        if key not in by_clf or r['soft'] > by_clf[key]:
            by_clf[key] = r['soft']
    by_clf_avg = {}
    for (desc, clf), s in by_clf.items():
        by_clf_avg.setdefault(clf, []).append(s)
    avg = {k: np.mean(v) for k, v in by_clf_avg.items()}
    best_clf = max(avg, key=avg.get)
    print(f"  {family:<6}  best={best_clf:<22}  avg_soft={avg[best_clf]:.2f}%")

# ── CSV 저장 ──────────────────────────────────────────────────────────────────
out_dir = os.path.join(BASE, "Final_Results(0522)")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "top5family_benchmark.csv")
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['descriptor','pca','family','classifier','soft','strict'])
    w.writeheader()
    for r in rows:
        w.writerow({**r, 'soft': f"{r['soft']:.4f}", 'strict': f"{r['strict']:.4f}"})
print(f"\nCSV 저장: {out_csv}")
