"""
0506 Update — 보완 실험 통합 파이프라인
Google Colab에서 실행
"""

# ============================================================
# 0. 환경 설정
# ============================================================
import os, glob, gc, time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.base import clone
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# Google Colab 환경
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_DIR = "/content/drive/MyDrive/URP" if IN_COLAB else "."
VECTOR_DIR = os.path.join(BASE_DIR, "1224_Vectors")
OUTPUT_DIR = os.path.join(BASE_DIR, "0506_update")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("All imports loaded.")
print(f"IN_COLAB={IN_COLAB}, VECTOR_DIR={VECTOR_DIR}")

# ============================================================
# 1. Ground Truth & Adjacent Phases & Soft Accuracy
# ============================================================
M1=[[0,0,1,1,1,1,1,1],[0,0,1,1,1,1,1,1],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3]]
M2=[[0,0,1,1,1,1,1,1],[0,0,1,1,1,1,1,1],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,3,4],[2,2,3,3,3,3,3,3],[2,2,3,3,3,3,4,4],[2,2,3,3,3,3,3,3]]
M3=[[6,6,7,7,7,7,7,7],[6,6,6,7,7,7,7,7],[9,6,3,3,3,3,3,3],[9,10,3,4,4,3,3,4],[9,10,3,3,4,4,3,4],[9,10,3,4,4,4,4,4],[9,10,3,4,3,4,4,4],[9,10,3,4,3,4,4,4]]
M4=[[6,6,12,12,7,7,7,7],[6,6,12,12,7,7,7,7],[9,6,6,11,7,7,4,4],[9,9,6,3,3,4,4,4],[9,9,10,3,3,4,4,4],[9,9,10,3,3,4,4,4],[9,9,10,4,4,4,4,4],[9,9,10,4,4,4,4,4]]
M5=[[6,6,12,12,12,12,7,7],[6,6,12,12,12,12,12,7],[9,9,6,11,11,11,12,11],[9,9,6,11,11,11,4,4],[9,9,13,13,4,4,4,4],[9,9,13,10,4,4,4,4],[9,9,13,10,4,4,4,4],[9,9,10,10,4,4,4,4]]
M6=[[6,12,12,12,12,12,12,12],[6,6,12,12,12,12,12,12],[9,6,6,11,11,11,11,11],[9,9,6,11,11,11,11,11],[9,9,6,6,6,13,4,4],[9,9,6,13,13,4,4,4],[9,9,6,13,4,4,4,4],[9,9,6,13,4,4,4,4]]
M7=[[6,6,12,12,12,12,12,12],[9,6,12,12,12,12,12,12],[9,6,6,11,11,11,11,12],[9,6,6,11,11,11,11,11],[9,9,6,6,11,11,11,11],[9,9,6,6,11,11,11,4],[9,9,6,6,13,13,4,4],[9,9,6,13,13,4,4,4]]
M8=[[6,12,12,12,12,12,12,12],[6,6,12,12,12,12,12,12],[9,6,6,6,11,11,11,11],[9,6,6,6,11,11,11,11],[9,9,6,6,11,11,11,11],[9,9,6,6,6,11,11,11],[9,9,6,6,13,13,11,11],[9,9,6,6,13,13,11,4]]
GROUND_TRUTH_M = np.asarray([M1,M2,M3,M4,M5,M6,M7,M8])

def get_label_from_index(task_id):
    idx = task_id - 1
    RR_idx = idx // 64
    RG_idx = (idx % 64) // 8
    GG_idx = idx % 8
    return GROUND_TRUTH_M[RG_idx][RR_idx][GG_idx]

def extract_adjacent_phases(matrices):
    adj = set()
    for M in matrices:
        M = np.array(M)
        r, c = M.shape
        for i in range(r):
            for j in range(c):
                cur = M[i,j]
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0<=ni<r and 0<=nj<c and cur!=M[ni,nj]:
                        adj.add(tuple(sorted([int(cur),int(M[ni,nj])])))
    d = {}
    for p1,p2 in adj:
        d.setdefault(p1,[]).append(p2)
        d.setdefault(p2,[]).append(p1)
    return d

ADJACENT_PHASES = extract_adjacent_phases(GROUND_TRUTH_M)

def soft_accuracy_score(y_true, y_pred, adj=ADJACENT_PHASES):
    n = len(y_true)
    correct = sum(1 for t,p in zip(y_true,y_pred)
                  if t==p or (t in adj and p in adj[t]) or (p in adj and t in adj[p]))
    return correct/n if n>0 else 0.0

ALL_CLASSES = sorted(np.unique(GROUND_TRUTH_M))
print(f"Classes: {ALL_CLASSES} ({len(ALL_CLASSES)} classes)")
print(f"Adjacent pairs: {len(ADJACENT_PHASES)} entries")

# ============================================================
# 2. 데이터 로딩 함수
# ============================================================

def load_generic_pi(data_dir, prefix):
    """Inter_PI / Ord_PI / 3D_PI 공용 PI 벡터 로더."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.npz")))
    X_list, y_list = [], []
    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
            label = get_label_from_index(sim_idx)
            data = np.load(fp, allow_pickle=True)
            features = []
            for key in ("arr_0", "arr_1"):
                arr = data[key]
                if hasattr(arr, "item") and arr.ndim == 0: arr = arr.item()
                elif arr.shape == (1,): arr = arr[0]
                if isinstance(arr, dict):
                    for k in sorted(arr.keys()):
                        val = arr[k]
                        if isinstance(val, dict):
                            for dk in sorted(val.keys()): features.extend(np.asarray(val[dk]).flatten())
                        else: features.extend(np.asarray(val).flatten())
                else: features.extend(np.asarray(arr).flatten())
            X_list.append(features); y_list.append(label)
        except Exception as e: print(f"  Error {fp}: {e}")
    if not X_list: return None, None
    return np.nan_to_num(np.array(X_list, dtype=float)), np.array(y_list)

def extract_statistical_features(barcode):
    if len(barcode)==0: return np.zeros(12)
    bc = np.array(barcode)
    if bc.ndim==1:
        if len(bc)%2==0: bc=bc.reshape(-1,2)
        else: bc=bc[:len(bc)//2*2].reshape(-1,2) if len(bc)>2 else np.array([[0.,0.]])
    if bc.ndim==1 or bc.shape[1]<2: return np.zeros(12)
    ls=bc[:,1]-bc[:,0]; b=bc[:,0]; d=bc[:,1]
    feats=[len(bc),np.mean(ls),np.std(ls),np.max(ls),np.min(ls),np.sum(ls),
           np.mean(b),np.std(b),np.mean(d),np.std(d),np.median(ls)]
    p=ls/np.sum(ls) if np.sum(ls)>0 else ls; p=p[p>0]
    feats.append(-np.sum(p*np.log(p+1e-10)) if len(p)>0 else 0)
    return np.array(feats)

BARCODE_TYPES = ["domain","codomain","relative","image","kernel","cokernel"]

def load_sixpack_rips(data_dir, selected_types=None):
    if selected_types is None: selected_types = BARCODE_TYPES
    files = sorted(glob.glob(os.path.join(data_dir, "Sixpack_Rips_*.npz")))
    X_list, y_list = [], []
    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
            label = get_label_from_index(sim_idx)
            data = np.load(fp, allow_pickle=True)
            sp = {"A_to_B": data["arr_0"].item(), "B_to_A": data["arr_1"].item()}
            feats = []
            for d_key in ["A_to_B","B_to_A"]:
                dd = sp[d_key]
                for bt in BARCODE_TYPES:
                    for dim_key in [0,1]:
                        if bt in selected_types and bt in dd and dim_key in dd[bt]:
                            feats.extend(extract_statistical_features(np.array(dd[bt][dim_key])))
                        elif bt in selected_types:
                            feats.extend(np.zeros(12))
            X_list.append(feats); y_list.append(label)
        except Exception as e: print(f"  Error {fp}: {e}")
    if not X_list: return None, None
    return np.nan_to_num(np.array(X_list)), np.array(y_list)

def load_sixpack_chroma(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "Sixpack_Chroma_*.npz")))
    X_list, y_list = [], []
    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split("_")[-1].split(".")[0])
            label = get_label_from_index(sim_idx)
            data = np.load(fp, allow_pickle=True)
            features = []
            for key in ("arr_0","arr_1"):
                arr = data[key]
                if hasattr(arr,"item") and arr.ndim==0: arr = arr.item()
                if isinstance(arr, dict):
                    for k in sorted(arr.keys()):
                        val = arr[k]
                        if isinstance(val, dict):
                            for dk in sorted(val.keys()): features.extend(np.asarray(val[dk]).flatten())
                        else: features.extend(np.asarray(val).flatten())
                else: features.extend(np.asarray(arr).flatten())
            X_list.append(features); y_list.append(label)
        except Exception as e: print(f"  Error {fp}: {e}")
    if not X_list: return None, None
    return np.nan_to_num(np.array(X_list, dtype=float)), np.array(y_list)

print("Data loading functions defined.")

# ============================================================
# 3. 통합 평가 함수 (수행 2-4: 모든 분류기 + Strict + F1)
# ============================================================
REDUCTION_DIM = 20
N_SPLITS = 5
RANDOM_STATE = 42
C_VALUES = [0.5, 1.0, 2.0]

def get_all_classifiers():
    clfs = {
        "KNN (k=3)":       KNeighborsClassifier(3),
        "KNN (k=12)":      KNeighborsClassifier(12),
        "SVM (RBF)":       SVC(kernel="rbf", C=1., gamma="scale"),
        "SVM (Linear)":    SVC(kernel="linear", C=1.),
        "Random Forest":   RandomForestClassifier(100, random_state=RANDOM_STATE),
    }
    for C in C_VALUES:
        clfs[f"Soft-SVM (C={C})"] = SVC(kernel="rbf", C=C, gamma="scale")
    # 수행 3: Ward Linkage
    clfs["Ward (k=12)"] = AgglomerativeClustering(n_clusters=12, linkage="ward")
    return clfs

def evaluate_all_classifiers(X, y, n_splits=N_SPLITS):
    """모든 분류기에 대해 Soft Acc, Strict Acc, Macro F1 계산."""
    clfs = get_all_classifiers()
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, ct in clfs.items():
        if isinstance(ct, AgglomerativeClustering):
            # Ward는 비지도 → 지도 평가 불가, skip
            continue
        acc_soft, acc_strict, f1_macro = [], [], []
        all_y_true, all_y_pred = [], []
        for tri, tei in skf.split(X, y):
            c = clone(ct); c.fit(X[tri], y[tri]); yp = c.predict(X[tei])
            acc_soft.append(soft_accuracy_score(y[tei], yp))
            acc_strict.append(accuracy_score(y[tei], yp))
            f1_macro.append(f1_score(y[tei], yp, average="macro", zero_division=0))
            all_y_true.extend(y[tei]); all_y_pred.extend(yp)
        results[name] = {
            "mean_soft": np.mean(acc_soft)*100, "std_soft": np.std(acc_soft)*100,
            "mean_strict": np.mean(acc_strict)*100, "std_strict": np.std(acc_strict)*100,
            "mean_f1": np.mean(f1_macro)*100, "std_f1": np.std(f1_macro)*100,
            "y_true": np.array(all_y_true), "y_pred": np.array(all_y_pred),
        }
    return results

def full_evaluate(X, y, reduction_dim=REDUCTION_DIM):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if X_scaled.shape[1] > reduction_dim:
        X_reduced = PCA(n_components=reduction_dim, random_state=RANDOM_STATE).fit_transform(X_scaled)
    else:
        X_reduced = X_scaled
    clf_results = evaluate_all_classifiers(X_reduced, y)
    return {"classifiers": clf_results,
            "original_dim": X.shape[1], "reduced_dim": X_reduced.shape[1],
            "X_reduced": X_reduced, "y": y}

CLASSIFIERS_LIST = ["KNN (k=3)","KNN (k=12)","SVM (RBF)","SVM (Linear)",
                    "Random Forest","Soft-SVM (C=0.5)","Soft-SVM (C=1.0)","Soft-SVM (C=2.0)"]

print(f"Evaluation functions defined: PCA={REDUCTION_DIM}D, {N_SPLITS}-fold CV")

# ============================================================
# 4. 전체 데이터 로딩
# ============================================================
def load_all_datasets():
    datasets = {}
    print("=" * 80)
    print("데이터 로딩")
    print("=" * 80)

    for name in ["Inter_PI", "3D_PI", "Ord_PI"]:
        print(f"\n[{name}]...", end=" ")
        X, y = load_generic_pi(os.path.join(VECTOR_DIR, name), name)
        if X is not None:
            datasets[name] = {"X": X, "y": y}
            print(f"✓ {X.shape}")
        else: print("✗")

    print("\n[Sixpack_Rips]...", end=" ")
    X, y = load_sixpack_rips(os.path.join(VECTOR_DIR, "Sixpack_Rips"))
    if X is not None:
        datasets["Sixpack_Rips"] = {"X": X, "y": y}
        print(f"✓ {X.shape}")

    print("\n[Sixpack_Chroma]...", end=" ")
    X, y = load_sixpack_chroma(os.path.join(VECTOR_DIR, "Sixpack_Chroma"))
    if X is not None:
        datasets["Sixpack_Chroma"] = {"X": X, "y": y}
        print(f"✓ {X.shape}")

    if "Inter_PI" in datasets and "Ord_PI" in datasets:
        datasets["Inter+Ord"] = {"X": np.hstack([datasets["Inter_PI"]["X"], datasets["Ord_PI"]["X"]]),
                                 "y": datasets["Inter_PI"]["y"]}
        print(f"\n[Inter+Ord] ✓ {datasets['Inter+Ord']['X'].shape}")
    if "3D_PI" in datasets and "Ord_PI" in datasets:
        datasets["3D+Ord"] = {"X": np.hstack([datasets["3D_PI"]["X"], datasets["Ord_PI"]["X"]]),
                              "y": datasets["3D_PI"]["y"]}
        print(f"[3D+Ord] ✓ {datasets['3D+Ord']['X'].shape}")

    return datasets

# ============================================================
# 5. Confusion Matrix 생성 (수행 5)
# ============================================================
def plot_confusion_matrix(y_true, y_pred, method_name, clf_name="SVM (Linear)"):
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix: {method_name}\n({clf_name})", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(ALL_CLASSES))); ax.set_xticklabels(ALL_CLASSES, fontsize=9)
    ax.set_yticks(range(len(ALL_CLASSES))); ax.set_yticklabels(ALL_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(ALL_CLASSES)):
        for j in range(len(ALL_CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{method_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()

# ============================================================
# 6. 2D Embedding Visualization (수행 6)
# ============================================================
def plot_2d_embedding(X_reduced, y, method_name):
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_2d = tsne.fit_transform(X_reduced)
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in ALL_CLASSES:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Phase {cls}", s=30, alpha=0.7)
    ax.set_title(f"t-SNE 2D Embedding: {method_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"tsne_{method_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()

# ============================================================
# 7. Clustering Statistics (수행 7)
# ============================================================
def compute_clustering_stats(X_reduced, y):
    sil = silhouette_score(X_reduced, y)
    ch = calinski_harabasz_score(X_reduced, y)
    db = davies_bouldin_score(X_reduced, y)
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}

# ============================================================
# MAIN: 전체 실행
# ============================================================
if __name__ == "__main__":
    datasets = load_all_datasets()
    METHODS = [m for m in ["Ord_PI","Inter_PI","3D_PI","Sixpack_Rips",
                           "Sixpack_Chroma","Inter+Ord","3D+Ord"] if m in datasets]

    # --- 수행 2-4: 전체 평가 ---
    all_results = {}
    print("\n" + "=" * 100)
    print("전체 평가 시작 (PCA 20D, 5-fold Stratified CV)")
    print("=" * 100)
    for name in METHODS:
        print(f"\n▶ [{name}] ({datasets[name]['X'].shape})")
        t0 = time.time()
        res = full_evaluate(datasets[name]["X"], datasets[name]["y"])
        all_results[name] = res
        cr = res["classifiers"]
        best_clf = max(cr, key=lambda k: cr[k]["mean_soft"])
        r = cr[best_clf]
        print(f"  Best: {best_clf} -> Soft={r['mean_soft']:.2f}% | "
              f"Strict={r['mean_strict']:.2f}% | F1={r['mean_f1']:.2f}%  ({time.time()-t0:.1f}s)")

    # --- 종합 결과 테이블 (Strict + F1 포함) ---
    print("\n\n=== 종합 결과 테이블 (모든 분류기) ===")
    for method in METHODS:
        print(f"\n--- {method} (dim={all_results[method]['original_dim']}) ---")
        cr = all_results[method]["classifiers"]
        print(f"{'Classifier':<22} {'Soft(%)':<14} {'Strict(%)':<14} {'F1(%)':<14}")
        print("-" * 64)
        for clf_name in CLASSIFIERS_LIST:
            if clf_name in cr:
                r = cr[clf_name]
                print(f"{clf_name:<22} {r['mean_soft']:.2f}±{r['std_soft']:.2f}  "
                      f"{r['mean_strict']:.2f}±{r['std_strict']:.2f}  "
                      f"{r['mean_f1']:.2f}±{r['std_f1']:.2f}")

    # --- 수행 5: Confusion Matrix ---
    print("\n\n=== Confusion Matrix 생성 (SVM Linear) ===")
    for method in METHODS:
        cr = all_results[method]["classifiers"]
        if "SVM (Linear)" in cr:
            plot_confusion_matrix(cr["SVM (Linear)"]["y_true"],
                                  cr["SVM (Linear)"]["y_pred"], method)

    # --- 수행 6: 2D Embedding ---
    print("\n\n=== 2D Embedding Visualization ===")
    for method in METHODS:
        plot_2d_embedding(all_results[method]["X_reduced"],
                          all_results[method]["y"], method)

    # --- 수행 7: Clustering Statistics ---
    print("\n\n=== Clustering Statistics ===")
    stats_rows = []
    for method in METHODS:
        stats = compute_clustering_stats(all_results[method]["X_reduced"],
                                         all_results[method]["y"])
        stats_rows.append({"Method": method, **stats})
        print(f"  {method}: Sil={stats['silhouette']:.4f}, CH={stats['calinski_harabasz']:.1f}, "
              f"DB={stats['davies_bouldin']:.4f}")
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "clustering_stats.csv"), index=False)
    print(f"\n결과 저장: {OUTPUT_DIR}")
