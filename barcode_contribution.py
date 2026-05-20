"""
Contribution of Individual Barcodes in Six-Pack (Sixpack_Rips)
61200D raw PI 벡터 기반 — preprocessing 없음

Six-pack barcode types (per direction):
  image      : Im(f*)        H0(100D) + H1(5000D) = 5100D
  kernel     : Ker(f*)       H0(100D) + H1(5000D) = 5100D
  cokernel   : Cok(f*)       H0(100D) + H1(5000D) = 5100D
  sub_complex: H*(L)=domain  H0(100D) + H1(5000D) = 5100D
  complex    : H*(K)=codomain H0(100D) + H1(5000D) = 5100D
  relative   : H*(K,L)       H0(100D) + H1(5000D) = 5100D

2 directions × 1 barcode × 5100D = 10200D per barcode
2 directions × 6 barcodes × 5100D = 61200D full
"""
import os, glob, warnings, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
warnings.filterwarnings('ignore')

BASE     = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"
RIPS_DIR = os.path.join(BASE, "Final_Vector", "Sixpack_Rips")

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

# ── 데이터 로딩: 선택한 barcode key만 flatten ─────────────────────────────────
ALL_KEYS = ["image", "kernel", "cokernel", "sub_complex", "complex", "relative"]
ALIAS    = {"sub_complex": "domain", "complex": "codomain"}

def load_selected(selected_keys):
    """선택된 barcode key들의 raw PI 벡터만 flatten (preprocessing 없음)."""
    files = sorted(glob.glob(os.path.join(RIPS_DIR, "Sixpack_Rips_*.npz")))
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

# ── 평가: StandardScaler → PCA(선택) → classifier ────────────────────────────
from sklearn.decomposition import PCA

SKF  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X, y, pca_dim=100):
    X_s = StandardScaler().fit_transform(X)
    if pca_dim and X_s.shape[1] > pca_dim:
        X_s = PCA(n_components=pca_dim, random_state=42).fit_transform(X_s)
    results = {}
    clfs = {
        "LinearSVM": LinearSVC(C=1, max_iter=5000, dual=True),
        "RF(200)":   RandomForestClassifier(200, random_state=42, n_jobs=-1),
    }
    for name, clf_tmpl in clfs.items():
        soft_l, strict_l = [], []
        for tr, te in SKF.split(X_s, y):
            c = clone(clf_tmpl); c.fit(X_s[tr], y[tr])
            yp = c.predict(X_s[te])
            soft_l.append(soft_acc(y[te], yp))
            strict_l.append(accuracy_score(y[te], yp))
        results[name] = (np.mean(soft_l)*100, np.mean(strict_l)*100)
    return results

SEP = "=" * 70

# ── 실험 1. 각 barcode 단독 (10200D → PCA100) ────────────────────────────────
print(SEP)
print("실험 1. 단독 barcode 기여도  (10200D raw → PCA100)")
print(f"  {'Barcode':<14}  {'LinearSVM Soft':>15}  {'LinearSVM Strict':>17}  {'RF Soft':>9}  {'RF Strict':>10}")
print("  " + "-"*68)

individual = {}
for k in ALL_KEYS:
    X, y = load_selected([k])
    res = evaluate(X, y, pca_dim=100)
    alias = ALIAS.get(k, k)
    individual[k] = res
    print(f"  {alias:<14}  {res['LinearSVM'][0]:>14.2f}%  {res['LinearSVM'][1]:>16.2f}%"
          f"  {res['RF(200)'][0]:>8.2f}%  {res['RF(200)'][1]:>9.2f}%")

# ── 실험 2. Full six-pack (61200D → PCA100) ───────────────────────────────────
print(f"\n{SEP}")
print("실험 2. Full six-pack  (61200D raw → PCA100)")
X_full, y_full = load_selected(ALL_KEYS)
res_full = evaluate(X_full, y_full, pca_dim=100)
for clf, (s, st) in res_full.items():
    print(f"  Full (all 6)  [{clf}]  Soft={s:.2f}%  Strict={st:.2f}%")

# ── 실험 3. Leave-one-out ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("실험 3. Leave-one-out  (50200D raw → PCA100)")
print(f"  {'Removed':<14}  {'LIN ΔSoft':>10}  {'LIN ΔStrict':>12}  {'RF ΔSoft':>9}  {'RF ΔStrict':>10}")
print("  " + "-"*60)

loo = {}
for k in ALL_KEYS:
    remaining = [x for x in ALL_KEYS if x != k]
    X, y = load_selected(remaining)
    res = evaluate(X, y, pca_dim=100)
    alias = ALIAS.get(k, k)
    ds_lin  = res['LinearSVM'][0] - res_full['LinearSVM'][0]
    dst_lin = res['LinearSVM'][1] - res_full['LinearSVM'][1]
    ds_rf   = res['RF(200)'][0]   - res_full['RF(200)'][0]
    dst_rf  = res['RF(200)'][1]   - res_full['RF(200)'][1]
    loo[k] = (ds_lin, dst_lin, ds_rf, dst_rf)
    print(f"  w/o {alias:<10}  {ds_lin:>+9.2f}%p  {dst_lin:>+11.2f}%p  {ds_rf:>+8.2f}%p  {dst_rf:>+9.2f}%p")

# ── 실험 4. 의미 기반 그룹핑 ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("실험 4. 의미 기반 그룹핑  (raw → PCA100)")
print(f"  {'Group':<38}  {'Dim':>6}  {'LIN Soft':>9}  {'RF Soft':>8}  {'RF Strict':>10}")
print("  " + "-"*75)

groups = {
    "Ordinary (domain+codomain)":        ["sub_complex", "complex"],
    "Six-pack core (img+ker+cok)":        ["image", "kernel", "cokernel"],
    "Relative only":                       ["relative"],
    "IKC + relative":                      ["image","kernel","cokernel","relative"],
    "Ordinary + relative":                 ["sub_complex","complex","relative"],
    "IKC + ordinary (w/o relative)":      ["image","kernel","cokernel","sub_complex","complex"],
    "Full (all 6)":                        ALL_KEYS,
}
for name, keys in groups.items():
    X, y = load_selected(keys)
    res = evaluate(X, y, pca_dim=100)
    dim = X.shape[1]
    print(f"  {name:<38}  {dim:>6}  {res['LinearSVM'][0]:>8.2f}%  {res['RF(200)'][0]:>7.2f}%  {res['RF(200)'][1]:>9.2f}%")

# ── 실험 5. Pairwise (2개씩, RF Soft 기준 정렬) ───────────────────────────────
print(f"\n{SEP}")
print("실험 5. Pairwise combinations  (raw → PCA100, RF Soft 기준)")
print(f"  {'Combination':<30}  {'LIN Soft':>9}  {'RF Soft':>8}  {'RF Strict':>10}")
print("  " + "-"*62)

pairs = []
for combo in itertools.combinations(ALL_KEYS, 2):
    X, y = load_selected(list(combo))
    res = evaluate(X, y, pca_dim=100)
    name = " + ".join(ALIAS.get(k,k) for k in combo)
    pairs.append((name, res['LinearSVM'][0], res['RF(200)'][0], res['RF(200)'][1]))

for name, s_lin, s_rf, st_rf in sorted(pairs, key=lambda x: -x[2]):
    print(f"  {name:<30}  {s_lin:>8.2f}%  {s_rf:>7.2f}%  {st_rf:>9.2f}%")

# ── 요약 ─────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("요약: 단독 기여도 순위 (RF Soft 기준)")
ranked = sorted(individual.items(), key=lambda x: -x[1]['RF(200)'][0])
for rank, (k, res) in enumerate(ranked, 1):
    alias = ALIAS.get(k, k)
    print(f"  {rank}. {alias:<12}  RF Soft={res['RF(200)'][0]:.2f}%  RF Strict={res['RF(200)'][1]:.2f}%"
          f"  | LIN Soft={res['LinearSVM'][0]:.2f}%")

print(f"\n요약: Leave-one-out — 제거 시 RF Soft 기준 손해 순")
ranked_loo = sorted(loo.items(), key=lambda x: x[1][2])
for k, (ds_lin, dst_lin, ds_rf, dst_rf) in ranked_loo:
    alias = ALIAS.get(k, k)
    print(f"  w/o {alias:<12}  RF ΔSoft={ds_rf:+.2f}%p  RF ΔStrict={dst_rf:+.2f}%p"
          f"  | LIN ΔSoft={ds_lin:+.2f}%p")
