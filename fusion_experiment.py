"""
Fusion Experiment: Ord_PI × Inter_PI / Ord_PI × 3D_PI
Early / Intermediate / Late fusion
"""

import os, glob, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
warnings.filterwarnings('ignore')

BASE = "/Users/hjxarchive/Multi-Chromatic-Spatial-Pattern-Classification"

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
    return GT[(idx % 64) // 8][idx // 64][idx % 8]

def build_adj(M_list):
    adj = {}
    for M in M_list:
        M = np.array(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                cur = int(M[i, j])
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < M.shape[0] and 0 <= nj < M.shape[1]:
                        nb = int(M[ni, nj])
                        if nb != cur:
                            adj.setdefault(cur, [])
                            if nb not in adj[cur]: adj[cur].append(nb)
    return adj

ADJ = build_adj(GT)
CLASSES = np.array(sorted(set(GT.flatten().tolist())))  # fixed class order for proba alignment

def soft_acc(yt, yp):
    return np.mean([t == p or p in ADJ.get(t, []) or t in ADJ.get(p, [])
                    for t, p in zip(yt, yp)])

# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_all(prefix, data_dir):
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

print("Loading Ord_PI ...")
X_ord,   y = load_all('Ord_PI',   os.path.join(BASE, 'Final_Vector', 'Ord_PI'))
print("Loading Inter_PI ...")
X_inter, _ = load_all('Inter_PI', os.path.join(BASE, 'Final_Vector', 'Inter_PI'))
print("Loading 3D_PI ...")
X_3d,    _ = load_all('3D_PI',    os.path.join(BASE, 'Final_Vector', '3D_PI'))
print(f"Loaded: Ord={X_ord.shape}, Inter={X_inter.shape}, 3D={X_3d.shape}, y={y.shape}\n")

# ── 공통 설정 ─────────────────────────────────────────────────────────────────
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
CLFS = {
    'LIN_hard': LinearSVC(C=1e5, max_iter=10000, dual=True),   # hard margin
    'LIN_soft': SVC(kernel='linear', C=1.0),                   # benchmark 기준 soft margin
    'RBF':      SVC(kernel='rbf', C=1.0),
    'RF200':    RandomForestClassifier(200, random_state=42, n_jobs=-1),
}

# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def evaluate_single(X, y, pca_dim):
    """단일 descriptor: StandardScaler → PCA → clf"""
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

# ── Early Fusion ──────────────────────────────────────────────────────────────
def early_fusion(X_a, X_b, y, pca_dim):
    """concat → StandardScaler → PCA(k) → clf"""
    X = np.hstack([X_a, X_b])
    return evaluate_single(X, y, pca_dim)

# ── Intermediate Fusion ───────────────────────────────────────────────────────
_CLIP = 1e6  # SS 이후 clip: std≈0 feature가 테스트셋에서 1e79까지 폭발하는 현상 방지

def _safe_pca_transform(pca, ss, X_tr, X_te):
    Xs_tr = np.clip(ss.transform(X_tr), -_CLIP, _CLIP)
    Xs_te = np.clip(ss.transform(X_te), -_CLIP, _CLIP)
    Ztr = pca.transform(Xs_tr)
    Zte = pca.transform(Xs_te)
    return Ztr, Zte

def inter_fusion(X_a, X_b, y, pca_a, pca_b):
    """각 descriptor를 SS+PCA(k) 후 concat embedding → clf (fold 내부에서 fit)"""
    results = {}
    for clf_name, clf_tmpl in CLFS.items():
        soft_l, strict_l = [], []
        for tr, te in SKF.split(X_a, y):
            # descriptor A
            ss_a = StandardScaler().fit(X_a[tr])
            pca_obj_a = PCA(pca_a, random_state=42).fit(np.clip(ss_a.transform(X_a[tr]), -_CLIP, _CLIP))
            Za_tr, Za_te = _safe_pca_transform(pca_obj_a, ss_a, X_a[tr], X_a[te])
            # descriptor B
            ss_b = StandardScaler().fit(X_b[tr])
            pca_obj_b = PCA(pca_b, random_state=42).fit(np.clip(ss_b.transform(X_b[tr]), -_CLIP, _CLIP))
            Zb_tr, Zb_te = _safe_pca_transform(pca_obj_b, ss_b, X_b[tr], X_b[te])
            # concat embedding → clf
            Z_tr = np.hstack([Za_tr, Zb_tr])
            Z_te = np.hstack([Za_te, Zb_te])
            c = clone(clf_tmpl)
            c.fit(Z_tr, y[tr])
            yp = c.predict(Z_te)
            soft_l.append(soft_acc(y[te], yp))
            strict_l.append(accuracy_score(y[te], yp))
        results[clf_name] = (np.mean(soft_l)*100, np.mean(strict_l)*100)
    return results

# ── Weighted Sum: Embedding level ────────────────────────────────────────────
def embedding_weighted_sum(X_a, X_b, y, pca_dim):
    """
    PCA(k) each → SS normalize → w·Z_a + (1-w)·Z_b → classifier
    w 최적화: outer fold마다 inner 4-fold CV (w ∈ 0.0~1.0, step 0.1)
    """
    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    W_GRID = np.linspace(0.0, 1.0, 11)
    results = {}

    for clf_name, clf_tmpl in CLFS.items():
        soft_l, strict_l, best_ws = [], [], []
        for outer_tr, outer_te in SKF.split(X_a, y):
            Xa_tr, Xa_te = X_a[outer_tr], X_a[outer_te]
            Xb_tr, Xb_te = X_b[outer_tr], X_b[outer_te]
            y_tr, y_te   = y[outer_tr],   y[outer_te]

            # 각 descriptor를 PCA로 압축 (outer train 기준)
            ss_a  = StandardScaler().fit(Xa_tr)
            pca_a = PCA(pca_dim, random_state=42).fit(np.clip(ss_a.transform(Xa_tr), -_CLIP, _CLIP))
            ss_b  = StandardScaler().fit(Xb_tr)
            pca_b = PCA(pca_dim, random_state=42).fit(np.clip(ss_b.transform(Xb_tr), -_CLIP, _CLIP))

            Za_tr, Za_te = _safe_pca_transform(pca_a, ss_a, Xa_tr, Xa_te)
            Zb_tr, Zb_te = _safe_pca_transform(pca_b, ss_b, Xb_tr, Xb_te)

            # embedding을 unit variance로 정규화 (가중합 전)
            norm_a = StandardScaler().fit(Za_tr)
            norm_b = StandardScaler().fit(Zb_tr)
            Na_tr = norm_a.transform(Za_tr); Na_te = norm_a.transform(Za_te)
            Nb_tr = norm_b.transform(Zb_tr); Nb_te = norm_b.transform(Zb_te)

            # inner CV로 최적 w 탐색
            best_w, best_val = 0.5, -1.0
            for w in W_GRID:
                val_softs = []
                for in_tr, in_val in inner_skf.split(Na_tr, y_tr):
                    Z_in  = w * Na_tr[in_tr]  + (1-w) * Nb_tr[in_tr]
                    Z_val = w * Na_tr[in_val] + (1-w) * Nb_tr[in_val]
                    c = clone(clf_tmpl)
                    c.fit(Z_in, y_tr[in_tr])
                    val_softs.append(soft_acc(y_tr[in_val], c.predict(Z_val)))
                score = np.mean(val_softs)
                if score > best_val:
                    best_val, best_w = score, w

            # 최적 w로 test 예측
            Z_tr = best_w * Na_tr + (1-best_w) * Nb_tr
            Z_te = best_w * Na_te + (1-best_w) * Nb_te
            c = clone(clf_tmpl)
            c.fit(Z_tr, y_tr)
            yp = c.predict(Z_te)
            soft_l.append(soft_acc(y_te, yp))
            strict_l.append(accuracy_score(y_te, yp))
            best_ws.append(best_w)

        results[clf_name] = (np.mean(soft_l)*100, np.mean(strict_l)*100)
        results[f'{clf_name}_w'] = np.mean(best_ws)
    return results

# ── Weighted Sum: Late level ──────────────────────────────────────────────────
def late_weighted_vote(X_a, X_b, y, pca_dim, clf_tmpl):
    """
    w·proba_a + (1-w)·proba_b → argmax
    w 최적화: outer fold마다 inner 4-fold CV (w ∈ 0.0~1.0, step 0.1)
    """
    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    W_GRID = np.linspace(0.0, 1.0, 11)
    soft_l, strict_l, best_ws = [], [], []

    for outer_tr, outer_te in SKF.split(X_a, y):
        Xa_tr, Xa_te = X_a[outer_tr], X_a[outer_te]
        Xb_tr, Xb_te = X_b[outer_tr], X_b[outer_te]
        y_tr, y_te   = y[outer_tr],   y[outer_te]

        # outer train 기준 PCA
        def _fit_modal(X_tr, X_te):
            ss  = StandardScaler().fit(X_tr)
            pca = PCA(pca_dim, random_state=42).fit(np.clip(ss.transform(X_tr), -_CLIP, _CLIP))
            return _safe_pca_transform(pca, ss, X_tr, X_te)

        Za_tr, Za_te = _fit_modal(Xa_tr, Xa_te)
        Zb_tr, Zb_te = _fit_modal(Xb_tr, Xb_te)

        def _get_proba(clf, Ztr, ytr, Zte):
            c = clone(clf)
            c.fit(Ztr, ytr)
            proba = np.zeros((len(Zte), len(CLASSES)))
            for ci, cls in enumerate(c.classes_):
                col = np.where(CLASSES == cls)[0][0]
                proba[:, col] = c.predict_proba(Zte)[:, ci]
            return proba

        # inner CV로 최적 w 탐색
        best_w, best_val = 0.5, -1.0
        for w in W_GRID:
            val_softs = []
            for in_tr, in_val in inner_skf.split(Za_tr, y_tr):
                pa = _get_proba(clf_tmpl, Za_tr[in_tr], y_tr[in_tr], Za_tr[in_val])
                pb = _get_proba(clf_tmpl, Zb_tr[in_tr], y_tr[in_tr], Zb_tr[in_val])
                yp = CLASSES[np.argmax(w*pa + (1-w)*pb, axis=1)]
                val_softs.append(soft_acc(y_tr[in_val], yp))
            score = np.mean(val_softs)
            if score > best_val:
                best_val, best_w = score, w

        # 최적 w로 test 예측
        pa_te = _get_proba(clf_tmpl, Za_tr, y_tr, Za_te)
        pb_te = _get_proba(clf_tmpl, Zb_tr, y_tr, Zb_te)
        yp = CLASSES[np.argmax(best_w*pa_te + (1-best_w)*pb_te, axis=1)]
        soft_l.append(soft_acc(y_te, yp))
        strict_l.append(accuracy_score(y_te, yp))
        best_ws.append(best_w)

    return np.mean(soft_l)*100, np.mean(strict_l)*100, np.mean(best_ws)

# ── Late Fusion: Soft Voting ──────────────────────────────────────────────────
def late_soft_vote(X_a, X_b, y, pca_dim, clf_tmpl):
    """RF / KNN: predict_proba 평균 → argmax (fold 내부에서 fit)"""
    soft_l, strict_l = [], []
    for tr, te in SKF.split(X_a, y):
        preds = []
        for X in [X_a, X_b]:
            ss = StandardScaler().fit(X[tr])
            pca = PCA(pca_dim, random_state=42).fit(np.clip(ss.transform(X[tr]), -_CLIP, _CLIP))
            Ztr, Zte = _safe_pca_transform(pca, ss, X[tr], X[te])
            c = clone(clf_tmpl)
            c.fit(Ztr, y[tr])
            # align proba to fixed CLASSES order
            proba = np.zeros((len(te), len(CLASSES)))
            for ci, cls in enumerate(c.classes_):
                col = np.where(CLASSES == cls)[0][0]
                proba[:, col] = c.predict_proba(Zte)[:, ci]
            preds.append(proba)
        avg_proba = np.mean(preds, axis=0)
        yp = CLASSES[np.argmax(avg_proba, axis=1)]
        soft_l.append(soft_acc(y[te], yp))
        strict_l.append(accuracy_score(y[te], yp))
    return np.mean(soft_l)*100, np.mean(strict_l)*100

# ── Late Fusion: Stacking ─────────────────────────────────────────────────────
def stacking(X_a, X_b, y, pca_dim):
    """
    Outer 5-fold.  Inner 4-fold OOF → meta-features.
    Base: RF(200).  Meta: RF(200).
    """
    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    n_cls = len(CLASSES)
    soft_l, strict_l = [], []

    for outer_tr, outer_te in SKF.split(X_a, y):
        Xa_tr, Xa_te = X_a[outer_tr], X_a[outer_te]
        Xb_tr, Xb_te = X_b[outer_tr], X_b[outer_te]
        y_tr, y_te   = y[outer_tr],   y[outer_te]

        # OOF meta-features for outer train
        oof_meta = np.zeros((len(outer_tr), n_cls * 2))
        rf_base = RandomForestClassifier(200, random_state=42, n_jobs=-1)

        for in_tr, in_val in inner_skf.split(Xa_tr, y_tr):
            for xi, (X_full, col_start) in enumerate([(Xa_tr, 0), (Xb_tr, n_cls)]):
                ss  = StandardScaler().fit(X_full[in_tr])
                pca = PCA(pca_dim, random_state=42).fit(np.clip(ss.transform(X_full[in_tr]), -_CLIP, _CLIP))
                Ztr, Zval = _safe_pca_transform(pca, ss, X_full[in_tr], X_full[in_val])
                c = clone(rf_base)
                c.fit(Ztr, y_tr[in_tr])
                proba = np.zeros((len(in_val), n_cls))
                for ci, cls in enumerate(c.classes_):
                    col = np.where(CLASSES == cls)[0][0]
                    proba[:, col] = c.predict_proba(Zval)[:, ci]
                oof_meta[in_val, col_start:col_start+n_cls] = proba

        # meta-clf trained on OOF
        meta_clf = RandomForestClassifier(200, random_state=42, n_jobs=-1)
        meta_clf.fit(oof_meta, y_tr)

        # test meta-features: base clfs trained on full outer train
        test_meta = np.zeros((len(outer_te), n_cls * 2))
        for xi, (X_full_tr, X_full_te, col_start) in enumerate(
                [(Xa_tr, Xa_te, 0), (Xb_tr, Xb_te, n_cls)]):
            ss  = StandardScaler().fit(X_full_tr)
            pca = PCA(pca_dim, random_state=42).fit(np.clip(ss.transform(X_full_tr), -_CLIP, _CLIP))
            Ztr, Zte = _safe_pca_transform(pca, ss, X_full_tr, X_full_te)
            c = clone(rf_base)
            c.fit(Ztr, y_tr)
            proba = np.zeros((len(outer_te), n_cls))
            for ci, cls in enumerate(c.classes_):
                col = np.where(CLASSES == cls)[0][0]
                proba[:, col] = c.predict_proba(Zte)[:, ci]
            test_meta[:, col_start:col_start+n_cls] = proba

        yp = meta_clf.predict(test_meta)
        soft_l.append(soft_acc(y_te, yp))
        strict_l.append(accuracy_score(y_te, yp))

    return np.mean(soft_l)*100, np.mean(strict_l)*100

# ── 실험 실행 ─────────────────────────────────────────────────────────────────
rows = []

def log(label, res_dict):
    row = {'label': label}
    for clf, (soft, strict) in res_dict.items():
        row[f'{clf}_soft']   = soft
        row[f'{clf}_strict'] = strict
    rows.append(row)
    lh  = res_dict.get('LIN_hard', (None, None))
    ls  = res_dict.get('LIN_soft', (None, None))
    rbf = res_dict.get('RBF',      (None, None))
    rf  = res_dict.get('RF200',    (None, None))
    print(f"  {label:<45}  LH={lh[0]:.2f}%/{lh[1]:.2f}%  LS={ls[0]:.2f}%/{ls[1]:.2f}%  RBF={rbf[0]:.2f}%/{rbf[1]:.2f}%  RF={rf[0]:.2f}%/{rf[1]:.2f}%")

def log2(label, soft, strict, extra=''):
    rows.append({'label': label, 'RF200_soft': soft, 'RF200_strict': strict})
    print(f"  {label:<45}  RF={soft:.2f}%/{strict:.2f}%{extra}")

def log_ws(label, res_dict):
    """Embedding weighted sum 전용 로그 (w 평균 포함)"""
    row = {'label': label}
    for clf, val in res_dict.items():
        if isinstance(val, tuple):
            row[f'{clf}_soft'], row[f'{clf}_strict'] = val
        else:
            row[f'{clf}_w'] = val
    rows.append(row)
    lh  = res_dict.get('LIN_hard', (None,None)); lhw = res_dict.get('LIN_hard_w', 0.5)
    ls  = res_dict.get('LIN_soft', (None,None)); lsw = res_dict.get('LIN_soft_w', 0.5)
    rbf = res_dict.get('RBF',      (None,None)); rw  = res_dict.get('RBF_w',      0.5)
    rf  = res_dict.get('RF200',    (None,None)); fw  = res_dict.get('RF200_w',     0.5)
    print(f"  {label:<45}  LH={lh[0]:.2f}%/{lh[1]:.2f}%(w={lhw:.1f})"
          f"  LS={ls[0]:.2f}%/{ls[1]:.2f}%(w={lsw:.1f})"
          f"  RBF={rbf[0]:.2f}%/{rbf[1]:.2f}%(w={rw:.1f})"
          f"  RF={rf[0]:.2f}%/{rf[1]:.2f}%(w={fw:.1f})")

print("=" * 75)
print("BASELINE (단일 Descriptor, PCA 100)")
print("=" * 75)
log("Ord_PI   [baseline]",   evaluate_single(X_ord,   y, 100))
log("Inter_PI [baseline]",   evaluate_single(X_inter, y, 100))
log("3D_PI    [baseline]",   evaluate_single(X_3d,    y, 100))

print()
print("=" * 75)
print("EARLY FUSION  (concat → PCA(k) → clf)")
print("=" * 75)
log("E1  [Ord+Inter] PCA100",  early_fusion(X_ord, X_inter, y, 100))
log("E2  [Ord+3D]    PCA100",  early_fusion(X_ord, X_3d,   y, 100))
log("E3  [Ord+Inter] PCA200",  early_fusion(X_ord, X_inter, y, 200))
log("E4  [Ord+3D]    PCA200",  early_fusion(X_ord, X_3d,   y, 200))

print()
print("=" * 75)
print("INTERMEDIATE FUSION  (PCA each → concat embedding → clf)")
print("=" * 75)
log("I1  [Ord+Inter] PCA50+50 →100D",    inter_fusion(X_ord, X_inter, y, 50,  50))
log("I2  [Ord+3D]    PCA50+50 →100D",    inter_fusion(X_ord, X_3d,   y, 50,  50))
log("I3  [Ord+Inter] PCA100+100 →200D",  inter_fusion(X_ord, X_inter, y, 100, 100))
log("I4  [Ord+3D]    PCA100+100 →200D",  inter_fusion(X_ord, X_3d,   y, 100, 100))

print()
print("=" * 75)
print("LATE FUSION — Soft Voting  (predict_proba 평균, PCA100 per modal)")
print("=" * 75)
rf_tmpl   = RandomForestClassifier(200, random_state=42, n_jobs=-1)
rbf_tmpl  = SVC(kernel='rbf',    C=1.0, probability=True)
lins_tmpl = SVC(kernel='linear', C=1.0, probability=True)

s, st = late_soft_vote(X_ord, X_inter, y, 100, rf_tmpl)
log2("L1  [Ord+Inter] RF       soft-vote", s, st)
s, st = late_soft_vote(X_ord, X_3d,   y, 100, rf_tmpl)
log2("L2  [Ord+3D]    RF       soft-vote", s, st)
s, st = late_soft_vote(X_ord, X_inter, y, 100, rbf_tmpl)
log2("L3  [Ord+Inter] RBF      soft-vote", s, st)
s, st = late_soft_vote(X_ord, X_3d,   y, 100, rbf_tmpl)
log2("L4  [Ord+3D]    RBF      soft-vote", s, st)
s, st = late_soft_vote(X_ord, X_inter, y, 100, lins_tmpl)
log2("L5  [Ord+Inter] LIN_soft soft-vote", s, st)
s, st = late_soft_vote(X_ord, X_3d,   y, 100, lins_tmpl)
log2("L6  [Ord+3D]    LIN_soft soft-vote", s, st)

print()
print("=" * 75)
print("LATE FUSION — Stacking  (RF base × 2 → RF meta, OOF, PCA100 per modal)")
print("=" * 75)
s, st = stacking(X_ord, X_inter, y, 100)
log2("L7  [Ord+Inter] Stacking RF→RF", s, st)
s, st = stacking(X_ord, X_3d,   y, 100)
log2("L8  [Ord+3D]    Stacking RF→RF", s, st)

print()
print("=" * 75)
print("WEIGHTED SUM — Embedding level  (PCA100 each → norm → w·Za+(1-w)·Zb)")
print("inner CV로 w 최적화 (w ∈ {0.0, 0.1, ..., 1.0})")
print("=" * 75)
log_ws("W1  [Ord+Inter] Embed WS PCA100", embedding_weighted_sum(X_ord, X_inter, y, 100))
log_ws("W2  [Ord+3D]    Embed WS PCA100", embedding_weighted_sum(X_ord, X_3d,   y, 100))

print()
print("=" * 75)
print("WEIGHTED SUM — Late level  (w·proba_a + (1-w)·proba_b, PCA100 per modal)")
print("inner CV로 w 최적화 (w ∈ {0.0, 0.1, ..., 1.0})")
print("=" * 75)
rbf_p  = SVC(kernel='rbf',    C=1.0, probability=True)
lins_p = SVC(kernel='linear', C=1.0, probability=True)
s, st, bw = late_weighted_vote(X_ord, X_inter, y, 100, rf_tmpl)
log2("W3  [Ord+Inter] Late WS RF",       s, st, f"  (avg w={bw:.2f})")
s, st, bw = late_weighted_vote(X_ord, X_3d,   y, 100, rf_tmpl)
log2("W4  [Ord+3D]    Late WS RF",       s, st, f"  (avg w={bw:.2f})")
s, st, bw = late_weighted_vote(X_ord, X_inter, y, 100, rbf_p)
log2("W5  [Ord+Inter] Late WS RBF",      s, st, f"  (avg w={bw:.2f})")
s, st, bw = late_weighted_vote(X_ord, X_3d,   y, 100, rbf_p)
log2("W6  [Ord+3D]    Late WS RBF",      s, st, f"  (avg w={bw:.2f})")
s, st, bw = late_weighted_vote(X_ord, X_inter, y, 100, lins_p)
log2("W7  [Ord+Inter] Late WS LIN_soft", s, st, f"  (avg w={bw:.2f})")
s, st, bw = late_weighted_vote(X_ord, X_3d,   y, 100, lins_p)
log2("W8  [Ord+3D]    Late WS LIN_soft", s, st, f"  (avg w={bw:.2f})")

# ── CSV 저장 ──────────────────────────────────────────────────────────────────
import csv
out_csv = os.path.join(BASE, "Final_Results(0521)", "fusion_experiment.csv")
all_keys = ['label',
            'LIN_hard_soft','LIN_hard_strict',
            'LIN_soft_soft','LIN_soft_strict',
            'RBF_soft','RBF_strict',
            'RF200_soft','RF200_strict']
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
    w.writeheader()
    for row in rows:
        w.writerow({k: (f"{row[k]:.4f}" if isinstance(row.get(k), float) else row.get(k, ''))
                    for k in all_keys})
print(f"\nCSV 저장: {out_csv}")
