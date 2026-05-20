"""
H1/H2/H3 가설 검증 실험 — Rips vs Chroma Embedding 구조 분석
Google Colab 복붙용. pipeline.py + load_sixpack_rips_correct 실행 후 사용.
"""

# ============================================================
# 실험 1: Performance–Dimension Scaling Curve (H1)
# ============================================================
print("=" * 80)
print("실험 1: Performance–Dimension Scaling Curve (H1)")
print("=" * 80)

from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

DIMS_SWEEP = [16, 32, 64, 128, 256, 512, 1024]
ALL_5 = ["Ord_PI", "Inter_PI", "3D_PI", "Sixpack_Rips", "Sixpack_Chroma"]
SEEDS = [42, 123, 456, 789, 1010]
COLORS = {"Ord_PI":"#4C72B0","Inter_PI":"#DD8452","3D_PI":"#55A868",
          "Sixpack_Rips":"#C44E52","Sixpack_Chroma":"#8172B3"}

def eval_at_dim(X, y, pca_dim, clf, seed=42):
    X_s = StandardScaler().fit_transform(X)
    if pca_dim is not None and X_s.shape[1] > pca_dim:
        X_s = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_s)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tri, tei in skf.split(X_s, y):
        c = clone(clf); c.fit(X_s[tri], y[tri])
        accs.append(accuracy_score(y[tei], c.predict(X_s[tei])))
    return np.mean(accs) * 100

# (A) SVM-L + (B) Best classifier
svml = SVC(kernel="linear", C=1.0)
best_clfs = {"Ord_PI": RandomForestClassifier(100, random_state=42),
             "Inter_PI": RandomForestClassifier(100, random_state=42),
             "3D_PI": RandomForestClassifier(100, random_state=42),
             "Sixpack_Rips": SVC(kernel="rbf", C=1.0, gamma="scale"),
             "Sixpack_Chroma": SVC(kernel="rbf", C=1.0, gamma="scale")}

exp1 = {m: {"svml": {}, "best": {}} for m in ALL_5}

for method in ALL_5:
    X, y = datasets[method]["X"], datasets[method]["y"]
    print(f"\n[{method}] (원본 dim={X.shape[1]})")
    for dim in DIMS_SWEEP:
        if dim > X.shape[1]: continue
        svml_scores = [eval_at_dim(X, y, dim, svml, s) for s in SEEDS]
        best_scores = [eval_at_dim(X, y, dim, best_clfs[method], s) for s in SEEDS]
        exp1[method]["svml"][dim] = svml_scores
        exp1[method]["best"][dim] = best_scores
        print(f"  D={dim:<5d} SVM-L={np.mean(svml_scores):.2f}±{np.std(svml_scores):.2f}  "
              f"Best={np.mean(best_scores):.2f}±{np.std(best_scores):.2f}")

# Saturating curve fit: Acc(D) = a - b * D^(-c)
def sat_func(D, a, b, c):
    return a - b * np.power(D, -c)

print("\n--- Saturation Fit ---")
for method in ALL_5:
    dims = sorted(exp1[method]["svml"].keys())
    means = [np.mean(exp1[method]["svml"][d]) for d in dims]
    try:
        popt, _ = curve_fit(sat_func, dims, means, p0=[80, 100, 0.5], maxfev=5000)
        print(f"  {method:<18s} a(asymptote)={popt[0]:.2f}%, c(rate)={popt[2]:.4f}")
    except:
        print(f"  {method:<18s} fit failed")

# Figure 1: Performance-Dimension Curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, mode, title in [(axes[0], "svml", "(A) SVM-Linear"),
                         (axes[1], "best", "(B) Best Classifier")]:
    for m in ALL_5:
        dims = sorted(exp1[m][mode].keys())
        means = [np.mean(exp1[m][mode][d]) for d in dims]
        stds = [np.std(exp1[m][mode][d]) for d in dims]
        ax.errorbar(dims, means, yerr=stds, fmt="o-", label=m,
                    color=COLORS[m], linewidth=2, markersize=6, capsize=3)
    ax.set_xlabel("Embedding Dimension D", fontsize=12)
    ax.set_ylabel("Strict Accuracy (%)", fontsize=12)
    ax.set_title(f"Exp 1 {title}: Acc vs Dim", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9); ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exp1_perf_dim_curve.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 실험 2: Principal Component 구조 분석 (H2a)
# ============================================================
print("\n" + "=" * 80)
print("실험 2: Principal Component 구조 분석 (H2a)")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 2-1: Eigenvalue spectrum
ax = axes[0]
d_effs = {}
for m in ALL_5:
    X = StandardScaler().fit_transform(datasets[m]["X"])
    pca = PCA(n_components=min(500, X.shape[1]), random_state=42).fit(X)
    evals = pca.explained_variance_[:min(300, len(pca.explained_variance_))]
    evals_norm = evals / evals[0]
    ax.plot(range(1, len(evals_norm)+1), evals_norm, label=m, color=COLORS[m], linewidth=1.5)
    # Participation ratio
    lam = pca.explained_variance_
    d_eff = (np.sum(lam)**2) / np.sum(lam**2)
    d_effs[m] = d_eff
ax.set_xlabel("PC Index"); ax.set_ylabel("Normalized Eigenvalue")
ax.set_title("Eigenvalue Spectrum", fontweight="bold")
ax.set_yscale("log"); ax.set_xscale("log"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2-2: Participation Ratio (d_eff) bar chart
ax = axes[1]
names = list(d_effs.keys())
vals = [d_effs[n] for n in names]
bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names])
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("d_eff (Participation Ratio)"); ax.set_title("Effective Dimensionality", fontweight="bold")
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# 2-3: Top-k PC Accuracy Curve (핵심!)
ax = axes[2]
TOP_K = [5, 10, 20, 50, 100, 200, 500]
for m in ALL_5:
    X = StandardScaler().fit_transform(datasets[m]["X"])
    y = datasets[m]["y"]
    pca = PCA(n_components=min(500, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X)
    accs = []
    ks_valid = []
    for k in TOP_K:
        if k > X_pca.shape[1]: continue
        ks_valid.append(k)
        X_k = X_pca[:, :k]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_acc = []
        for tri, tei in skf.split(X_k, y):
            clf = SVC(kernel="linear", C=1.0); clf.fit(X_k[tri], y[tri])
            fold_acc.append(accuracy_score(y[tei], clf.predict(X_k[tei])))
        accs.append(np.mean(fold_acc) * 100)
    ax.plot(ks_valid, accs, "o-", label=m, color=COLORS[m], linewidth=2, markersize=6)
ax.set_xlabel("Top-k PCs"); ax.set_ylabel("Strict Accuracy (%)")
ax.set_title("Top-k PC Accuracy (SVM-L)", fontweight="bold")
ax.set_xscale("log", base=2); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exp2_pc_structure.png"), dpi=150, bbox_inches="tight")
plt.show()

# 요약 테이블
print("\n--- PC 구조 요약 ---")
print(f"{'Method':<18s} {'d_eff':<10s} {'90%Var@PC':<12s} {'95%Var@PC':<12s} {'99%Var@PC':<12s}")
print("-" * 64)
for m in ALL_5:
    X = StandardScaler().fit_transform(datasets[m]["X"])
    pca = PCA(n_components=min(500, X.shape[1]), random_state=42).fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    pc90 = np.searchsorted(cumvar, 90) + 1
    pc95 = np.searchsorted(cumvar, 95) + 1
    pc99 = np.searchsorted(cumvar, 99) + 1
    print(f"{m:<18s} {d_effs[m]:<10.1f} {pc90:<12d} {pc95:<12d} {pc99:<12d}")

# ============================================================
# 실험 3: Linearity Test (H2b)
# ============================================================
print("\n" + "=" * 80)
print("실험 3: Linearity Test — Δ = Acc(RBF) - Acc(Linear) (H2b)")
print("=" * 80)

DIMS_LIN = [20, 50, 100, 200, 500]

exp3 = {m: [] for m in ALL_5}
for method in ALL_5:
    X, y = datasets[method]["X"], datasets[method]["y"]
    print(f"\n[{method}]")
    print(f"  {'D':<6s} {'SVM-L':<12s} {'SVM-RBF':<12s} {'Δ':<10s} {'Fisher LDA':<12s}")
    print("  " + "-" * 52)
    for dim in DIMS_LIN:
        if dim > X.shape[1]: continue
        X_s = StandardScaler().fit_transform(X)
        X_pca = PCA(n_components=dim, random_state=42).fit_transform(X_s)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_l, acc_rbf = [], []
        for tri, tei in skf.split(X_pca, y):
            cl = SVC(kernel="linear", C=1.0); cl.fit(X_pca[tri], y[tri])
            acc_l.append(accuracy_score(y[tei], cl.predict(X_pca[tei])))
            cr = SVC(kernel="rbf", C=1.0, gamma="scale"); cr.fit(X_pca[tri], y[tri])
            acc_rbf.append(accuracy_score(y[tei], cr.predict(X_pca[tei])))
        ml, mr = np.mean(acc_l)*100, np.mean(acc_rbf)*100
        delta = mr - ml
        # Fisher LDA score
        classes = np.unique(y); grand = X_pca.mean(axis=0)
        between = sum(np.sum(y==c) * np.linalg.norm(X_pca[y==c].mean(0) - grand)**2 for c in classes)
        within = sum(np.sum((X_pca[y==c] - X_pca[y==c].mean(0))**2) for c in classes)
        fisher = between / (within + 1e-10)
        exp3[method].append({"dim": dim, "svml": ml, "rbf": mr, "delta": delta, "fisher": fisher})
        print(f"  {dim:<6d} {ml:<12.2f} {mr:<12.2f} {delta:<+10.2f} {fisher:<12.4f}")

# Figure 3: Delta(D) curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for m in ALL_5:
    dims = [e["dim"] for e in exp3[m]]
    deltas = [e["delta"] for e in exp3[m]]
    ax.plot(dims, deltas, "o-", label=m, color=COLORS[m], linewidth=2, markersize=6)
ax.set_xlabel("PCA Dimension D"); ax.set_ylabel("Δ = Acc(RBF) - Acc(Linear) (%)")
ax.set_title("Linearity Gap Δ(D)", fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=8); ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3)

ax = axes[1]
for m in ALL_5:
    dims = [e["dim"] for e in exp3[m]]
    fishers = [e["fisher"] for e in exp3[m]]
    ax.plot(dims, fishers, "o-", label=m, color=COLORS[m], linewidth=2, markersize=6)
ax.set_xlabel("PCA Dimension D"); ax.set_ylabel("Fisher LDA Score")
ax.set_title("Fisher Discriminant Ratio vs Dim", fontweight="bold")
ax.legend(fontsize=8); ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exp3_linearity_test.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 실험 4: Equivalence Test — TOST (H3)
# ============================================================
print("\n" + "=" * 80)
print("실험 4: Equivalence Test — TOST (H3)")
print("=" * 80)

from scipy.stats import ttest_1samp

def tost_equivalence(scores_a, scores_b, delta=1.0):
    """TOST: H0: |mean_a - mean_b| >= delta"""
    diffs = np.array(scores_a) - np.array(scores_b)
    mean_diff = np.mean(diffs)
    # Test 1: mean_diff > -delta (lower bound)
    t1, p1 = ttest_1samp(diffs + delta, 0, alternative="greater")
    # Test 2: mean_diff < +delta (upper bound)
    t2, p2 = ttest_1samp(diffs - delta, 0, alternative="less")
    p_tost = max(p1, p2)
    return mean_diff, p_tost

# 고차원에서 비교 (D=512, NoPCA)
D_HIGH_LIST = [256, 512, None]
DELTA = 2.0  # equivalence margin (%)
N_REPEATS = 20  # 반복 횟수

print(f"\nEquivalence margin δ = {DELTA}%")
print(f"Repeats = {N_REPEATS}\n")

for D_high in D_HIGH_LIST:
    label = f"PCA={D_high}" if D_high else "NoPCA"
    rips_scores, chroma_scores = [], []
    X_r, y_r = datasets["Sixpack_Rips"]["X"], datasets["Sixpack_Rips"]["y"]
    X_c, y_c = datasets["Sixpack_Chroma"]["X"], datasets["Sixpack_Chroma"]["y"]

    for seed in range(N_REPEATS):
        rs = eval_at_dim(X_r, y_r, D_high, SVC(kernel="linear", C=1.0), seed=seed)
        cs = eval_at_dim(X_c, y_c, D_high, SVC(kernel="linear", C=1.0), seed=seed)
        rips_scores.append(rs); chroma_scores.append(cs)

    mean_diff, p_tost = tost_equivalence(chroma_scores, rips_scores, delta=DELTA)
    equiv = "✓ EQUIVALENT" if p_tost < 0.05 else "✗ NOT equivalent"

    print(f"[{label}]")
    print(f"  Rips:   {np.mean(rips_scores):.2f} ± {np.std(rips_scores):.2f}%")
    print(f"  Chroma: {np.mean(chroma_scores):.2f} ± {np.std(chroma_scores):.2f}%")
    print(f"  Diff:   {mean_diff:+.2f}%  |  TOST p={p_tost:.4f}  →  {equiv}")
    print()

# 전체 요약 테이블
print("\n" + "=" * 80)
print("전체 요약")
print("=" * 80)
print(f"\n{'Method':<18s} {'d_eff':<8s} {'Δ@D=500':<10s} {'Top500 Acc':<12s}")
print("-" * 48)
for m in ALL_5:
    deff = d_effs[m]
    delta_500 = [e["delta"] for e in exp3[m] if e["dim"] == 500]
    delta_str = f"{delta_500[0]:+.2f}" if delta_500 else "N/A"
    # top-500 acc
    X = StandardScaler().fit_transform(datasets[m]["X"])
    y = datasets[m]["y"]
    X_pca = PCA(n_components=min(500, X.shape[1]), random_state=42).fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tri, tei in skf.split(X_pca, y):
        clf = SVC(kernel="linear", C=1.0); clf.fit(X_pca[tri], y[tri])
        accs.append(accuracy_score(y[tei], clf.predict(X_pca[tei])))
    print(f"{m:<18s} {deff:<8.1f} {delta_str:<10s} {np.mean(accs)*100:.2f}%")
