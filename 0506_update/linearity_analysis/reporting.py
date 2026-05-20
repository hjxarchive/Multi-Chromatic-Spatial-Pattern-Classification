"""Reporting — 세 실험의 결과를 집계하여 figure + table 생성."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from config import COLORS, RESULTS_DIR, DESCRIPTORS


def load_csv(exp_name, filename):
    path = os.path.join(RESULTS_DIR, exp_name, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [WARN] {path} not found")
    return None


def paired_wilcoxon(df, metric, desc_a, desc_b, group_col="descriptor"):
    """동일 (seed, fold)에서의 paired Wilcoxon signed-rank test."""
    a = df[df[group_col] == desc_a].sort_values(["seed", "fold"])[metric].values
    b = df[df[group_col] == desc_b].sort_values(["seed", "fold"])[metric].values
    if len(a) != len(b) or len(a) < 5:
        return np.nan, np.nan
    stat, p = wilcoxon(a, b)
    return stat, p


def plot_exp1(df, output_dir):
    """Figure 1: Kernel Ladder line plot."""
    if df is None: return
    kernel_order = ["Linear", "Poly-2", "Poly-3", "Poly-5",
                    "RBF-small", "RBF-mid", "RBF-large"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for desc in DESCRIPTORS:
        sub = df[df["descriptor"] == desc]
        if sub.empty: continue
        means, stds = [], []
        for k in kernel_order:
            vals = sub[sub["kernel"] == k]["accuracy"]
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) else 0)
        ax.errorbar(range(len(kernel_order)), means, yerr=stds, fmt="o-",
                    label=desc, color=COLORS.get(desc, "gray"),
                    linewidth=2, markersize=7, capsize=3)
    ax.set_xticks(range(len(kernel_order)))
    ax.set_xticklabels(kernel_order, rotation=30, ha="right")
    ax.set_xlabel("Kernel (complexity →)", fontsize=12)
    ax.set_ylabel("Strict Accuracy (%)", fontsize=12)
    ax.set_title("Exp 1: Kernel Complexity Ladder", fontweight="bold", fontsize=14)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_kernel_ladder.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_exp2(df, output_dir):
    """Figure 2: Classifier hierarchy bar chart + r_LDA, J annotation."""
    if df is None: return
    clf_order = ["NCM", "LDA", "QDA", "KNN", "SVM-RBF"]
    n_desc = len(DESCRIPTORS)
    n_clf = len(clf_order)
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_clf)

    for i, desc in enumerate(DESCRIPTORS):
        sub = df[df["descriptor"] == desc]
        if sub.empty: continue
        means = [sub[sub["classifier"] == c]["accuracy"].mean() for c in clf_order]
        stds = [sub[sub["classifier"] == c]["accuracy"].std() for c in clf_order]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=desc, color=COLORS.get(desc, "gray"), capsize=2)

    ax.set_xticks(x + width * (n_desc - 1) / 2)
    ax.set_xticklabels(clf_order, fontsize=11)
    ax.set_ylabel("Strict Accuracy (%)", fontsize=12)
    ax.set_title("Exp 2: Classifier Hierarchy", fontweight="bold", fontsize=14)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # r_LDA, J annotation
    for desc in DESCRIPTORS:
        sub = df[df["descriptor"] == desc]
        if sub.empty: continue
        lda_acc = sub[sub["classifier"] == "LDA"]["accuracy"].mean()
        rbf_acc = sub[sub["classifier"] == "SVM-RBF"]["accuracy"].mean()
        J = sub["fisher_J"].mean()
        r_lda = lda_acc / rbf_acc if rbf_acc > 0 else 0
        print(f"  {desc:<18s} r_LDA={r_lda:.4f}  J={J:.2f}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_classifier_hierarchy.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_exp4(df, output_dir):
    """Figure 3: CKA comparison table + bar."""
    if df is None: return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["cka_linear", "cka_rbf", "eta"]
    titles = ["CKA Linear", "CKA RBF", "η = CKA_lin / CKA_rbf"]

    for ax, metric, title in zip(axes, metrics, titles):
        means, stds, names = [], [], []
        for desc in DESCRIPTORS:
            sub = df[df["descriptor"] == desc]
            if sub.empty: continue
            names.append(desc)
            means.append(sub[metric].mean())
            stds.append(sub[metric].std())
        ax.bar(range(len(names)), means, yerr=stds,
               color=[COLORS.get(n, "gray") for n in names], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(means):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle("Exp 4: CKA Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_cka.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def generate_report(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("REPORTING")
    print("=" * 80)

    df1 = load_csv("exp1", "exp1_kernel_ladder.csv")
    df2 = load_csv("exp2", "exp2_classifier_hierarchy.csv")
    df4 = load_csv("exp4", "exp4_cka.csv")

    plot_exp1(df1, output_dir)
    plot_exp2(df2, output_dir)
    plot_exp4(df4, output_dir)

    # Paired Wilcoxon: Chroma vs Rips
    print("\n--- Paired Wilcoxon (Chroma vs Rips) ---")
    if df1 is not None:
        lin_c = df1[(df1["descriptor"]=="Sixpack_Chroma") & (df1["kernel"]=="Linear")]
        lin_r = df1[(df1["descriptor"]=="Sixpack_Rips") & (df1["kernel"]=="Linear")]
        if len(lin_c) == len(lin_r) and len(lin_c) >= 5:
            s, p = wilcoxon(lin_c["accuracy"].values, lin_r["accuracy"].values)
            print(f"  Exp1 Linear kernel: W={s}, p={p:.4f}")

    if df4 is not None:
        _, p = paired_wilcoxon(df4, "eta", "Sixpack_Chroma", "Sixpack_Rips")
        print(f"  Exp4 η: p={p:.4f}")

    print(f"\nFigures 저장: {output_dir}/")
