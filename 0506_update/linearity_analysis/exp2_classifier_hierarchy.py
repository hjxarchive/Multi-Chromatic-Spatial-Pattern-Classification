"""실험 2 — Classifier Hierarchy (H2b 검증).
Generative 가정 강도 순: NCM → LDA → QDA → KNN → SVM-RBF.
r_LDA = Acc_LDA / Acc_SVM-RBF 가 1에 가까울수록 linear 구조.
fold마다 Fisher J도 함께 계산.
"""
import numpy as np
import os
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from common import (generate_cv_indices, scale_train_test,
                     compute_fisher_ratio, results_to_dataframe)
from config import SEEDS, N_OUTER_FOLDS, N_INNER_FOLDS, RESULTS_DIR


def get_classifier_hierarchy():
    """5개 분류기를 generative 가정 강도 순으로 반환."""
    return [
        ("NCM",
         NearestCentroid(), {}),

        ("LDA",
         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}),

        ("QDA",
         QuadraticDiscriminantAnalysis(reg_param=0.01), {}),

        ("KNN",
         KNeighborsClassifier(),
         {"n_neighbors": [1, 5, 10]}),

        ("SVM-RBF",
         SVC(kernel="rbf", gamma="scale"),
         {"C": [0.1, 1, 10]}),
    ]


def run_exp2(datasets, descriptor_list):
    records = []

    for desc in descriptor_list:
        if desc not in datasets:
            print(f"  [SKIP] {desc}"); continue
        X, y = datasets[desc]["X"], datasets[desc]["y"]
        cv_map = generate_cv_indices(y)
        print(f"\n[Exp2] {desc} (dim={X.shape[1]})")

        for seed in SEEDS:
            for fold_idx in range(N_OUTER_FOLDS):
                tri, tei = cv_map[(seed, fold_idx)]
                X_train, X_test = scale_train_test(X, tri, tei)
                y_train, y_test = y[tri], y[tei]

                # Fisher J (train only)
                J = compute_fisher_ratio(X_train, y_train)

                for clf_name, base_clf, param_grid in get_classifier_hierarchy():
                    if param_grid:
                        gs = GridSearchCV(base_clf, param_grid,
                                          cv=N_INNER_FOLDS, scoring="accuracy",
                                          n_jobs=-1, refit=True)
                        gs.fit(X_train, y_train)
                        acc = accuracy_score(y_test, gs.predict(X_test))
                    else:
                        from sklearn.base import clone
                        clf = clone(base_clf)
                        clf.fit(X_train, y_train)
                        acc = accuracy_score(y_test, clf.predict(X_test))

                    records.append({
                        "descriptor": desc,
                        "classifier": clf_name,
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": acc * 100,
                        "fisher_J": J,
                    })

        # Summary
        df_tmp = results_to_dataframe([r for r in records if r["descriptor"] == desc])
        pivot = df_tmp.groupby("classifier")["accuracy"].mean()
        rbf_acc = pivot.get("SVM-RBF", 1)
        print(f"  NCM={pivot.get('NCM',0):.2f}  LDA={pivot.get('LDA',0):.2f}  "
              f"QDA={pivot.get('QDA',0):.2f}  KNN={pivot.get('KNN',0):.2f}  "
              f"RBF={rbf_acc:.2f}")
        print(f"  r_LDA={pivot.get('LDA',0)/rbf_acc:.4f}  "
              f"r_NCM={pivot.get('NCM',0)/rbf_acc:.4f}  "
              f"J={df_tmp['fisher_J'].mean():.4f}")

    df = results_to_dataframe(records)
    out_dir = os.path.join(RESULTS_DIR, "exp2")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "exp2_classifier_hierarchy.csv"), index=False)
    print(f"\n[Exp2] 저장: {out_dir}/exp2_classifier_hierarchy.csv")
    return df
