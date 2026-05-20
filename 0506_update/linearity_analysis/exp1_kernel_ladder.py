"""실험 1 — Kernel Complexity Ladder (H2b 검증).
Linear → Poly(2,3,5) → RBF(small/mid/large gamma) 순으로 kernel 복잡도를 올리며
각 descriptor의 정확도 변화를 관찰한다.
Chroma: linear에서 이미 plateau, Rips: 우상향 기대.
"""
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from common import generate_cv_indices, scale_train_test, results_to_dataframe
from config import SEEDS, N_OUTER_FOLDS, N_INNER_FOLDS, RESULTS_DIR


def get_kernel_ladder():
    """7개 kernel을 complexity 순으로 반환. (name, base_clf, param_grid)"""
    return [
        ("Linear",
         SVC(kernel="linear"),
         {"C": [0.01, 0.1, 1, 10, 100]}),

        ("Poly-2",
         SVC(kernel="poly", degree=2, coef0=1),
         {"C": [0.1, 1, 10]}),

        ("Poly-3",
         SVC(kernel="poly", degree=3, coef0=1),
         {"C": [0.1, 1, 10]}),

        ("Poly-5",
         SVC(kernel="poly", degree=5, coef0=1),
         {"C": [0.1, 1, 10]}),

        ("RBF-small",
         None,  # gamma set dynamically
         {"C": [0.1, 1, 10]}),

        ("RBF-mid",
         None,
         {"C": [0.1, 1, 10]}),

        ("RBF-large",
         None,
         {"C": [0.1, 1, 10]}),
    ]


def run_exp1(datasets, descriptor_list):
    """실험 1 실행. datasets: dict[name] -> {"X": array, "y": array}"""
    records = []

    for desc in descriptor_list:
        if desc not in datasets:
            print(f"  [SKIP] {desc} not in datasets"); continue
        X, y = datasets[desc]["X"], datasets[desc]["y"]
        d = X.shape[1]
        cv_map = generate_cv_indices(y)

        print(f"\n[Exp1] {desc} (dim={d})")
        ladder = get_kernel_ladder()

        for seed in SEEDS:
            for fold_idx in range(N_OUTER_FOLDS):
                tri, tei = cv_map[(seed, fold_idx)]
                X_train, X_test = scale_train_test(X, tri, tei)
                y_train, y_test = y[tri], y[tei]

                for kern_name, base_clf, param_grid in ladder:
                    # RBF gamma 동적 설정
                    if kern_name == "RBF-small":
                        base_clf = SVC(kernel="rbf", gamma=0.1/d)
                    elif kern_name == "RBF-mid":
                        base_clf = SVC(kernel="rbf", gamma=1.0/d)
                    elif kern_name == "RBF-large":
                        base_clf = SVC(kernel="rbf", gamma=10.0/d)

                    gs = GridSearchCV(base_clf, param_grid,
                                      cv=N_INNER_FOLDS, scoring="accuracy",
                                      n_jobs=-1, refit=True)
                    gs.fit(X_train, y_train)
                    acc = accuracy_score(y_test, gs.predict(X_test))

                    records.append({
                        "descriptor": desc,
                        "kernel": kern_name,
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": acc * 100,
                        "best_C": gs.best_params_.get("C"),
                    })

        # Progress
        df_tmp = results_to_dataframe([r for r in records if r["descriptor"] == desc])
        summary = df_tmp.groupby("kernel")["accuracy"].agg(["mean", "std"])
        for kern, row in summary.iterrows():
            print(f"  {kern:<12s} {row['mean']:.2f} ± {row['std']:.2f}%")

    df = results_to_dataframe(records)
    out_dir = os.path.join(RESULTS_DIR, "exp1")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "exp1_kernel_ladder.csv"), index=False)
    print(f"\n[Exp1] 저장: {out_dir}/exp1_kernel_ladder.csv")
    return df
