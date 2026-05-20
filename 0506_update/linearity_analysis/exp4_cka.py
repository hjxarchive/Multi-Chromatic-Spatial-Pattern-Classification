"""실험 4 — Linear vs RBF CKA (representation-level linearity 측정).
η = CKA_linear / CKA_RBF → 1에 가까울수록 linear 구조.
"""
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist, squareform
from common import generate_cv_indices, scale_train_test, results_to_dataframe
from config import SEEDS, N_OUTER_FOLDS, RESULTS_DIR


def linear_cka(X, Y_onehot):
    """Feature-space Linear CKA. O(d^2 + dC) 메모리."""
    Xc = X - X.mean(axis=0)
    Yc = Y_onehot - Y_onehot.mean(axis=0)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return num / (denom + 1e-10)


def rbf_cka(X, Y_onehot, subsample=4000, rng=None):
    """Kernel-space RBF CKA. Median heuristic for sigma."""
    n = X.shape[0]
    if n > subsample and rng is not None:
        idx = rng.choice(n, subsample, replace=False)
        X, Y_onehot = X[idx], Y_onehot[idx]
        n = subsample

    D2 = squareform(pdist(X, "sqeuclidean"))
    sigma2 = np.median(D2[np.triu_indices(n, k=1)])
    if sigma2 < 1e-10:
        sigma2 = 1.0
    K_X = np.exp(-D2 / (2 * sigma2))
    K_Y = Y_onehot @ Y_onehot.T

    # Centering: H = I - 1/n 11^T
    H = np.eye(n) - np.ones((n, n)) / n
    HKxH = H @ K_X @ H
    HKyH = H @ K_Y @ H

    num = np.sum(HKxH * HKyH)
    denom = np.sqrt(np.sum(HKxH ** 2) * np.sum(HKyH ** 2))
    return num / (denom + 1e-10)


def run_exp4(datasets, descriptor_list):
    records = []
    enc = OneHotEncoder(sparse_output=False)

    for desc in descriptor_list:
        if desc not in datasets:
            print(f"  [SKIP] {desc}"); continue
        X, y = datasets[desc]["X"], datasets[desc]["y"]
        cv_map = generate_cv_indices(y)
        Y_oh_full = enc.fit_transform(y.reshape(-1, 1))
        print(f"\n[Exp4] {desc} (dim={X.shape[1]})")

        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            for fold_idx in range(N_OUTER_FOLDS):
                tri, tei = cv_map[(seed, fold_idx)]
                X_train, _ = scale_train_test(X, tri, tei)
                Y_train = Y_oh_full[tri]

                cka_lin = linear_cka(X_train, Y_train)
                cka_rbf = rbf_cka(X_train, Y_train, rng=rng)
                eta = cka_lin / (cka_rbf + 1e-10)

                records.append({
                    "descriptor": desc,
                    "seed": seed,
                    "fold": fold_idx,
                    "cka_linear": cka_lin,
                    "cka_rbf": cka_rbf,
                    "eta": eta,
                })

        df_tmp = results_to_dataframe([r for r in records if r["descriptor"] == desc])
        print(f"  CKA_lin={df_tmp['cka_linear'].mean():.4f}  "
              f"CKA_rbf={df_tmp['cka_rbf'].mean():.4f}  "
              f"η={df_tmp['eta'].mean():.4f}")

    df = results_to_dataframe(records)
    out_dir = os.path.join(RESULTS_DIR, "exp4")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "exp4_cka.csv"), index=False)
    print(f"\n[Exp4] 저장: {out_dir}/exp4_cka.csv")
    return df
