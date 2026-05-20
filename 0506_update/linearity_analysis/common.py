"""공통 모듈 — CV 인덱스 생성, 데이터 로딩, Fisher ratio, 유틸리티."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from config import SEEDS, N_OUTER_FOLDS


def generate_cv_indices(y, seeds=SEEDS, n_splits=N_OUTER_FOLDS):
    """모든 실험이 공유하는 (seed, fold) → (train_idx, test_idx) 매핑 생성.
    Returns dict[(seed, fold_idx)] = (train_idx, test_idx)
    """
    cv_map = {}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (tri, tei) in enumerate(skf.split(np.zeros(len(y)), y)):
            cv_map[(seed, fold_idx)] = (tri, tei)
    return cv_map


def scale_train_test(X, train_idx, test_idx, pca_dim=500):
    """Train에만 fit하고 test에 transform — data leak 방지.
    실행 속도를 위해 고차원 데이터(예: 61200D)는 500D로 PCA 축소.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    
    # 61200D 같은 고차원 데이터만 PCA 적용 (원래 차원이 500 이하이면 무시됨)
    if pca_dim and X_train.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
    return X_train, X_test


def compute_fisher_ratio(X, y):
    """Fisher discriminant ratio  J = tr(S_W^{-1} S_B).
    regularized S_W로 안정적 계산.
    """
    classes = np.unique(y)
    n, d = X.shape
    grand_mean = X.mean(axis=0)

    S_W = np.zeros((d, d))
    S_B = np.zeros((d, d))

    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        mu_c = X_c.mean(axis=0)
        diff_c = X_c - mu_c
        S_W += diff_c.T @ diff_c
        diff_m = (mu_c - grand_mean).reshape(-1, 1)
        S_B += n_c * (diff_m @ diff_m.T)

    # Regularization
    eps = 1e-4 * np.mean(np.diag(S_W))
    S_W_reg = S_W + eps * np.eye(d)

    # J = tr(S_W^{-1} S_B)
    J = np.trace(np.linalg.solve(S_W_reg, S_B))
    return J


def results_to_dataframe(records):
    """List of dicts → pandas DataFrame."""
    return pd.DataFrame(records)
