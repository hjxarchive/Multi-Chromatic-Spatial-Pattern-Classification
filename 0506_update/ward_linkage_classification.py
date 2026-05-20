"""
Ward Linkage (Hierarchical Clustering) Classification 핫픽스
- 비지도 클러스터링 → 클러스터-클래스 매핑(majority voting) → 평가
- Google Colab에서 바로 실행 가능
"""

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np


def cluster_to_class_mapping(y_true, cluster_labels, n_classes):
    """
    Hungarian algorithm으로 최적의 cluster→class 매핑을 찾는다.
    Cost matrix: cluster i에 속한 샘플 중 class j가 아닌 것의 수
    """
    unique_clusters = np.unique(cluster_labels)
    unique_classes = np.unique(y_true)
    n_clusters = len(unique_clusters)
    n_cls = len(unique_classes)

    # cost matrix 생성
    size = max(n_clusters, n_cls)
    cost = np.zeros((size, size))
    for i, cl in enumerate(unique_clusters):
        mask = cluster_labels == cl
        for j, cls in enumerate(unique_classes):
            # 매칭되지 않는 수 = 비용
            cost[i, j] = np.sum(mask) - np.sum(y_true[mask] == cls)

    # Hungarian algorithm으로 최적 매칭
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_clusters and c < n_cls:
            mapping[unique_clusters[r]] = unique_classes[c]

    return mapping


def ward_classification(X, y, n_clusters=12):
    """
    Ward Linkage로 전체 데이터 클러스터링 후 평가.
    (비지도이므로 CV 없이 전체 데이터에 적용)
    """
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = ward.fit_predict(X)

    # 최적 매핑
    mapping = cluster_to_class_mapping(y, cluster_labels, n_clusters)

    # 매핑 적용
    y_pred = np.array([mapping.get(cl, -1) for cl in cluster_labels])

    strict = accuracy_score(y, y_pred) * 100
    f1 = f1_score(y, y_pred, average="macro", zero_division=0) * 100

    # soft accuracy (adjacent tolerance)
    soft = soft_accuracy_score(y, y_pred) * 100

    return {
        "soft_acc": soft,
        "strict_acc": strict,
        "f1_macro": f1,
        "y_pred": y_pred,
        "cluster_labels": cluster_labels,
        "mapping": mapping,
    }


# === 실행 코드 ===
# (pipeline.py의 datasets, all_results가 이미 로딩되어 있다고 가정)

print("=" * 80)
print("Ward Linkage (Hierarchical Clustering) Classification")
print("=" * 80)

ward_results = {}
for method in METHODS:
    X_red = all_results[method]["X_reduced"]  # PCA 20D 적용된 데이터
    y = all_results[method]["y"]

    res = ward_classification(X_red, y, n_clusters=len(ALL_CLASSES))
    ward_results[method] = res

    print(f"  {method:<20s} | Soft={res['soft_acc']:.2f}% | "
          f"Strict={res['strict_acc']:.2f}% | F1={res['f1_macro']:.2f}%")

# 결과 테이블
print("\n--- Ward Linkage 종합 결과 ---")
print(f"{'Method':<20s} {'Soft(%)':<12s} {'Strict(%)':<12s} {'F1(%)':<12s}")
print("-" * 56)
for method in METHODS:
    r = ward_results[method]
    print(f"{method:<20s} {r['soft_acc']:<12.2f} {r['strict_acc']:<12.2f} {r['f1_macro']:<12.2f}")
