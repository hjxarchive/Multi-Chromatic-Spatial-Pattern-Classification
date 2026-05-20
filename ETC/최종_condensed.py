"""
최종.ipynb 축약 버전 — Google Colab 복붙용
============================================
원본 노트북의 핵심 로직을 하나의 깔끔한 파일로 정리.
중복 코드, 반복 정의, 출력 아티팩트 모두 제거.

구성:
  1. 환경 설정 및 임포트
  2. Ground Truth & 인접 위상 정의
  3. 데이터 로딩 함수
  4. 전처리 유틸리티
  5. 평가 함수 (분류기 + Soft Margin SVM)
  6. 시각화 함수
  7. 메인 실행
"""

# ============================================================
# 1. 환경 설정 및 임포트
# ============================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# Google Colab 환경
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# 경로 설정
BASE_DIR = '/content/drive/MyDrive/URP' if IN_COLAB else '.'
DATA_PATHS = {
    'Inter_PI':       os.path.join(BASE_DIR, '1224_Vectors/Inter_PI'),
    '3D_PI':          os.path.join(BASE_DIR, '1224_Vectors/3D_PI'),
    'Ord_PI':         os.path.join(BASE_DIR, '1224_Vectors/Ord_PI'),
    'Sixpack_Chroma': os.path.join(BASE_DIR, '1224_Vectors/Sixpack_Chroma'),
    'Sixpack_Rips':   os.path.join(BASE_DIR, '1224_Vectors/Sixpack_Rips'),
}

# 하이퍼파라미터
C_VALUES       = [0.5, 1.0, 2.0]
REDUCTION_DIM  = 20
N_SPLITS       = 5
RANDOM_STATE   = 42

print(f"IN_COLAB={IN_COLAB}")

# ============================================================
# 2. Ground Truth & 인접 위상 정의
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
    """task_id (1-based) → GT label.  idx = RR*64 + RG*8 + GG"""
    idx = task_id - 1
    return GROUND_TRUTH_M[(idx % 64) // 8][idx // 64][idx % 8]

# 인접 위상 (실제 phase diagram 기반)
ADJACENT_PHASES = {
    0:  [1, 2],
    1:  [0, 3],
    2:  [0, 3],
    3:  [1, 2, 4, 6, 7, 10, 11],
    4:  [3, 7, 10, 11, 12, 13],
    6:  [3, 7, 9, 10, 11, 12, 13],
    7:  [3, 4, 6, 11, 12],
    9:  [6, 10, 13],
    10: [3, 4, 6, 9, 13],
    11: [3, 4, 6, 7, 12, 13],
    12: [4, 6, 7, 11],
    13: [4, 6, 9, 10, 11],
}

ALL_CLASSES = sorted(np.unique(GROUND_TRUTH_M))
print(f"Classes: {ALL_CLASSES} ({len(ALL_CLASSES)} classes)")

# ============================================================
# 3. 데이터 로딩 함수
# ============================================================
def _load_generic_pi(data_dir, prefix):
    """Inter_PI / 3D_PI / Ord_PI 공용 로더."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.npz")))
    print(f"  Found {len(files)} files")
    X_list, y_list = [], []
    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split('_')[-1].split('.')[0])
            label   = get_label_from_index(sim_idx)
            data    = np.load(fp, allow_pickle=True)
            features = []
            for key in ('arr_0', 'arr_1'):
                arr = data[key]
                if hasattr(arr, 'item') and arr.ndim == 0:
                    arr = arr.item()
                elif arr.shape == (1,):
                    arr = arr[0]
                if isinstance(arr, dict):
                    for k in sorted(arr.keys()):
                        val = arr[k]
                        if isinstance(val, dict):
                            for dk in sorted(val.keys()):
                                features.extend(np.asarray(val[dk]).flatten())
                        else:
                            features.extend(np.asarray(val).flatten())
                else:
                    features.extend(np.asarray(arr).flatten())
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"  Error {fp}: {e}")
    if not X_list:
        return None, None
    return np.nan_to_num(np.array(X_list, dtype=float)), np.array(y_list)


def _extract_stat_features(barcode):
    """barcode → 12-dim 통계 feature 벡터."""
    if len(barcode) == 0:
        return np.zeros(12)
    bc = np.array(barcode)
    if bc.ndim == 1:
        if len(bc) % 2 == 0:
            bc = bc.reshape(-1, 2)
        elif len(bc) > 2:
            bc = bc[:len(bc)//2*2].reshape(-1, 2)
        else:
            bc = np.array([[0., 0.]])
    if bc.ndim == 1 or bc.shape[1] < 2:
        return np.zeros(12)
    ls = bc[:, 1] - bc[:, 0]
    b, d = bc[:, 0], bc[:, 1]
    feats = [len(bc), np.mean(ls), np.std(ls), np.max(ls), np.min(ls),
             np.sum(ls), np.mean(b), np.std(b), np.mean(d), np.std(d),
             np.median(ls)]
    p = ls / np.sum(ls) if np.sum(ls) > 0 else ls
    p = p[p > 0]
    feats.append(-np.sum(p * np.log(p + 1e-10)) if len(p) > 0 else 0)
    return np.array(feats)

BARCODE_TYPES = ['domain', 'codomain', 'relative', 'image', 'kernel', 'cokernel']

def _load_sixpack_rips(data_dir):
    """Sixpack_Rips → 288D (2방향×6type×2dim×12stat)."""
    files = sorted(glob.glob(os.path.join(data_dir, "Sixpack_Rips_*.npz")))
    print(f"  Found {len(files)} files")
    X_list, y_list = [], []
    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split('_')[-1].split('.')[0])
            label   = get_label_from_index(sim_idx)
            data    = np.load(fp, allow_pickle=True)
            sp = {'A_to_B': data['arr_0'].item(), 'B_to_A': data['arr_1'].item()}
            feats = []
            for d_key in ['A_to_B', 'B_to_A']:
                dd = sp[d_key]
                for bt in BARCODE_TYPES:
                    for dim_key in [0, 1]:
                        if bt in dd and dim_key in dd[bt]:
                            raw = np.array(dd[bt][dim_key])
                            if len(raw) == 0:
                                feats.extend(np.zeros(12))
                            else:
                                feats.extend(_extract_stat_features(raw))
                        else:
                            feats.extend(np.zeros(12))
            X_list.append(feats)
            y_list.append(label)
        except Exception as e:
            print(f"  Error {fp}: {e}")
    if not X_list:
        return None, None
    return np.nan_to_num(np.array(X_list)), np.array(y_list)


def _load_sixpack_chroma(data_dir):
    """Sixpack_Chroma 로더 (dict 기반 PI 벡터)."""
    files = sorted(glob.glob(os.path.join(data_dir, "Sixpack_Chroma_*.npz")))
    print(f"  Found {len(files)} files")
    X_list, y_list = [], []
    for fp in files:
        try:
            sim_idx = int(os.path.basename(fp).split('_')[-1].split('.')[0])
            label   = get_label_from_index(sim_idx)
            data    = np.load(fp, allow_pickle=True)
            features = []
            for key in ('arr_0', 'arr_1'):
                arr = data[key]
                if hasattr(arr, 'item') and arr.ndim == 0:
                    arr = arr.item()
                if isinstance(arr, dict):
                    for k in sorted(arr.keys()):
                        val = arr[k]
                        if isinstance(val, dict):
                            for dk in sorted(val.keys()):
                                features.extend(np.asarray(val[dk]).flatten())
                        else:
                            features.extend(np.asarray(val).flatten())
                else:
                    features.extend(np.asarray(arr).flatten())
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"  Error {fp}: {e}")
    if not X_list:
        return None, None
    return np.nan_to_num(np.array(X_list, dtype=float)), np.array(y_list)


def load_all_datasets():
    """모든 데이터셋 로드 + Inter+Ord, 3D_Ord 결합."""
    datasets = {}
    loaders = {
        'Inter_PI':       lambda d: _load_generic_pi(d, 'Inter_PI'),
        '3D_PI':          lambda d: _load_generic_pi(d, '3D_PI'),
        'Ord_PI':         lambda d: _load_generic_pi(d, 'Ord_PI'),
        'Sixpack_Chroma': _load_sixpack_chroma,
        'Sixpack_Rips':   _load_sixpack_rips,
    }
    for name, loader in loaders.items():
        path = DATA_PATHS.get(name)
        if path and os.path.exists(path):
            print(f"\n[{name}]")
            X, y = loader(path)
            if X is not None and len(X) > 0:
                datasets[name] = {'X': X, 'y': y}
                print(f"  Shape: {X.shape}, Classes: {len(np.unique(y))}")

    # 결합 데이터셋
    if 'Inter_PI' in datasets and 'Ord_PI' in datasets:
        datasets['Inter+Ord'] = {
            'X': np.hstack([datasets['Inter_PI']['X'], datasets['Ord_PI']['X']]),
            'y': datasets['Inter_PI']['y'],
        }
        print(f"\n[Inter+Ord] Shape: {datasets['Inter+Ord']['X'].shape}")
    if '3D_PI' in datasets and 'Ord_PI' in datasets:
        datasets['3D_Ord'] = {
            'X': np.hstack([datasets['3D_PI']['X'], datasets['Ord_PI']['X']]),
            'y': datasets['3D_PI']['y'],
        }
        print(f"[3D_Ord] Shape: {datasets['3D_Ord']['X'].shape}")
    return datasets

# ============================================================
# 4. 전처리 유틸리티
# ============================================================
def _flatten_dict_cell(cell):
    if isinstance(cell, dict):
        flat = []
        for key in sorted(cell.keys()):
            val = cell[key]
            if isinstance(val, np.ndarray):
                flat.extend(val.flatten().tolist())
            elif isinstance(val, (list, tuple)):
                flat.extend(list(val))
            else:
                flat.append(float(val))
        return flat
    elif isinstance(cell, np.ndarray):
        return cell.flatten().tolist()
    elif isinstance(cell, (list, tuple)):
        return list(cell)
    return [float(cell)]

def preprocess_X(X):
    """X를 float64 ndarray로 정규화 (object dtype 지원)."""
    if hasattr(X, 'numpy'):  X = X.numpy()
    if hasattr(X, 'values'): X = X.values
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.dtype == object:
        rows = []
        for row in X:
            flat = []
            if hasattr(row, '__iter__') and not isinstance(row, (str, dict)):
                for c in row:
                    flat.extend(_flatten_dict_cell(c))
            else:
                flat.extend(_flatten_dict_cell(row))
            rows.append(flat)
        max_len = max(len(r) for r in rows)
        rows = [r + [0.0]*(max_len - len(r)) for r in rows]
        X = np.array(rows, dtype=np.float64)
    X = X.astype(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def preprocess_y(y):
    if hasattr(y, 'numpy'):  y = y.numpy()
    if hasattr(y, 'values'): y = y.values
    return np.array(y).flatten().astype(int)

# ============================================================
# 5. 평가 함수
# ============================================================
def are_adjacent(label1, label2):
    if label1 == label2:
        return True
    if label1 in ADJACENT_PHASES and label2 in ADJACENT_PHASES[label1]:
        return True
    if label2 in ADJACENT_PHASES and label1 in ADJACENT_PHASES[label2]:
        return True
    return False

def soft_accuracy(y_true, y_pred):
    """인접 위상 오분류를 정답으로 처리하는 정확도."""
    correct = sum(1 for t, p in zip(y_true, y_pred)
                  if t == p or are_adjacent(int(t), int(p)))
    return correct / len(y_true) if len(y_true) > 0 else 0.0

def strict_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def clone_clf(clf):
    if isinstance(clf, KNeighborsClassifier):
        return KNeighborsClassifier(n_neighbors=clf.n_neighbors)
    elif isinstance(clf, SVC):
        return SVC(kernel=clf.kernel, C=clf.C,
                   gamma=getattr(clf, 'gamma', 'scale'),
                   random_state=RANDOM_STATE)
    elif isinstance(clf, RandomForestClassifier):
        return RandomForestClassifier(n_estimators=clf.n_estimators,
                                      random_state=RANDOM_STATE)
    return clone(clf)

def evaluate_all_classifiers(X, y, C_values=C_VALUES, n_splits=N_SPLITS):
    """모든 분류기 (기본 5개 + Soft-SVM) 평가, soft/strict 둘 다 계산."""
    classifiers = {
        'KNN (k=3)':       KNeighborsClassifier(n_neighbors=3),
        'KNN (k=12)':      KNeighborsClassifier(n_neighbors=12),
        'SVM (RBF)':       SVC(kernel='rbf', C=1.0, gamma='scale'),
        'SVM (Linear)':    SVC(kernel='linear', C=1.0),
        'Random Forest':   RandomForestClassifier(100, random_state=RANDOM_STATE),
    }
    for C in C_values:
        classifiers[f'Soft-SVM (C={C})'] = SVC(kernel='rbf', C=C, gamma='scale')

    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in classifiers.items():
        accs_soft, accs_strict = [], []
        for tri, tei in skf.split(X, y):
            c = clone_clf(clf)
            c.fit(X[tri], y[tri])
            yp = c.predict(X[tei])
            accs_soft.append(soft_accuracy(y[tei], yp))
            accs_strict.append(strict_accuracy(y[tei], yp))
        results[name] = {
            'mean_soft':    np.mean(accs_soft)   * 100,
            'std_soft':     np.std(accs_soft)    * 100,
            'mean_strict':  np.mean(accs_strict) * 100,
            'std_strict':   np.std(accs_strict)  * 100,
        }
    return results

# ============================================================
# 6. 시각화 함수
# ============================================================
CLF_NAMES = ['KNN (k=3)', 'KNN (k=12)', 'SVM (RBF)', 'SVM (Linear)',
             'Random Forest', 'Soft-SVM (C=0.5)', 'Soft-SVM (C=1.0)', 'Soft-SVM (C=2.0)']

def print_comparison_table(all_results, use_soft=True):
    metric = 'soft' if use_soft else 'strict'
    title  = "Adjacent Tolerance" if use_soft else "Strict"
    print(f"\n{'='*170}")
    print(f"Full Classifier Comparison ({title})")
    print(f"{'='*170}")
    header = f"{'Method':<20} {'Dim':>8}"
    for c in CLF_NAMES:
        header += f" {c[:14]:>16}"
    print(header)
    print('-'*170)
    for method, result in all_results.items():
        row = f"{method[:20]:<20} {result.get('original_dim','N/A'):>8}"
        clf_res = result.get('classifiers', {})
        for c in CLF_NAMES:
            if c in clf_res:
                m = clf_res[c][f'mean_{metric}']
                s = clf_res[c][f'std_{metric}']
                row += f" {m:>7.1f}+/-{s:<5.1f}%"
            else:
                row += f" {'N/A':>16}"
        print(row)
    print('='*170)

def plot_soft_svm_bar(all_results, C_values=C_VALUES, use_soft=True, save_path=None):
    metric = 'soft' if use_soft else 'strict'
    methods = list(all_results.keys())
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(methods)*1.5), 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, C in enumerate(C_values):
        means, stds = [], []
        for m in methods:
            sr = all_results[m].get('soft_svm', {}).get(f'C={C}', {})
            means.append(sr.get(f'mean_{metric}', 0))
            stds.append(sr.get(f'std_{metric}', 0))
        bars = ax.bar(x+i*width, means, width, yerr=stds, label=f'C={C}',
                      color=colors[i], capsize=3, alpha=0.8)
        for b, mv in zip(bars, means):
            if mv > 0:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                        f'{mv:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
    ax.set_xlabel('Vectorization Method')
    ax.set_ylabel('Accuracy (%)')
    suffix = "(Adjacent Tolerance)" if use_soft else "(Strict)"
    ax.set_title(f'Soft Margin SVM Classification Accuracy {suffix}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(title='C'); ax.set_ylim([0, 110]); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 7. 메인 실행
# ============================================================
if __name__ == '__main__':
    # --- 데이터 로딩 ---
    print("=" * 80)
    print("데이터 로딩")
    print("=" * 80)
    datasets = load_all_datasets()
    print(f"\n로드된 데이터셋: {list(datasets.keys())}")

    # --- 전체 평가 ---
    print("\n" + "=" * 80)
    print(f"EVALUATION  (PCA {REDUCTION_DIM}D, {N_SPLITS}-fold CV)")
    print("=" * 80)

    all_results = {}
    for method_name, data in datasets.items():
        print(f"\n--- [{method_name}] ---")
        X = preprocess_X(data['X'])
        y = preprocess_y(data['y'])
        original_dim = X.shape[1]
        print(f"  Shape: {X.shape}")

        # Scaling + PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if original_dim > REDUCTION_DIM:
            X_reduced = PCA(n_components=REDUCTION_DIM,
                            random_state=RANDOM_STATE).fit_transform(X_scaled)
            print(f"  PCA: {original_dim}D -> {REDUCTION_DIM}D")
        else:
            X_reduced = X_scaled

        # 분류기 평가
        clf_results = evaluate_all_classifiers(X_reduced, y)

        all_results[method_name] = {
            'original_dim': original_dim,
            'reduced_dim':  X_reduced.shape[1],
            'samples':      X.shape[0],
            'classifiers':  clf_results,
        }

        # Soft-SVM 결과만 별도 저장 (호환성)
        soft_svm = {}
        for C in C_VALUES:
            key = f'Soft-SVM (C={C})'
            if key in clf_results:
                soft_svm[f'C={C}'] = clf_results[key]
        all_results[method_name]['soft_svm'] = soft_svm

    # --- 결과 출력 ---
    if all_results:
        print_comparison_table(all_results, use_soft=True)
        print_comparison_table(all_results, use_soft=False)
        plot_soft_svm_bar(all_results, use_soft=True,
                          save_path='soft_svm_results_adjacent_tol.png')
        plot_soft_svm_bar(all_results, use_soft=False,
                          save_path='soft_svm_results_strict.png')

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
