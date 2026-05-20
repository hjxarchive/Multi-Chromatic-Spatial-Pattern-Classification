# Project Guidelines

## 벡터 로딩 방식

**Sixpack_Rips / Sixpack_Chroma 데이터는 항상 raw PI 벡터를 직접 사용한다.**

- npz 파일에 저장된 PI 벡터 (H0: 100D, H1: 5000D)를 flatten해서 그대로 사용
- 2 directions × 6 barcodes × (100 + 5000) = **61200D**
- `extract_statistical_features` 같은 추가 preprocessing을 적용하면 안 됨
- 288D는 PI 벡터에 stat features를 뽑는 비표준 방식이며, 실험 비교 시 공정하지 않음

```python
# 올바른 로딩 방법
def load_pi(data_dir, prefix):
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
```

## 로컬 환경

- Python 가상환경: `.venv/` (numpy, scikit-learn, pandas 등 설치됨)
- 실행: `.venv/bin/python script.py`
- 벡터 데이터 위치: `Final_Vector/Sixpack_Rips/`, `Final_Vector/Sixpack_Chroma/`
- 벤치마크 결과: `Final_Results(0521)/full_benchmark.csv`
