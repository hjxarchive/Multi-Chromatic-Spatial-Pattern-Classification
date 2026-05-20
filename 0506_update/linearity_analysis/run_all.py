"""run_all.py — 전체 실험 실행 (Google Colab 복붙용).
pipeline.py의 datasets가 이미 로딩되어 있다고 가정.
try/except로 감싸서 부분 실패에도 나머지 진행.
"""
import sys, os

# linearity_analysis/ 를 path에 추가
LA_DIR = os.path.join(os.path.dirname(__file__) if "__file__" in dir()
                       else "/content/drive/MyDrive/URP/0506_update",
                       "linearity_analysis")
if LA_DIR not in sys.path:
    sys.path.insert(0, LA_DIR)

from config import DESCRIPTORS, RESULTS_DIR
from exp1_kernel_ladder import run_exp1
from exp2_classifier_hierarchy import run_exp2
from exp4_cka import run_exp4
from reporting import generate_report

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_all(datasets, descriptor_list=None):
    """세 실험을 차례로 실행하고 report 생성.

    Args:
        datasets: dict[name] -> {"X": ndarray, "y": ndarray}
                  pipeline.py의 load_all_datasets()에서 반환된 것.
        descriptor_list: 실험할 descriptor 목록. None이면 config 기본값.
    """
    if descriptor_list is None:
        descriptor_list = [d for d in DESCRIPTORS if d in datasets]

    print(f"실험 대상: {descriptor_list}")
    print(f"결과 저장: {RESULTS_DIR}")
    print("=" * 80)

    # --- 실험 1: Kernel Complexity Ladder ---
    try:
        print("\n▶ 실험 1: Kernel Complexity Ladder")
        df1 = run_exp1(datasets, descriptor_list)
    except Exception as e:
        print(f"  [ERROR] 실험 1 실패: {e}")
        df1 = None

    # --- 실험 2: Classifier Hierarchy ---
    try:
        print("\n▶ 실험 2: Classifier Hierarchy")
        df2 = run_exp2(datasets, descriptor_list)
    except Exception as e:
        print(f"  [ERROR] 실험 2 실패: {e}")
        df2 = None

    # --- 실험 4: CKA ---
    try:
        print("\n▶ 실험 4: Linear vs RBF CKA")
        df4 = run_exp4(datasets, descriptor_list)
    except Exception as e:
        print(f"  [ERROR] 실험 4 실패: {e}")
        df4 = None

    # --- Reporting ---
    try:
        print("\n▶ Report 생성")
        generate_report()
    except Exception as e:
        print(f"  [ERROR] Report 생성 실패: {e}")

    print("\n" + "=" * 80)
    print("전체 실험 완료")
    print(f"결과: {RESULTS_DIR}")
    return {"exp1": df1, "exp2": df2, "exp4": df4}


# Colab에서 직접 실행 시
if __name__ == "__main__":
    # datasets 변수가 전역에 있다고 가정
    run_all(datasets)  # noqa: F821
