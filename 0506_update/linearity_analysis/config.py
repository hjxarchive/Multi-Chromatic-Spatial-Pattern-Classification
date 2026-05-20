"""공통 설정 — descriptor 목록, seed, CV 프로토콜."""
import os

DESCRIPTORS = ["Ord_PI", "Inter_PI", "3D_PI", "Sixpack_Rips", "Sixpack_Chroma"]
SEEDS = [42, 123, 456, 789, 1010]
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

BASE_DIR = "/content/drive/MyDrive/URP"
VECTOR_DIR = os.path.join(BASE_DIR, "1224_Vectors")
RESULTS_DIR = os.path.join(BASE_DIR, "0506_update", "linearity_analysis", "results")

COLORS = {
    "Ord_PI": "#4C72B0",
    "Inter_PI": "#DD8452",
    "3D_PI": "#55A868",
    "Sixpack_Rips": "#C44E52",
    "Sixpack_Chroma": "#8172B3",
}
