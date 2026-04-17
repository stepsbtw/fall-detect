import os
import random

import numpy as np
import torch

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.normpath(os.path.join(ROOT_DIR, "dataset"))
OUTPUT_ROOT = os.path.normpath(os.path.join(ROOT_DIR, "output"))
MODELS_ROOT = os.path.normpath(os.path.join(ROOT_DIR, "models"))

SEED = 42
DEFAULT_ABLATION = "acc_gyr_magacc_maggyr"
CLASSICAL_MODELS = {"LogisticRegression", "SVM", "DecisionTree", "XGBoost", "LightGBM", "RandomForest"}

DATASETS = ("chest", "left", "right", "chest_left", "chest_right", "left_right", "chest_left_right")
INPUT_SIZE = {
    "chest": (460, 8), "left": (460, 8), "right": (460, 8),
    "chest_left": (460, 16), "chest_right": (460, 16), "left_right": (460, 16),
    "chest_left_right": (460, 24),
}

NUM_WORKERS = 0
PIN_MEMORY = True
DEVICE = "cpu"
INNER_FOLDS = 3

EPOCHS = 100
PATIENCE = 5
BATCH_SIZE = 64
SENSOR_DROPOUT_P = 0.5
SENSOR_DROPOUT_MAX_OFF = 2
GPU_NORMALIZATION = True
STACKING_PRELOAD_TO_GPU = True
STACKING_BATCH_SIZE = 256
STACKING_SAVE_SUBMODEL_ARTIFACTS = True
STACKING_META_SAVE_VALIDATION_CURVE = True
STACKING_META_DIAG_EPOCHS = 40
STACKING_META_DIAG_LR = 1e-2

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
DECISION_THRESHOLD = 0.5
WIDTH = 64
DEPTH = 2
HEAD_DEPTH = 1

def seed_setup(seed=SEED):
    import torch

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Useful for deterministic CUDA kernels when supported by the runtime.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print("Using device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g