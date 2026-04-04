import os
import random

import numpy as np
import torch

# =========================
# DEVICE / SEED
# =========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_LABELS = 2

# =========================
# PATHS
# =========================
ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(ROOT_DIR, '..', 'dataset'))
OUTPUT_ROOT = os.path.normpath(os.path.join(ROOT_DIR, '..', 'output'))
MODELS_ROOT = os.path.normpath(os.path.join(ROOT_DIR, '..', 'models'))

# =========================
# DATA STRUCTURE
# =========================
CANONICAL_SENSORS = ('chest', 'left', 'right')
BLOCK_SIZE = 8

SCENARIOS = {
    'chest_T_1100': ['chest', 'data_time_domain.npy', (1100, 8)],
    'chest_T': ['chest', 'data_time_domain.npy', (460, 8)],
    'left_T': ['left', 'data_time_domain.npy', (460, 8)],
    'right_T': ['right', 'data_time_domain.npy', (460, 8)],
    'left_right_T': ['left_right', 'data_time_domain.npy', (460, 16)],
    'chest_left_T': ['chest_left', 'data_time_domain.npy', (460, 16)],
    'chest_right_T': ['chest_right', 'data_time_domain.npy', (460, 16)],
    'chest_left_right_T': ['chest_left_right', 'data_time_domain.npy', (460, 24)],
}

# =========================
# GLOBAL MODEL CAPACITY
# =========================
# These define comparable model size across neural architectures
MODEL_CAPACITY = {
    'width': 16,
    'depth': 2,
    'head_depth': 1,
}

# =========================
# GLOBAL TRAINING CONFIG
# =========================
TRAINING_CONFIG = {
    'epochs': 100,
    'patience': 10,
    'batch_size': 64,
    'num_workers': 0,
    'pin_memory': True,

    # GLOBAL neural training params
    'learning_rate': 1e-5,
    'weight_decay': 1e-3,
    'dropout': 0.5,

    # GLOBAL inference param
    'decision_threshold': 0.5,
}

# =========================
# MODEL-SPECIFIC PARAMS
# =========================
# Only identity parameters — no duplication of global ones
DEFAULT_PARAMS = {
    'CNN1D': {
        'model_type': 'CNN1D',
        'kernel_size': 5,
    },
    'MLP': {
        'model_type': 'MLP',
    },
    'LSTM': {
        'model_type': 'LSTM',
    },
    'GRU': {
        'model_type': 'GRU',
    },
    'LogReg': {
        'model_type': 'LogReg',
    },

    # Classical models
    'RF': {
        'model_type': 'RF',
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
    },
    'SVM': {
        'model_type': 'SVM',
        'C': 1.0,
    },
    'XGBoost': {
        'model_type': 'XGBoost',
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    'CatBoost': {
        'model_type': 'CatBoost',
        'n_estimators': 200,
        'depth': 6,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3.0,
    },
    'LogisticRegression': {
        'model_type': 'LogisticRegression',
        'C': 1.0,
    },
}

# =========================
# MODEL TYPE GROUPS
# =========================
CLASSICAL_MODELS = {'RF', 'SVM', 'XGBoost', 'CatBoost', 'LogisticRegression'}

# =========================
# RUNTIME SETUP
# =========================
def setup_runtime(seed=SEED):
    print('Using device:', DEVICE)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('GPU count:', torch.cuda.device_count())

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g