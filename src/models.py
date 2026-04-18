import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import src.config as config

CLASSICAL_MODELS = config.CLASSICAL_MODELS


def create_model(model_type, input_shape, number_of_labels=1):
    from src.neural_models import create_model as create_neural_model

    return create_neural_model(model_type, input_shape, number_of_labels)


def _cuda_available():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _class_balance_stats(y_train):
    if y_train is None:
        return 1, 1, 1.0
    y_arr = np.asarray(y_train)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))
    return pos, neg, scale_pos_weight


def make_classical_model(model_type, y_train=None):
    use_gpu = _cuda_available()
    _, _, scale_pos_weight = _class_balance_stats(y_train)

    if model_type == "SVM":
        return SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            gamma="scale",
            random_state=config.SEED,
        )

    if model_type == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=config.SEED,
        )

    if model_type == "XGBoost":
        from xgboost import XGBClassifier

        xgb_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "random_state": config.SEED,
            "n_jobs": -1,
        }
        if use_gpu:
            xgb_params.update({"tree_method": "hist", "device": "cuda"})

        return XGBClassifier(**xgb_params)

    if model_type == "LogisticRegression":
        if y_train is not None:
            class_weight = {0: 1.0, 1: scale_pos_weight}
        else:
            class_weight = "balanced"

        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            class_weight=class_weight,
            random_state=config.SEED,
        )

    if model_type == "LightGBM":
        from lightgbm import LGBMClassifier

        lgbm_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "num_leaves": 31,
            "scale_pos_weight": scale_pos_weight,
            "random_state": config.SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        if use_gpu:
            lgbm_params.update({"device": "gpu"})

        return LGBMClassifier(**lgbm_params)

    if model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight={0: 1.0, 1: scale_pos_weight},
            random_state=config.SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classical model: {model_type}")
