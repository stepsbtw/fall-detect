import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import optuna
import optuna.visualization as vis
from optuna.samplers import TPESampler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.train import train, create_model, _make_classical_model, drop_mag_channels, keep_only_mag_channels, _input_shape_from_data
from src.test import save_results, save_results_classical, plot_loss_curve
from src.sensor_fusion import CANONICAL_SENSORS, scenario_output_name

SCENARIO_CHOICES = list(Config.SCENARIOS.keys())


def _print_best_params(model_type, best_value, best_params):
    """Print a formatted summary of the best hyperparameters found by Optuna."""
    print(f"\n{'=' * 50}")
    print("MELHORES HIPERPARAMETROS ENCONTRADOS")
    print(f"{'=' * 50}")
    print(f"Modelo: {model_type}")
    print(f"Melhor F1: {best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()


def objective(
    trial,
    input_shape_dict,
    X_trainval,
    y_trainval,
    groups,
    output_dir,
    num_labels,
    device,
    restrict_model_type=None,
    inner_cv="kfold",
    loss_type="weighted",
    scale=False,
    no_mag=False,
    only_mag=False,
):
    """Objective function used by Optuna for inner CV."""
    print(f"\nIniciando Trial #{trial.number}\n")

    model_type = (
        restrict_model_type
        if restrict_model_type
        else trial.suggest_categorical("model_type", [m for m in Config.DEFAULT_PARAMS if m not in Config.CLASSICAL_MODELS])
    )
    is_classical = model_type in Config.CLASSICAL_MODELS

    if not is_classical:
        dropout = trial.suggest_float(
            "dropout",
            Config.METRICS_CONFIG["dropout_range"][0],
            Config.METRICS_CONFIG["dropout_range"][1],
            step=Config.METRICS_CONFIG["dropout_step"],
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            Config.OPTIMIZER_CONFIG["lr_range"][0],
            Config.OPTIMIZER_CONFIG["lr_range"][1],
            log=Config.OPTIMIZER_CONFIG["lr_log"],
        )

    decision_threshold = trial.suggest_float(
        "decision_threshold",
        Config.METRICS_CONFIG["decision_threshold_range"][0],
        Config.METRICS_CONFIG["decision_threshold_range"][1],
        step=Config.METRICS_CONFIG["decision_threshold_step"],
    )

    f1_scores = []

    if inner_cv == "none":
        all_idx = np.arange(len(X_trainval))
        splits = [(all_idx, all_idx)]
        n_folds = 1
        print(f" Trial #{trial.number} - inner_cv=none (in-sample, {len(X_trainval)} samples)")
    else:
        if inner_cv == "holdout":
            cv = GroupShuffleSplit(n_splits=1, test_size=1 / 3, random_state=Config.SEED)
        else:
            cv = GroupKFold(n_splits=3)
        n_folds = cv.get_n_splits(X_trainval, y_trainval, groups)
        splits = list(cv.split(X_trainval, y_trainval, groups))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_groups = np.unique(groups[val_idx]).tolist() if inner_cv != "none" else ["all"]
        print(f"\n Fold {fold_idx + 1}/{n_folds} - val groups {val_groups} ({model_type})")

        Config.set_seed(Config.SEED + fold_idx)

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        if no_mag:
            X_train = drop_mag_channels(X_train)
            X_val = drop_mag_channels(X_val)
        if only_mag:
            X_train = keep_only_mag_channels(X_train)
            X_val = keep_only_mag_channels(X_val)
        if scale:
            n_tr, t_steps, n_ch = X_train.shape
            feature_scaler = StandardScaler()
            X_train = feature_scaler.fit_transform(X_train.reshape(-1, n_ch)).reshape(n_tr, t_steps, n_ch)
            X_val = feature_scaler.transform(X_val.reshape(-1, n_ch)).reshape(X_val.shape[0], t_steps, n_ch)

        y_train_flat = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
        y_val_flat = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
        fall_class = Config.METRICS_CONFIG["fall_class"]
        fold_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(fold_dir, exist_ok=True)


        if is_classical:
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_val_flat = X_val.reshape(len(X_val), -1)

            if model_type == "RF":
                rf_cfg = Config.MODEL_CONFIGS["RF"]
                n_estimators = trial.suggest_int(
                    "n_estimators",
                    rf_cfg["n_estimators_range"][0],
                    rf_cfg["n_estimators_range"][1],
                    log=True,
                )
                max_depth = trial.suggest_int(
                    "max_depth",
                    rf_cfg["max_depth_range"][0],
                    rf_cfg["max_depth_range"][1],
                )
                min_samples_split = trial.suggest_int(
                    "min_samples_split",
                    rf_cfg["min_samples_split_range"][0],
                    rf_cfg["min_samples_split_range"][1],
                )
                clf = _make_classical_model(
                    "RF",
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                    },
                    y_train_flat,
                )
            elif model_type == "SVM":
                svm_cfg = Config.MODEL_CONFIGS["SVM"]
                C = trial.suggest_float("C", svm_cfg["C_range"][0], svm_cfg["C_range"][1], log=True)
                clf = _make_classical_model("SVM", {"C": C}, y_train_flat)
            elif model_type == "XGBoost":
                xgb_cfg = Config.MODEL_CONFIGS["XGBoost"]
                n_estimators = trial.suggest_int(
                    "n_estimators",
                    xgb_cfg["n_estimators_range"][0],
                    xgb_cfg["n_estimators_range"][1],
                    log=True,
                )
                max_depth = trial.suggest_int(
                    "max_depth", xgb_cfg["max_depth_range"][0], xgb_cfg["max_depth_range"][1]
                )
                learning_rate_xg = trial.suggest_float(
                    "learning_rate",
                    xgb_cfg["learning_rate_range"][0],
                    xgb_cfg["learning_rate_range"][1],
                    log=True,
                )
                subsample = trial.suggest_float(
                    "subsample", xgb_cfg["subsample_range"][0], xgb_cfg["subsample_range"][1]
                )
                colsample_bytree = trial.suggest_float(
                    "colsample_bytree",
                    xgb_cfg["colsample_bytree_range"][0],
                    xgb_cfg["colsample_bytree_range"][1],
                )
                clf = _make_classical_model(
                    "XGBoost",
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate_xg,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                    },
                    y_train_flat,
                )
            elif model_type == "CatBoost":
                cb_cfg = Config.MODEL_CONFIGS["CatBoost"]
                n_estimators = trial.suggest_int(
                    "n_estimators",
                    cb_cfg["n_estimators_range"][0],
                    cb_cfg["n_estimators_range"][1],
                    log=True,
                )
                depth = trial.suggest_int("depth", cb_cfg["depth_range"][0], cb_cfg["depth_range"][1])
                learning_rate_cb = trial.suggest_float(
                    "learning_rate",
                    cb_cfg["learning_rate_range"][0],
                    cb_cfg["learning_rate_range"][1],
                    log=True,
                )
                l2_leaf_reg = trial.suggest_float(
                    "l2_leaf_reg",
                    cb_cfg["l2_leaf_reg_range"][0],
                    cb_cfg["l2_leaf_reg_range"][1],
                    log=True,
                )
                clf = _make_classical_model(
                    "CatBoost",
                    {
                        "n_estimators": n_estimators,
                        "depth": depth,
                        "learning_rate": learning_rate_cb,
                        "l2_leaf_reg": l2_leaf_reg,
                    },
                    y_train_flat,
                )
            elif model_type == "LogisticRegression":
                lr_cfg = Config.MODEL_CONFIGS["LogisticRegression"]
                C = trial.suggest_float("C", lr_cfg["C_range"][0], lr_cfg["C_range"][1], log=True)
                clf = _make_classical_model("LogisticRegression", {"C": C}, y_train_flat)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            clf.fit(X_train_flat, y_train_flat)
            save_results_classical(
                clf,
                X_val_flat,
                y_val_flat,
                decision_threshold,
                f"{trial.number}fold{fold_idx + 1}",
                fold_dir,
            )

            y_probs_cl = clf.predict_proba(X_val_flat)
            y_pred_thresh = (y_probs_cl[:, 1] >= decision_threshold).astype(int)
            f1 = f1_score(y_val_flat, y_pred_thresh, pos_label=fall_class, zero_division=0)
            f1_scores.append(f1)

            trial.report(1.0 - f1, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        else:
            batch_size = Config.TRAINING_CONFIG["batch_size"]

            if model_type == "CNN1D":
                cnn_config = Config.MODEL_CONFIGS["CNN1D"]
                filter_size = trial.suggest_int(
                    "filter_size",
                    cnn_config["filter_size_range"][0],
                    cnn_config["filter_size_range"][1],
                    log=True,
                )
                kernel_size = trial.suggest_int(
                    "kernel_size",
                    cnn_config["kernel_size_range"][0],
                    cnn_config["kernel_size_range"][1],
                )
                num_layers = trial.suggest_int(
                    "num_layers",
                    cnn_config["num_layers_range"][0],
                    cnn_config["num_layers_range"][1],
                )
                num_dense = trial.suggest_int(
                    "num_dense_layers",
                    cnn_config["num_dense_layers_range"][0],
                    cnn_config["num_dense_layers_range"][1],
                )
                dense_neurons = trial.suggest_int(
                    "dense_neurons",
                    cnn_config["dense_neurons_range"][0],
                    cnn_config["dense_neurons_range"][1],
                    log=True,
                )

                max_seq_len = input_shape_dict["CNN1D"][0]
                reduced_seq_len = max_seq_len // (2 ** num_layers)
                if reduced_seq_len <= kernel_size:
                    raise optuna.exceptions.TrialPruned()

                model = create_model(
                    "CNN1D",
                    {
                        "filter_size": filter_size,
                        "kernel_size": kernel_size,
                        "num_layers": num_layers,
                        "num_dense_layers": num_dense,
                        "dense_neurons": dense_neurons,
                        "dropout": dropout,
                    },
                    input_shape_dict["CNN1D"],
                    num_labels,
                )

            elif model_type == "MLP":
                mlp_config = Config.MODEL_CONFIGS["MLP"]
                num_layers = trial.suggest_int(
                    "num_layers",
                    mlp_config["num_layers_range"][0],
                    mlp_config["num_layers_range"][1],
                )
                max_dense = min(
                    mlp_config["dense_neurons_range"][1],
                    max(mlp_config["dense_neurons_range"][0], input_shape_dict["MLP"] // 4),
                )
                dense_neurons = trial.suggest_int(
                    "dense_neurons",
                    mlp_config["dense_neurons_range"][0],
                    max_dense,
                    log=True,
                )
                model = create_model(
                    "MLP",
                    {
                        "num_layers": num_layers,
                        "dense_neurons": dense_neurons,
                        "dropout": dropout,
                    },
                    input_shape_dict["MLP"],
                    num_labels,
                )

            elif model_type == "LSTM":
                lstm_config = Config.MODEL_CONFIGS["LSTM"]
                hidden_dim = trial.suggest_int(
                    "hidden_dim",
                    lstm_config["hidden_dim_range"][0],
                    lstm_config["hidden_dim_range"][1],
                    log=True,
                )
                num_layers = trial.suggest_int(
                    "num_layers",
                    lstm_config["num_layers_range"][0],
                    lstm_config["num_layers_range"][1],
                )
                model = create_model(
                    "LSTM",
                    {
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                    },
                    input_shape_dict["LSTM"],
                    num_labels,
                )
            elif model_type == "GRU":
                gru_config = Config.MODEL_CONFIGS["GRU"]
                hidden_dim = trial.suggest_int(
                    "hidden_dim",
                    gru_config["hidden_dim_range"][0],
                    gru_config["hidden_dim_range"][1],
                    log=True,
                )
                num_layers = trial.suggest_int(
                    "num_layers",
                    gru_config["num_layers_range"][0],
                    gru_config["num_layers_range"][1],
                )
                model = create_model(
                    "GRU",
                    {
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                    },
                    input_shape_dict["GRU"],
                    num_labels,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.to(device)

            if torch.cuda.device_count() > 1:
                print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
                model = torch.nn.DataParallel(model)
                batch_size = batch_size * torch.cuda.device_count()
                print(f"Batch size ajustado para {batch_size} (batch_size * num_gpus)")

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            if loss_type == "weighted":
                class_counts = np.bincount(y_train_flat, minlength=num_labels)
                class_counts = np.maximum(class_counts, 1)
                class_weights = len(y_train_flat) / (num_labels * class_counts.astype(float))
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            fold_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=Config.TRAINING_CONFIG.get("patience"), min_lr=1e-6
            )

            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train_flat, dtype=torch.long),
                ),
                batch_size=batch_size,
                shuffle=Config.TRAINING_CONFIG["shuffle"],
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
                generator=getattr(Config, "TORCH_GENERATOR", None),
            )

            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val_flat, dtype=torch.long),
                ),
                batch_size=batch_size,
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
                generator=getattr(Config, "TORCH_GENERATOR", None),
            )

            scaler = torch.cuda.amp.GradScaler(enabled=getattr(device, "type", str(device)) == "cuda")

            y_pred, y_true, val_losses, train_losses = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs=Config.TRAINING_CONFIG["epochs"],
                early_stopping=Config.TRAINING_CONFIG["early_stopping"],
                patience=Config.TRAINING_CONFIG["patience"],
                scaler=scaler,
                trial=trial,
                step_offset=fold_idx * Config.TRAINING_CONFIG["epochs"],
                scheduler=fold_scheduler,
            )

            plot_loss_curve(train_losses, val_losses, fold_dir, f"{trial.number}fold{fold_idx + 1}")

            save_results(
                model=model,
                val_loader=val_loader,
                y_val_onehot=y_val,
                i=f"{trial.number}fold{fold_idx + 1}",
                decision_threshold=decision_threshold,
                output_dir=fold_dir,
                device=device,
            )

            y_probs = []
            model.eval()
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                    y_probs.extend(probs)
            y_pred_thresh = (np.array(y_probs) >= decision_threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh, pos_label=fall_class, zero_division=0)

            f1_scores.append(f1)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            del model
            del optimizer
            torch.cuda.empty_cache()

    mean_f1 = np.mean(f1_scores)
    print(f"Trial {trial.number} - Media F1 (fall): {mean_f1:.4f}")

    trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    summary = {
        "trial_number": trial.number,
        "model_type": model_type,
        "params": {"decision_threshold": decision_threshold},
        "mean_f1": float(mean_f1),
        "f1_scores": f1_scores,
    }

    if not is_classical:
        summary["params"].update({"dropout": dropout, "learning_rate": learning_rate})

    if model_type == "CNN1D":
        summary["params"].update(
            {
                "filter_size": filter_size,
                "kernel_size": kernel_size,
                "num_layers": num_layers,
                "num_dense_layers": num_dense,
                "dense_neurons": dense_neurons,
            }
        )
    elif model_type == "MLP":
        summary["params"].update({"num_layers": num_layers, "dense_neurons": dense_neurons})
    elif model_type == "LSTM":
        summary["params"].update({"hidden_dim": hidden_dim, "num_layers": num_layers})
    elif model_type == "GRU":
        summary["params"].update({"hidden_dim": hidden_dim, "num_layers": num_layers})
    elif model_type == "RF":
        summary["params"].update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            }
        )
    elif model_type == "SVM":
        summary["params"].update({"C": C})
    elif model_type == "XGBoost":
        summary["params"].update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate_xg,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
        )
    elif model_type == "CatBoost":
        summary["params"].update(
            {
                "n_estimators": n_estimators,
                "depth": depth,
                "learning_rate": learning_rate_cb,
                "l2_leaf_reg": l2_leaf_reg,
            }
        )

    with open(os.path.join(trial_dir, "trial_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return mean_f1


def run_optuna(
    input_shape_dict,
    X_trainval,
    y_trainval,
    groups,
    output_dir,
    num_labels,
    device,
    study_name,
    restrict_model_type=None,
    inner_cv="kfold",
    loss_type="weighted",
    scale=False,
    no_mag=False,
    only_mag=False,
):
    """Execute Optuna optimization and persist study artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Estudo existente carregado de: {db_path}")
    except KeyError:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=Config.SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
            load_if_exists=True,
        )
        print(f"Novo estudo criado e salvo em: {db_path}")

    study.optimize(
        lambda trial: objective(
            trial,
            input_shape_dict,
            X_trainval,
            y_trainval,
            groups,
            output_dir,
            num_labels,
            device,
            restrict_model_type,
            inner_cv,
            loss_type,
            scale,
            no_mag,
            only_mag,
        ),
        n_trials=(n_trials if n_trials is not None else Config.OPTUNA_CONFIG["n_trials"]),
        n_jobs=Config.OPTUNA_CONFIG["n_jobs"],
    )

    print("Melhor F1:", study.best_value)
    print("Melhores hiperparametros:", study.best_params)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    try:
        fig = vis.plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, "param_importance.png"))
    except Exception as e:
        print(f"Could not save importance plot: {e}")

    return study


def run_nested_logo(args):
    """Nested LOGO: outer LOGO over all subjects, inner Optuna per outer fold."""
    scenario = args.scenario
    model_type_arg = args.model
    n_trials = args.n_trials
    epochs = args.epochs
    inner_cv = args.inner
    loss_type = getattr(args, "loss", "weighted")
    scale = getattr(args, "scale", False)
    no_mag = getattr(args, "no_mag", False)
    only_mag = getattr(args, "only_mag", False)

    scenario_out = scenario_output_name(
        model_type_arg,
        scenario,
        loss=loss_type,
        inner_val_groups=1,
        scale=scale,
        no_mag=no_mag,
        only_mag=only_mag,
    )
    base_out = os.path.join(Config.get_output_dir(model_type_arg, scenario_out), "nested")
    os.makedirs(base_out, exist_ok=True)

    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))
    window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(scenario)), "window_ids.npy")
    window_ids = np.load(window_ids_path, allow_pickle=True) if os.path.exists(window_ids_path) else None

    print(f"\nNested LOGO  |  scenario={scenario}  model={model_type_arg or 'auto'}")
    print(f"Subjects: {sorted(np.unique(groups).tolist())}  ({len(np.unique(groups))} total)")
    print(f"Inner n_trials per fold: {n_trials}")
    print(f"Loss: {loss_type} | scale={scale} | no_mag={no_mag} | only_mag={only_mag}")

    input_shape_dict = Config.get_input_shape_dict(scenario, model_type_arg)
    logo_outer = LeaveOneGroupOut()
    n_outer = logo_outer.get_n_splits(groups=groups)
    batch_size = Config.TRAINING_CONFIG.get("batch_size", 32)

    for outer_idx, (inner_idx, test_idx) in enumerate(logo_outer.split(X, y, groups)):
        left_out = groups[test_idx[0]]
        print(f"\n{'=' * 60}")
        print(f"Outer fold {outer_idx + 1}/{n_outer}  -  test subject: {left_out}")
        print(f"{'=' * 60}")

        X_inner = X[inner_idx]
        y_inner = y[inner_idx]
        groups_inner = groups[inner_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]

        if no_mag:
            X_inner = drop_mag_channels(X_inner)
            X_test_fold = drop_mag_channels(X_test_fold)
        if only_mag:
            X_inner = keep_only_mag_channels(X_inner)
            X_test_fold = keep_only_mag_channels(X_test_fold)

        fold_dir = os.path.join(base_out, f"outer_s{left_out}")
        os.makedirs(fold_dir, exist_ok=True)

        study_name = (
            f"{scenario}_{model_type_arg}_outer_s{left_out}"
            if model_type_arg
            else f"{scenario}_outer_s{left_out}"
        )
        study = run_optuna(
            input_shape_dict=input_shape_dict,
            X_trainval=X_inner,
            y_trainval=y_inner,
            groups=groups_inner,
            output_dir=fold_dir,
            num_labels=Config.NUM_LABELS,
            device=Config.DEVICE,
            restrict_model_type=model_type_arg,
            study_name=study_name,
            inner_cv=inner_cv,
            loss_type=loss_type,
            n_trials=n_trials,
            scale=scale,
            no_mag=no_mag,
            only_mag=only_mag,
        )

        best_params = study.best_params
        model_type = best_params["model_type"] if not model_type_arg else model_type_arg
        threshold = best_params.get("decision_threshold", 0.5)

        with open(os.path.join(fold_dir, "best_hyperparameters.json"), "w") as f:
            json.dump(
                {
                    "outer_subject": int(left_out),
                    "model_type": model_type,
                    "best_value": float(study.best_value),
                    "best_params": best_params,
                    "n_trials": len(study.trials),
                    "optimization_history": [t.value for t in study.trials if t.value is not None],
                },
                f,
                indent=2,
            )

        study.trials_dataframe().to_csv(os.path.join(fold_dir, "optuna_trials.csv"), index=False)

        try:
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(fold_dir, "param_importance.png"))
        except Exception as e:
            print(f"  [AVISO] Nao foi possivel salvar param_importance.png: {e}")

        _print_best_params(model_type, study.best_value, best_params)

        if model_type in Config.CLASSICAL_MODELS:
            X_inner_fit = X_inner
            X_test_eval = X_test_fold
            if scale:
                n_in, t_steps, n_ch = X_inner.shape
                feature_scaler = StandardScaler()
                X_inner_fit = feature_scaler.fit_transform(X_inner.reshape(-1, n_ch)).reshape(n_in, t_steps, n_ch)
                X_test_eval = feature_scaler.transform(X_test_fold.reshape(-1, n_ch)).reshape(X_test_fold.shape[0], t_steps, n_ch)
            X_tr_flat = X_inner_fit.reshape(len(X_inner_fit), -1)
            X_te_flat = X_test_eval.reshape(len(X_test_eval), -1)
            clf = _make_classical_model(model_type, best_params, y_inner)
            clf.fit(X_tr_flat, y_inner)
            save_results_classical(
                clf=clf,
                X_test_flat=X_te_flat,
                y_test=y_test_fold,
                decision_threshold=threshold,
                i=f"outer_s{left_out}",
                output_dir=fold_dir,
            )
        else:
            input_shape = input_shape_dict[model_type]

            inner_groups = np.unique(groups_inner)
            val_subject = inner_groups[outer_idx % len(inner_groups)]
            val_mask = groups_inner == val_subject
            X_tr = X_inner[~val_mask]
            y_tr = y_inner[~val_mask]
            X_vl = X_inner[val_mask]
            y_vl = y_inner[val_mask]

            if scale:
                n_tr, t_steps, n_ch = X_tr.shape
                feature_scaler = StandardScaler()
                X_tr = feature_scaler.fit_transform(X_tr.reshape(-1, n_ch)).reshape(n_tr, t_steps, n_ch)
                X_vl = feature_scaler.transform(X_vl.reshape(-1, n_ch)).reshape(X_vl.shape[0], t_steps, n_ch)
                X_test_fold = feature_scaler.transform(X_test_fold.reshape(-1, n_ch)).reshape(X_test_fold.shape[0], t_steps, n_ch)

            model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
            model.to(Config.DEVICE)

            effective_batch_size = batch_size
            if torch.cuda.device_count() > 1 and Config.DEVICE.type == "cuda":
                print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
                model = torch.nn.DataParallel(model)
                effective_batch_size = batch_size * torch.cuda.device_count()
                print(f"Batch size ajustado para {effective_batch_size} (batch_size * num_gpus)")

            optimizer = torch.optim.Adam(
                model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=Config.TRAINING_CONFIG.get("patience"), min_lr=1e-6
            )
            if loss_type == "weighted":
                class_counts = np.bincount(y_tr, minlength=Config.NUM_LABELS)
                class_counts = np.maximum(class_counts, 1)
                class_weights = len(y_tr) / (Config.NUM_LABELS * class_counts.astype(float))
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
                criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
                batch_size=effective_batch_size,
                shuffle=Config.TRAINING_CONFIG["shuffle"],
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
                generator=getattr(Config, "TORCH_GENERATOR", None),
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_vl, dtype=torch.float32), torch.tensor(y_vl, dtype=torch.long)),
                batch_size=effective_batch_size,
                shuffle=False,
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
                generator=getattr(Config, "TORCH_GENERATOR", None),
            )
            test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_test_fold, dtype=torch.float32),
                    torch.tensor(y_test_fold, dtype=torch.long),
                ),
                batch_size=effective_batch_size,
                shuffle=False,
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
                generator=getattr(Config, "TORCH_GENERATOR", None),
            )

            scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")

            fold_label = f"outer_s{left_out}"
            _, _, val_losses, train_losses = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                Config.DEVICE,
                epochs=epochs,
                early_stopping=True,
                patience=Config.TRAINING_CONFIG["patience"],
                scaler=scaler,
                scheduler=scheduler,
            )

            plot_loss_curve(train_losses, val_losses, fold_dir, fold_label)

            pd.DataFrame(
                {
                    "epoch": range(1, len(train_losses) + 1),
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                }
            ).to_csv(os.path.join(fold_dir, f"losses_{fold_label}.csv"), index=False)

            save_results(
                model=model,
                val_loader=test_loader,
                y_val_onehot=y_test_fold,
                i=fold_label,
                decision_threshold=threshold,
                output_dir=fold_dir,
                device=Config.DEVICE,
            )

        print(f"  Outer fold s{left_out} concluido - salvo em {fold_dir}")

    print(f"\nNested LOGO concluido! Resultados em: {base_out}")



BASE_SCENARIOS = {
    "chest": "chest_T",
    "left": "left_T",
    "right": "right_T",
}


def _select_validation_subjects(groups_train, fold_idx, inner_val_groups):
    inner_subjects = np.unique(groups_train)
    n_val_groups = min(int(inner_val_groups), max(len(inner_subjects) - 1, 0))
    if n_val_groups <= 0:
        raise ValueError("At least 2 training groups are required to create an inner validation split.")
    start_idx = int(fold_idx) % len(inner_subjects)
    return [inner_subjects[(start_idx + k) % len(inner_subjects)] for k in range(n_val_groups)]


def _load_sensor_arrays(sensor_name, args):
    scenario = BASE_SCENARIOS[sensor_name]
    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))
    window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(scenario)), "window_ids.npy")
    if not os.path.exists(window_ids_path):
        raise FileNotFoundError(f"Missing window_ids.npy for scenario '{scenario}'")
    window_ids = np.load(window_ids_path, allow_pickle=True)

    if getattr(args, "no_mag", False):
        X = drop_mag_channels(X)
    if getattr(args, "only_mag", False):
        X = keep_only_mag_channels(X)

    return {
        "sensor": sensor_name,
        "scenario": scenario,
        "X": X,
        "y": y,
        "groups": groups,
        "window_ids": window_ids,
    }


def _predict_positive_proba_from_model(model, model_type, X, device=None):
    if model_type in Config.CLASSICAL_MODELS:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return probs[:, 1].astype(float)
            return probs.reshape(-1).astype(float)
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-scores))
        raise ValueError(f"Model {model_type} does not expose predict_proba or decision_function")

    device = device or Config.DEVICE
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X), dtype=torch.long)),
        batch_size=Config.TRAINING_CONFIG.get("batch_size", 32),
        shuffle=False,
        pin_memory=Config.TRAINING_CONFIG["pin_memory"],
        num_workers=Config.TRAINING_CONFIG["num_workers"],
    )
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(batch_probs)
    return np.concatenate(probs).astype(float)


def _fit_sensor_model_and_predict(sensor_bundle, train_idx, predict_idx, args, fit_fold_idx):
    model_type = args.model
    X = sensor_bundle["X"]
    y = sensor_bundle["y"]
    groups = sensor_bundle["groups"]
    best_params = Config.DEFAULT_PARAMS[model_type]

    X_train_all = X[train_idx]
    y_train_all = y[train_idx]
    groups_train = groups[train_idx]

    val_subjects = _select_validation_subjects(groups_train, fit_fold_idx, getattr(args, "inner_val_groups", 1))
    val_mask = np.isin(groups_train, val_subjects)
    train_mask = ~val_mask
    if not bool(train_mask.any()):
        raise ValueError("Validation split consumed all training samples.")

    X_train = X_train_all[train_mask]
    y_train = y_train_all[train_mask]
    X_val = X_train_all[val_mask]
    y_val = y_train_all[val_mask]
    X_pred = X[predict_idx]

    if model_type in Config.CLASSICAL_MODELS:
        X_train_fit = X_train.reshape(len(X_train), -1)
        X_pred_fit = X_pred.reshape(len(X_pred), -1)
        if getattr(args, "scale", False):
            scaler = StandardScaler()
            X_train_fit = scaler.fit_transform(X_train_fit)
            X_pred_fit = scaler.transform(X_pred_fit)
        clf = _make_classical_model(model_type, best_params, y_train)
        clf.fit(X_train_fit, y_train)
        probs = _predict_positive_proba_from_model(clf, model_type, X_pred_fit)
        return probs

    X_val_fit = X_val
    if getattr(args, "scale", False):
        n_tr, t_steps, n_ch = X_train.shape
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, n_ch)).reshape(n_tr, t_steps, n_ch)
        X_val_fit = scaler.transform(X_val.reshape(-1, n_ch)).reshape(X_val.shape[0], t_steps, n_ch)
        X_pred = scaler.transform(X_pred.reshape(-1, n_ch)).reshape(X_pred.shape[0], t_steps, n_ch)

    input_shape = Config.get_input_shape_dict(sensor_bundle["scenario"], model_type)[model_type]
    if getattr(args, "no_mag", False) or getattr(args, "only_mag", False):
        input_shape = _input_shape_from_data(X_train, model_type)

    Config.set_seed(Config.SEED + int(fit_fold_idx))
    model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
    model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=Config.TRAINING_CONFIG.get("patience"),
        min_lr=1e-6,
    )
    if getattr(args, "loss", "weighted") == "weighted":
        class_counts = np.bincount(y_train, minlength=Config.NUM_LABELS)
        class_counts = np.maximum(class_counts, 1)
        class_weights = len(y_train) / (Config.NUM_LABELS * class_counts.astype(float))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    batch_size = Config.TRAINING_CONFIG.get("batch_size", 32)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=Config.TRAINING_CONFIG["shuffle"],
        pin_memory=Config.TRAINING_CONFIG["pin_memory"],
        num_workers=Config.TRAINING_CONFIG["num_workers"],
        generator=getattr(Config, "TORCH_GENERATOR", None),
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_fit, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=Config.TRAINING_CONFIG["pin_memory"],
        num_workers=Config.TRAINING_CONFIG["num_workers"],
        generator=getattr(Config, "TORCH_GENERATOR", None),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        Config.DEVICE,
        epochs=getattr(args, "epochs", Config.TRAINING_CONFIG.get("epochs")),
        early_stopping=True,
        patience=Config.TRAINING_CONFIG.get("patience"),
        scaler=scaler,
        scheduler=scheduler,
    )
    return _predict_positive_proba_from_model(model, model_type, X_pred, device=Config.DEVICE)


def _merge_sensor_prediction_frames(sensor_frames):
    merged = None
    for sensor, df in sensor_frames.items():
        current = df.rename(columns={"y_true": f"y_true_{sensor}"})
        merged = current if merged is None else merged.merge(current, on=["window_id", "group_id"], how="inner")
    if merged is None or merged.empty:
        raise ValueError("No aligned per-sensor samples were found while building stacking metadata.")
    ref_sensor = CANONICAL_SENSORS[0]
    ref_col = f"y_true_{ref_sensor}"
    for sensor in CANONICAL_SENSORS[1:]:
        col = f"y_true_{sensor}"
        mismatch = merged[ref_col].to_numpy() != merged[col].to_numpy()
        if mismatch.any():
            bad = merged.loc[mismatch, ["group_id", "window_id", ref_col, col]].head(20)
            raise ValueError(
                f"y_true mismatch between {ref_sensor} and {sensor} after window_id alignment.\nExamples:\n{bad.to_string(index=False)}"
            )
    merged["y_true"] = merged[ref_col].astype(int)
    sort_cols = ["group_id", "window_id"]
    sample_idx_col = "sample_index_chest"
    if sample_idx_col in merged.columns:
        sort_cols.append(sample_idx_col)
    return merged.sort_values(sort_cols).reset_index(drop=True)


def _collect_sensor_predictions(sensor_data, args, fit_specs, predict_subjects=None):
    sensor_frames = {}
    for sensor, bundle in sensor_data.items():
        rows = []
        for local_fit_idx, spec in enumerate(fit_specs):
            train_idx = spec["train_idx"]
            pred_idx = spec["predict_idx"]
            probs = _fit_sensor_model_and_predict(
                bundle,
                train_idx=train_idx,
                predict_idx=pred_idx,
                args=args,
                fit_fold_idx=spec.get("fit_fold_idx", local_fit_idx),
            )
            frame = pd.DataFrame(
                {
                    "window_id": bundle["window_ids"][pred_idx].astype(object),
                    "group_id": bundle["groups"][pred_idx],
                    "y_true": bundle["y"][pred_idx].astype(int),
                    f"p_{sensor}": probs,
                    f"sample_index_{sensor}": pred_idx,
                }
            )
            rows.append(frame)
        sensor_df = pd.concat(rows, ignore_index=True)
        if predict_subjects is not None:
            sensor_df = sensor_df[np.isin(sensor_df["group_id"], np.asarray(predict_subjects))].copy()
        sensor_frames[sensor] = sensor_df.sort_values(["group_id", "window_id"]).reset_index(drop=True)
    report_window_id_overlap(sensor_frames)
    return _merge_sensor_prediction_frames(sensor_frames)


def _build_honest_stacking_matrices(sensor_data, args, outer_train_groups, outer_test_group, outer_fold_idx):
    outer_train_groups = np.asarray(sorted(np.unique(outer_train_groups)))
    if outer_train_groups.size < 2:
        raise ValueError("Need at least two outer-train groups to construct honest stacking features.")

    full_groups = next(iter(sensor_data.values()))["groups"]

    # Fast honest stacking:
    # use a group-based inner holdout (default: 1 subject, matching the main training pipeline)
    # to generate meta-train features, then refit on all outer-train groups for the outer test subject.
    n_meta_groups = int(getattr(args, "inner_val_groups", 1) or 1)
    n_meta_groups = max(1, min(n_meta_groups, outer_train_groups.size - 1))

    rng = np.random.default_rng(Config.SEED + int(outer_fold_idx))
    held_out_meta_groups = np.sort(rng.choice(outer_train_groups, size=n_meta_groups, replace=False))
    base_train_groups = outer_train_groups[~np.isin(outer_train_groups, held_out_meta_groups)]

    train_idx = np.where(np.isin(full_groups, base_train_groups))[0]
    predict_idx = np.where(np.isin(full_groups, held_out_meta_groups))[0]
    fit_specs_train = [{
        "train_idx": train_idx,
        "predict_idx": predict_idx,
        "fit_fold_idx": outer_fold_idx * 100 + 1,
    }]
    train_df = _collect_sensor_predictions(sensor_data, args, fit_specs_train, predict_subjects=held_out_meta_groups)

    full_train_idx = np.where(np.isin(full_groups, outer_train_groups))[0]
    outer_test_idx = np.where(full_groups == outer_test_group)[0]
    fit_specs_test = [{
        "train_idx": full_train_idx,
        "predict_idx": outer_test_idx,
        "fit_fold_idx": outer_fold_idx * 100 + 99,
    }]
    test_df = _collect_sensor_predictions(sensor_data, args, fit_specs_test, predict_subjects=[outer_test_group])
    return train_df, test_df


def load_predictions_for_sensor(model, sensor_name, args):
    scenario = BASE_SCENARIOS[sensor_name]
    scenario_out = scenario_output_name(
        model=model,
        scenario=scenario,
        loss=args.loss,
        inner_val_groups=args.inner_val_groups,
        scale=args.scale,
        no_mag=args.no_mag,
        only_mag=args.only_mag,
    )

    base_dir = os.path.join(Config.get_output_dir(model, scenario_out))
    fold_files = sorted(
        os.path.join(base_dir, fp)
        for fp in os.listdir(base_dir)
        if fp.startswith("fold_") and os.path.exists(os.path.join(base_dir, fp, "predictions.csv"))
    ) if os.path.exists(base_dir) else []
    if not fold_files:
        raise FileNotFoundError(f"No predictions.csv files found for {sensor_name} at {base_dir}")

    frames = []
    for fold_dir in fold_files:
        fp = os.path.join(fold_dir, "predictions.csv")
        df = pd.read_csv(fp)
        missing = {"window_id", "group_id", "y_true", "y_prob_1"} - set(df.columns)
        if missing:
            raise ValueError(f"{fp} is missing required columns: {sorted(missing)}")
        keep_cols = ["window_id", "group_id", "y_true", "y_prob_1"]
        if "sample_index" in df.columns:
            keep_cols.append("sample_index")
        df = df[keep_cols].copy()
        df = df.rename(columns={"y_prob_1": f"p_{sensor_name}"})
        if "sample_index" in df.columns:
            df = df.rename(columns={"sample_index": f"sample_index_{sensor_name}"})
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["group_id", "window_id"]).drop_duplicates(["group_id", "window_id"], keep="last")
    return out, scenario_out


def report_window_id_overlap(sensor_frames):
    sensor_names = list(sensor_frames.keys())
    if len(sensor_names) < 2:
        return
    for i in range(len(sensor_names)):
        for j in range(i + 1, len(sensor_names)):
            a = sensor_names[i]
            b = sensor_names[j]
            a_ids = set(sensor_frames[a]["window_id"].astype(str))
            b_ids = set(sensor_frames[b]["window_id"].astype(str))
            common = len(a_ids & b_ids)
            only_a = len(a_ids - b_ids)
            only_b = len(b_ids - a_ids)
            print(f"[window_id overlap] {a} vs {b}: common={common} only_{a}={only_a} only_{b}={only_b}")


def build_multisensor_meta_dataframe(model, args):
    merged = None
    scenario_tags = {}
    sensor_frames = {}
    for sensor in CANONICAL_SENSORS:
        df, tag = load_predictions_for_sensor(model, sensor, args)
        scenario_tags[sensor] = tag
        sensor_frames[sensor] = df
        df = df.rename(columns={"y_true": f"y_true_{sensor}"})
        merged = df if merged is None else merged.merge(df, on=["window_id", "group_id"], how="inner")
    report_window_id_overlap(sensor_frames)
    if merged is None or merged.empty:
        raise ValueError("No aligned per-sensor predictions were found.")
    ref_sensor = CANONICAL_SENSORS[0]
    ref_col = f"y_true_{ref_sensor}"
    for sensor in CANONICAL_SENSORS[1:]:
        col = f"y_true_{sensor}"
        mismatch = merged[ref_col].to_numpy() != merged[col].to_numpy()
        if mismatch.any():
            bad = merged.loc[mismatch, ["group_id", "window_id", ref_col, col]].head(20)
            raise ValueError(
                f"y_true mismatch between {ref_sensor} and {sensor} after window_id alignment.\nExamples:\n{bad.to_string(index=False)}"
            )
    merged["y_true"] = merged[ref_col].astype(int)
    sort_cols = ["group_id", "window_id"]
    if "sample_index_chest" in merged.columns:
        sort_cols.append("sample_index_chest")
    return merged.sort_values(sort_cols).reset_index(drop=True), scenario_tags


def multisensor_conditions():
    return {
        "all_present": ["chest", "left", "right"],
        "missing_chest": ["left", "right"],
        "missing_left": ["chest", "right"],
        "missing_right": ["chest", "left"],
        "only_chest": ["chest"],
        "only_left": ["left"],
        "only_right": ["right"],
    }


def compute_multisensor_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }, y_pred


def save_multisensor_outputs(output_dir, condition_name, df, y_prob, threshold=0.5, prob_col="y_prob_fused", pred_col="y_pred_fused"):
    os.makedirs(output_dir, exist_ok=True)
    metrics, y_pred = compute_multisensor_metrics(df["y_true"].to_numpy(), y_prob, threshold=threshold)
    per_sample = df.copy()
    per_sample[prob_col] = y_prob
    per_sample[pred_col] = y_pred
    per_sample.to_csv(os.path.join(output_dir, f"predictions_{condition_name}.csv"), index=False)
    pd.DataFrame([{"condition": condition_name, **metrics}]).to_csv(
        os.path.join(output_dir, f"metrics_{condition_name}.csv"), index=False
    )
    return metrics


def run_multisensor_ensemble(args):
    df, scenario_tags = build_multisensor_meta_dataframe(args.model, args)
    threshold = float(args.threshold)
    output_dir = os.path.join(Config.get_output_dir(args.model, f"multisensor_ensemble_{args.tag}"))
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for condition_name, available in multisensor_conditions().items():
        probs = df[[f"p_{sensor}" for sensor in available]].mean(axis=1).to_numpy()
        metrics = save_multisensor_outputs(output_dir, condition_name, df, probs, threshold=threshold)
        rows.append({"method": "ensemble", "condition": condition_name, "available_sensors": ",".join(available), **metrics})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "threshold": threshold, "source_scenarios": scenario_tags}, f, indent=2)
    print(f"Ensemble results saved to: {output_dir}")


def prepare_stacking_features(df, available):
    return np.column_stack([
        df["p_chest"].to_numpy() if "chest" in available else np.zeros(len(df), dtype=float),
        df["p_left"].to_numpy() if "left" in available else np.zeros(len(df), dtype=float),
        df["p_right"].to_numpy() if "right" in available else np.zeros(len(df), dtype=float),
        np.full(len(df), 1.0 if "chest" in available else 0.0),
        np.full(len(df), 1.0 if "left" in available else 0.0),
        np.full(len(df), 1.0 if "right" in available else 0.0),
    ])


def run_multisensor_stacking(args):
    from sklearn.linear_model import LogisticRegression
    import joblib

    sensor_data = {sensor: _load_sensor_arrays(sensor, args) for sensor in CANONICAL_SENSORS}
    scenario_tags = {sensor: bundle["scenario"] for sensor, bundle in sensor_data.items()}
    output_dir = os.path.join(Config.get_output_dir(args.model, f"multisensor_stacking_{args.tag}"))
    os.makedirs(output_dir, exist_ok=True)
    threshold = float(args.threshold)

    reference_groups = sensor_data[CANONICAL_SENSORS[0]]["groups"]
    all_groups = np.unique(reference_groups)
    summary_rows = []
    condition_frames = {name: [] for name in multisensor_conditions()}
    condition_fold_rows = {name: [] for name in multisensor_conditions()}

    for outer_fold_idx, outer_test_group in enumerate(all_groups):
        print(f"[stacking] outer fold {outer_fold_idx + 1}/{len(all_groups)} | left-out group={int(outer_test_group)}")
        outer_train_groups = all_groups[all_groups != outer_test_group]
        meta_train_df, meta_test_df = _build_honest_stacking_matrices(
            sensor_data,
            args,
            outer_train_groups=outer_train_groups,
            outer_test_group=outer_test_group,
            outer_fold_idx=outer_fold_idx,
        )

        y_train = meta_train_df["y_true"].to_numpy().astype(int)
        y_test = meta_test_df["y_true"].to_numpy().astype(int)

        for condition_name, available in multisensor_conditions().items():
            X_train = prepare_stacking_features(meta_train_df, available)
            X_test = prepare_stacking_features(meta_test_df, available)
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=Config.SEED + outer_fold_idx,
            )
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= threshold).astype(int)
            metrics, _ = compute_multisensor_metrics(y_test, probs, threshold=threshold)
            out_df = meta_test_df.copy()
            out_df["condition"] = condition_name
            out_df["available_sensors"] = ",".join(available)
            out_df["outer_fold"] = outer_fold_idx + 1
            out_df["left_out_group"] = int(outer_test_group)
            out_df["y_prob_stacked"] = probs
            out_df["y_pred_stacked"] = preds
            condition_frames[condition_name].append(out_df)
            condition_fold_rows[condition_name].append({
                "fold": outer_fold_idx + 1,
                "left_out_group": int(outer_test_group),
                "available_sensors": ",".join(available),
                **metrics,
            })

    for condition_name, available in multisensor_conditions().items():
        if not condition_frames[condition_name]:
            raise ValueError(f"No stacking predictions were produced for condition '{condition_name}'")
        out_df = pd.concat(condition_frames[condition_name], ignore_index=True)
        out_df = out_df.sort_values(["group_id", "window_id"]).reset_index(drop=True)
        out_df.to_csv(os.path.join(output_dir, f"predictions_{condition_name}.csv"), index=False)
        pd.DataFrame(condition_fold_rows[condition_name]).to_csv(
            os.path.join(output_dir, f"fold_metrics_{condition_name}.csv"), index=False
        )
        metrics, _ = compute_multisensor_metrics(
            out_df["y_true"].to_numpy().astype(int),
            out_df["y_prob_stacked"].to_numpy(dtype=float),
            threshold=threshold,
        )
        summary_rows.append({
            "method": "stacking",
            "condition": condition_name,
            "available_sensors": ",".join(available),
            **metrics,
        })

        full_meta_df = _collect_sensor_predictions(
            sensor_data,
            args,
            fit_specs=[{
                "train_idx": np.where(sensor_data[CANONICAL_SENSORS[0]]["groups"] != left_out)[0],
                "predict_idx": np.where(sensor_data[CANONICAL_SENSORS[0]]["groups"] == left_out)[0],
                "fit_fold_idx": 10_000 + idx,
            } for idx, left_out in enumerate(all_groups)],
            predict_subjects=all_groups,
        )
        X_full = prepare_stacking_features(full_meta_df, available)
        y_full = full_meta_df["y_true"].to_numpy().astype(int)
        final_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=Config.SEED)
        final_clf.fit(X_full, y_full)
        joblib.dump(final_clf, os.path.join(output_dir, f"stacker_{condition_name}.pkl"))

    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "threshold": threshold,
                "source_scenarios": scenario_tags,
                "stacking_strategy": "retrained_group_holdout",
                "meta_features": ["p_chest", "p_left", "p_right", "flag_chest", "flag_left", "flag_right"],
            },
            f,
            indent=2,
        )
    print(f"Stacking results saved to: {output_dir}")

def build_parser():
    parser = argparse.ArgumentParser(description="Nested validation and multisensor evaluation CLI")
    parser.add_argument("-scenario", required=False, choices=SCENARIO_CHOICES)
    parser.add_argument(
        "--model",
        required=False,
        choices=list(Config.DEFAULT_PARAMS.keys()) + ["LogisticRegression"],
    )
    parser.add_argument("--n_trials", type=int, default=Config.OPTUNA_CONFIG["n_trials"])
    parser.add_argument("--epochs", type=int, default=Config.TRAINING_CONFIG["epochs"])
    parser.add_argument("--inner", choices=["kfold", "holdout", "none"], default="kfold")
    parser.add_argument("--scale", action="store_true", default=False)
    parser.add_argument("--no-mag", dest="no_mag", action="store_true", default=False)
    parser.add_argument("--only-mag", dest="only_mag", action="store_true", default=False)
    parser.add_argument("--loss", choices=["weighted", "unweighted"], default="weighted")
    parser.add_argument("--multisensor-mode", choices=["ensemble", "stacking", "all"], default=None)
    parser.add_argument("--inner-val-groups", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tag", default="default")
    return parser



def run_ensemble(**kwargs):
    """Backward-compatible wrapper used by run.py."""
    args = argparse.Namespace(**kwargs)
    return run_multisensor_ensemble(args)


def run_stacking(**kwargs):
    """Backward-compatible wrapper used by run.py."""
    args = argparse.Namespace(**kwargs)
    return run_multisensor_stacking(args)

def main(args=None):
    Config.setup_device()
    Config.set_seed()

    parser = build_parser()
    if args is None:
        args = parser.parse_args()

    if args.multisensor_mode:
        if not args.model:
            raise ValueError("--model is required when using --multisensor-mode")
        if args.multisensor_mode in {"ensemble", "all"}:
            run_multisensor_ensemble(args)
        if args.multisensor_mode in {"stacking", "all"}:
            run_multisensor_stacking(args)
        return

    if not args.scenario:
        raise ValueError("-scenario is required unless --multisensor-mode is used")
    run_nested_logo(args)


if __name__ == "__main__":
    main()
