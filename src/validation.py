"""Validation and hyperparameter-search routines."""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import optuna
import optuna.visualization as vis
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from config import Config
from training import train, create_model, _make_classical_model, drop_mag_channels, keep_only_mag_channels
from test import save_results, save_results_classical, plot_loss_curve

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
            )

            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val_flat, dtype=torch.long),
                ),
                batch_size=batch_size,
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
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

    scenario_out = scenario if loss_type == "weighted" else scenario + "_NW"
    if scale:
        scenario_out = f"{scenario_out}_SC"
    if no_mag:
        scenario_out = f"{scenario_out}_NM"
    if only_mag:
        scenario_out = f"{scenario_out}_OM"
    base_out = os.path.join(Config.get_output_dir(model_type_arg, scenario_out), "nested")
    os.makedirs(base_out, exist_ok=True)

    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))

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
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_vl, dtype=torch.float32), torch.tensor(y_vl, dtype=torch.long)),
                batch_size=effective_batch_size,
                shuffle=False,
                pin_memory=Config.TRAINING_CONFIG["pin_memory"],
                num_workers=Config.TRAINING_CONFIG["num_workers"],
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

            import pandas as pd

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


def build_parser():
    parser = argparse.ArgumentParser(description="Nested validation CLI")
    parser.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
    parser.add_argument(
        "--model",
        required=False,
        choices=list(Config.DEFAULT_PARAMS.keys()),
    )
    parser.add_argument("--n_trials", type=int, default=Config.OPTUNA_CONFIG["n_trials"])
    parser.add_argument("--epochs", type=int, default=Config.TRAINING_CONFIG["epochs"])
    parser.add_argument("--inner", choices=["kfold", "holdout", "none"], default="kfold")
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Fit a StandardScaler on each training split and apply it to validation/test splits.",
    )
    parser.add_argument(
        "--no-mag",
        dest="no_mag",
        action="store_true",
        default=False,
        help="Drop the engineered magnitude channels before nested training.",
    )
    parser.add_argument(
        "--only-mag",
        dest="only_mag",
        action="store_true",
        default=False,
        help="Keep only the engineered magnitude channels before nested training.",
    )
    parser.add_argument(
        "--loss",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Loss weighting for neural models: 'weighted' uses inverse-frequency class weights; 'unweighted' uses plain CrossEntropyLoss.",
    )
    return parser


def main():
    Config.setup_device()
    Config.set_seed()

    parser = build_parser()
    args = parser.parse_args()
    run_nested_logo(args)


if __name__ == "__main__":
    main()
