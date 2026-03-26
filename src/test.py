import argparse
import json
import os
from collections import OrderedDict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.sensor_fusion import expand_to_canonical, scenario_output_name, sensors_from_scenario, transfer_sensor_status


def save_prediction_artifacts(
    output_dir,
    y_true,
    y_probs,
    y_pred,
    sample_indices=None,
    group_ids=None,
    window_ids=None,
    scenario_name=None,
    sensor_status=None,
):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "y_true.npy"), np.asarray(y_true))
    np.save(os.path.join(output_dir, "y_probs.npy"), np.asarray(y_probs))
    np.save(os.path.join(output_dir, "y_pred.npy"), np.asarray(y_pred))

    data = {
        "y_true": np.asarray(y_true).astype(int),
        "y_prob_0": np.asarray(y_probs)[:, 0],
        "y_prob_1": np.asarray(y_probs)[:, 1],
        "y_pred": np.asarray(y_pred).astype(int),
    }

    n = len(y_true)

    if sample_indices is not None:
        arr = np.asarray(sample_indices)
        data["sample_index"] = arr[:n]
        np.save(os.path.join(output_dir, "sample_indices.npy"), arr[:n])

    if group_ids is not None:
        arr = np.asarray(group_ids)
        data["group_id"] = arr[:n]
        np.save(os.path.join(output_dir, "group_ids.npy"), arr[:n])

    if window_ids is not None:
        arr = np.asarray(window_ids, dtype=object)
        data["window_id"] = arr[:n]
        np.save(os.path.join(output_dir, "window_ids.npy"), arr[:n])

    if scenario_name is not None:
        data["scenario"] = [scenario_name] * n

    if sensor_status is not None:
        missing = ",".join(sensor_status.get("missing", []))
        available = ",".join(sensor_status.get("available", []))
        data["missing_sensors"] = [missing] * n
        data["available_sensors"] = [available] * n

    pd.DataFrame(data).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


def _cross_sensor_output_dirs(train_scenario, test_scenario, model_type, loss_type="weighted", scale=False, no_mag=False, only_mag=False):
    train_tag = train_scenario if loss_type == "weighted" else f"{train_scenario}_NW"
    if scale:
        train_tag += "_SC"
    if no_mag:
        train_tag += "_NM"
    if only_mag:
        train_tag += "_OM"
    scenario_name = f"cross_sensor_{train_tag}_to_{test_scenario}"
    base_out = Config.get_output_dir(model_type, scenario_name)
    model_out = Config.get_models_dir(model_type, scenario_name)
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(model_out, exist_ok=True)
    return base_out, model_out


def _build_cross_sensor_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=getattr(Config, "TORCH_GENERATOR", None),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def run_cross_sensor_eval(
    train_scenario,
    model_type,
    loss_type="weighted",
    epochs=None,
    scale=False,
    no_mag=False,
    only_mag=False,
    inner_val_groups=1,
):
    """Train with outer LOGO on one sensor and test the matching held-out subject on other sensors,
    using strict alignment on (group_id, window_id) for the target test set.
    """
    from src.train import (
        _input_shape_from_data,
        _make_classical_model,
        create_model,
        train,
        drop_mag_channels,
        keep_only_mag_channels,
    )

    Config.setup_device()
    Config.set_seed()

    print(
        f"\n[Cross-Sensor Eval | LOGO] Train: {train_scenario} | Model: {model_type} | "
        f"Loss: {loss_type} | inner_val_groups={inner_val_groups}"
    )

    X_full = np.load(Config.get_data_file(train_scenario))
    y_full = np.load(Config.get_labels_file(train_scenario)).astype(np.int64)
    groups_full = np.load(Config.get_groups_file(train_scenario))
    window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(train_scenario)), "window_ids.npy")
    window_ids_full = np.load(window_ids_path, allow_pickle=True) if os.path.exists(window_ids_path) else None

    if window_ids_full is None:
        raise ValueError(
            f"Cross-sensor aligned evaluation requires window_ids.npy in train scenario: {train_scenario}"
        )

    if no_mag:
        X_full = drop_mag_channels(X_full)
    if only_mag:
        X_full = keep_only_mag_channels(X_full)

    best_params = dict(Config.DEFAULT_PARAMS[model_type])
    threshold = best_params.get("decision_threshold", 0.5)
    batch_size = Config.TRAINING_CONFIG.get("batch_size", 32)
    epochs = epochs if epochs is not None else Config.TRAINING_CONFIG.get("epochs")

    unique_subjects = np.unique(groups_full)
    print(f"Subjects (LOGO): {sorted(unique_subjects.tolist())} ({len(unique_subjects)} total)")

    allowed_pairs = {
        "left_T": ["chest_T", "right_T"],
        "right_T": ["chest_T", "left_T"],
        "chest_T": ["left_T", "right_T"],
    }

    logo = LeaveOneGroupOut()

    for test_scenario in Config.SCENARIOS:
        if test_scenario == train_scenario:
            continue
        if train_scenario in allowed_pairs and test_scenario not in allowed_pairs[train_scenario]:
            continue

        print(f"\n[Cross-Sensor Eval | LOGO] Train sensor: {train_scenario} -> Test sensor: {test_scenario}")
        X_target = np.load(Config.get_data_file(test_scenario))
        y_target = np.load(Config.get_labels_file(test_scenario)).astype(np.int64)
        groups_target = np.load(Config.get_groups_file(test_scenario))
        target_window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(test_scenario)), "window_ids.npy")
        window_ids_target = np.load(target_window_ids_path, allow_pickle=True) if os.path.exists(target_window_ids_path) else None

        if window_ids_target is None:
            raise ValueError(
                f"Cross-sensor aligned evaluation requires window_ids.npy in test scenario: {test_scenario}"
            )

        if no_mag:
            X_target = drop_mag_channels(X_target)
        if only_mag:
            X_target = keep_only_mag_channels(X_target)

        target_subjects = set(np.unique(groups_target).tolist())
        missing_subjects = sorted(set(unique_subjects.tolist()) - target_subjects)
        if missing_subjects:
            print(f"[WARNING] Subjects present in {train_scenario} but missing in {test_scenario}: {missing_subjects}")

        base_out, model_out = _cross_sensor_output_dirs(
            train_scenario,
            test_scenario,
            model_type,
            loss_type,
            scale=scale,
            no_mag=no_mag,
            only_mag=only_mag,
        )
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(model_out, exist_ok=True)

        rows = []
        n_folds = logo.get_n_splits(groups=groups_full)

        window_ids_full_arr = np.asarray(window_ids_full, dtype=object)
        window_ids_target_arr = np.asarray(window_ids_target, dtype=object)

        for fold_idx, (train_idx, test_idx_source) in enumerate(logo.split(X_full, y_full, groups_full)):
            left_out = groups_full[test_idx_source[0]]

            if left_out not in target_subjects:
                print(f"  Fold {fold_idx + 1}/{n_folds} - subject {left_out} missing in target scenario; skipping.")
                continue

            # source held-out windows for this subject
            if len(test_idx_source) == 0:
                print(f"  Fold {fold_idx + 1}/{n_folds} - no source samples for subject {left_out}; skipping.")
                continue

            # target windows for same held-out subject
            test_idx_target_subject = np.where(groups_target == left_out)[0]
            if len(test_idx_target_subject) == 0:
                print(f"  Fold {fold_idx + 1}/{n_folds} - no target samples for subject {left_out}; skipping.")
                continue

            # strict alignment on (group_id, window_id)
            src_df = pd.DataFrame({
                "src_idx": test_idx_source,
                "group_id": groups_full[test_idx_source],
                "window_id": window_ids_full_arr[test_idx_source],
                "y_src": y_full[test_idx_source],
            })

            tgt_df = pd.DataFrame({
                "tgt_idx": test_idx_target_subject,
                "group_id": groups_target[test_idx_target_subject],
                "window_id": window_ids_target_arr[test_idx_target_subject],
                "y_tgt": y_target[test_idx_target_subject],
            })

            aligned = (
                src_df.merge(tgt_df, on=["group_id", "window_id"], how="inner")
                .sort_values(["group_id", "window_id"])
                .reset_index(drop=True)
            )

            if aligned.empty:
                print(
                    f"  Fold {fold_idx + 1}/{n_folds} - no aligned windows for subject {left_out} "
                    f"between {train_scenario} and {test_scenario}; skipping."
                )
                continue

            mismatch = aligned["y_src"].to_numpy() != aligned["y_tgt"].to_numpy()
            if mismatch.any():
                bad = aligned.loc[mismatch, ["group_id", "window_id", "y_src", "y_tgt"]].head(20)
                raise ValueError(
                    f"Label mismatch after cross-sensor alignment for subject {left_out}.\n"
                    f"Examples:\n{bad.to_string(index=False)}"
                )

            test_idx_target = aligned["tgt_idx"].to_numpy(dtype=int)

            fold_label = f"s{left_out}"
            fold_dir = os.path.join(base_out, f"fold_{fold_label}")
            model_fold_dir = os.path.join(model_out, f"fold_{fold_label}")
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(model_fold_dir, exist_ok=True)

            done_marker = os.path.join(fold_dir, "done.csv")
            metrics_marker = os.path.join(fold_dir, "metrics.csv")
            if os.path.exists(done_marker) or os.path.exists(metrics_marker):
                print(f"  Fold {fold_idx + 1}/{n_folds} - {fold_label} already done; skipping.")
                marker_to_read = done_marker if os.path.exists(done_marker) else metrics_marker
                row = pd.read_csv(marker_to_read).iloc[0].to_dict()
                row["fold"] = fold_label
                rows.append(row)
                continue

            print(
                f"  Fold {fold_idx + 1}/{n_folds} - held-out subject: {left_out} | "
                f"aligned windows: {len(test_idx_target)} "
                f"(source={len(test_idx_source)}, target={len(test_idx_target_subject)})"
            )

            # optional audit trail
            pd.DataFrame([{
                "left_out_group": int(left_out),
                "n_source_subject_windows": int(len(test_idx_source)),
                "n_target_subject_windows": int(len(test_idx_target_subject)),
                "n_aligned_windows": int(len(test_idx_target)),
                "train_scenario": train_scenario,
                "test_scenario": test_scenario,
            }]).to_csv(os.path.join(fold_dir, "alignment_stats.csv"), index=False)

            # training still uses only non-held-out source-subject data
            X_train_all = X_full[train_idx]
            y_train_all = y_full[train_idx]
            groups_train = groups_full[train_idx]

            inner_subjects = np.unique(groups_train)
            n_val_groups = min(int(inner_val_groups), len(inner_subjects) - 1)
            if n_val_groups <= 0:
                raise ValueError("Cross-sensor LOGO requires at least 2 training groups in each outer fold.")

            start_idx = fold_idx % len(inner_subjects)
            val_subjects = [inner_subjects[(start_idx + k) % len(inner_subjects)] for k in range(n_val_groups)]
            val_mask = np.isin(groups_train, val_subjects)

            X_train = X_train_all[~val_mask]
            y_train = y_train_all[~val_mask]
            X_val = X_train_all[val_mask]
            y_val = y_train_all[val_mask]

            # target test set is now strictly aligned
            X_te = X_target[test_idx_target]
            y_te = y_target[test_idx_target]

            if scale:
                n_tr, t_steps, n_ch = X_train.shape
                feature_scaler = StandardScaler()
                X_train = feature_scaler.fit_transform(X_train.reshape(-1, n_ch)).reshape(n_tr, t_steps, n_ch)
                X_val = feature_scaler.transform(X_val.reshape(-1, n_ch)).reshape(X_val.shape[0], t_steps, n_ch)
                X_te = feature_scaler.transform(X_te.reshape(-1, n_ch)).reshape(X_te.shape[0], t_steps, n_ch)

            if model_type in Config.CLASSICAL_MODELS:
                X_train_flat = X_train.reshape(len(X_train), -1)
                X_te_flat = X_te.reshape(len(X_te), -1)

                clf = _make_classical_model(model_type, best_params, y_train)
                clf.fit(X_train_flat, y_train)

                save_results_classical(
                    clf=clf,
                    X_test_flat=X_te_flat,
                    y_test=y_te,
                    decision_threshold=threshold,
                    i=fold_label,
                    output_dir=fold_dir,
                    model_output_dir=model_fold_dir,
                    save_model=True,
                    sample_indices=test_idx_target,
                    group_ids=groups_target[test_idx_target],
                    window_ids=window_ids_target_arr[test_idx_target],
                    scenario_name=test_scenario,
                    sensor_status=transfer_sensor_status(train_scenario, test_scenario),
                )
            else:
                input_shape = _input_shape_from_data(X_train, model_type)
                model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
                model.to(Config.DEVICE)

                if loss_type == "weighted":
                    class_counts = np.bincount(y_train, minlength=Config.NUM_LABELS)
                    class_counts = np.maximum(class_counts, 1)
                    class_weights = len(y_train) / (Config.NUM_LABELS * class_counts.astype(float))
                    criterion = torch.nn.CrossEntropyLoss(
                        weight=torch.tensor(class_weights, dtype=torch.float32, device=Config.DEVICE)
                    )
                else:
                    criterion = torch.nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
                scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")
                train_loader, val_loader, test_loader = _build_cross_sensor_loaders(
                    X_train, y_train, X_val, y_val, X_te, y_te, batch_size
                )

                _, _, val_losses, train_losses = train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=Config.DEVICE,
                    epochs=epochs,
                    early_stopping=True,
                    patience=Config.TRAINING_CONFIG.get("patience"),
                    scaler=scaler,
                )

                plot_loss_curve(train_losses, val_losses, fold_dir, fold_label)

                save_results(
                    model=model,
                    val_loader=test_loader,
                    y_val_onehot=y_te,
                    i=fold_label,
                    decision_threshold=threshold,
                    output_dir=fold_dir,
                    device=Config.DEVICE,
                    model_output_dir=model_fold_dir,
                    save_model=True,
                    sample_indices=test_idx_target,
                    group_ids=groups_target[test_idx_target],
                    window_ids=window_ids_target_arr[test_idx_target],
                    scenario_name=test_scenario,
                    sensor_status=transfer_sensor_status(train_scenario, test_scenario),
                )

            metrics_path = os.path.join(fold_dir, "metrics.csv")
            if os.path.exists(metrics_path):
                row = pd.read_csv(metrics_path).iloc[0].to_dict()
                row["fold"] = fold_label
                rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(base_out, "summary_metrics.csv"), index=False)
            print(f"Saved LOGO cross-sensor summary to: {os.path.join(base_out, 'summary_metrics.csv')}")

    print(f"LOGO cross-sensor results saved for training sensor: {train_scenario}")




def _fused_eval_output_root(model_type, train_out, test_scenario, calibration="none", tune_threshold=False, threshold_metric="f1", threshold=0.5):
    name = f"padded_eval_{train_out}_on_{test_scenario}"
    if calibration != "none":
        name += f"_CAL_{calibration}"
    if tune_threshold:
        name += f"_TT_{threshold_metric}"
    else:
        name += f"_TH_{str(float(threshold)).replace('.', 'p')}"
    return os.path.join(Config.get_output_dir(model_type, name))


def _collect_neural_outputs(model, loader, device):
    model.eval()
    logits_list, y_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            logits_list.append(out.detach().cpu().numpy())
            y_true.extend(yb.numpy())
    logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, Config.NUM_LABELS), dtype=float)
    y_true = np.asarray(y_true)
    probs = _softmax_np(logits)
    return logits, probs, y_true


def _softmax_np(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return np.empty((0, Config.NUM_LABELS), dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.clip(np.sum(exps, axis=1, keepdims=True), 1e-12, None)


def _fit_temperature_from_logits(logits, y_true, max_iter=200, lr=0.01):
    if len(logits) == 0:
        return 1.0
    device = torch.device("cpu")
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_true, dtype=torch.long, device=device)
    log_temp = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits_t / temp, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).detach().cpu().item())


def _fit_probability_calibrator(pos_probs, y_true, method):
    pos_probs = np.asarray(pos_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=Config.SEED, solver="lbfgs")
        lr.fit(pos_probs.reshape(-1, 1), y_true)
        return {"type": "platt", "model": lr}
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pos_probs, y_true)
        return {"type": "isotonic", "model": iso}
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_probability_calibrator(pos_probs, calibrator):
    pos_probs = np.asarray(pos_probs, dtype=float)
    model = calibrator["model"]
    if calibrator["type"] == "platt":
        pos = model.predict_proba(pos_probs.reshape(-1, 1))[:, 1]
    elif calibrator["type"] == "isotonic":
        pos = model.predict(pos_probs)
    else:
        raise ValueError(f"Unsupported calibrator type: {calibrator['type']}")
    pos = np.clip(pos, 0.0, 1.0)
    return np.column_stack([1.0 - pos, pos])


def _threshold_score(y_true, y_prob_pos, threshold, metric):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob_pos) >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    if metric == "f1":
        return f1_score(y_true, y_pred, pos_label=Config.METRICS_CONFIG["fall_class"], zero_division=0)
    if metric == "balanced_accuracy":
        return 0.5 * (tpr + tnr)
    if metric == "youden":
        return tpr + tnr - 1.0
    raise ValueError(f"Unsupported threshold metric: {metric}")


def _pick_best_threshold(y_true, y_prob_pos, metric="f1"):
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best_threshold = 0.5
    best_score = float("-inf")
    for thr in candidates:
        score = _threshold_score(y_true, y_prob_pos, thr, metric)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(thr)
    return float(best_threshold), float(best_score)


def _save_eval_from_probs(
    y_true,
    y_probs,
    decision_threshold,
    output_dir,
    i,
    sample_indices=None,
    group_ids=None,
    window_ids=None,
    scenario_name=None,
    sensor_status=None,
):
    os.makedirs(output_dir, exist_ok=True)
    y_probs = np.asarray(y_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    nan_mask = ~np.isnan(y_probs).any(axis=1)
    y_probs = y_probs[nan_mask]
    y_true = y_true[nan_mask]
    if sample_indices is not None:
        sample_indices = np.asarray(sample_indices)[nan_mask]
    if group_ids is not None:
        group_ids = np.asarray(group_ids)[nan_mask]
    if window_ids is not None:
        window_ids = np.asarray(window_ids, dtype=object)[nan_mask]
    if len(y_probs) == 0:
        print(f"[WARNING] All predictions are NaN for {i}. Skipping metrics and plots.")
        return
    y_pred = (y_probs[:, 1] >= float(decision_threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir)
    plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)
    plot_precision_recall_curve(y_probs[:, 1], y_true, output_dir, i)
    metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
    record_metrics(metrics, tp, tn, fp, fn, i, output_dir)
    save_prediction_artifacts(
        output_dir=output_dir,
        y_true=y_true,
        y_probs=y_probs,
        y_pred=y_pred,
        sample_indices=sample_indices,
        group_ids=group_ids,
        window_ids=window_ids,
        scenario_name=scenario_name,
        sensor_status=sensor_status,
    )


def evaluate_padded_fused_model(
    model_type,
    train_scenario,
    test_scenario,
    loss="weighted",
    inner_val_groups=1,
    scale=False,
    no_mag=False,
    only_mag=False,
    sensor_dropout=False,
    sensor_dropout_p=0.5,
    sensor_dropout_max_off=1,
    threshold=0.5,
    tune_threshold=False,
    threshold_metric="f1",
    calibration="none",
):
    """Evaluate a fused model on valid missing-sensor subsets of its training sensor set."""
    from src.train import create_model, _input_shape_from_data, drop_mag_channels, keep_only_mag_channels

    train_sensors = tuple(sensors_from_scenario(train_scenario))
    test_sensors = tuple(sensors_from_scenario(test_scenario))
    train_shape = Config.SCENARIOS[train_scenario][2]

    allowed_targets = [
        scenario_name
        for scenario_name, (_, _, shape) in Config.SCENARIOS.items()
        if shape[0] == train_shape[0]
        and tuple(sensors_from_scenario(scenario_name)) != train_sensors
        and set(sensors_from_scenario(scenario_name)).issubset(set(train_sensors))
    ]

    if test_scenario not in allowed_targets:
        raise ValueError(
            f"Invalid fused-missing pair: {train_scenario} -> {test_scenario}. "
            f"Allowed test scenarios for {train_scenario}: {allowed_targets}"
        )

    if calibration == "temperature" and model_type in Config.CLASSICAL_MODELS:
        raise ValueError("temperature calibration is only supported for neural models")

    train_out = scenario_output_name(
        model_type,
        train_scenario,
        loss=loss,
        inner_val_groups=inner_val_groups,
        scale=scale,
        no_mag=no_mag,
        only_mag=only_mag,
        sensor_dropout=sensor_dropout,
        sensor_dropout_p=sensor_dropout_p,
        sensor_dropout_max_off=sensor_dropout_max_off,
    )
    model_root = os.path.join(Config.get_models_dir(model_type, train_out))
    output_root = _fused_eval_output_root(
        model_type=model_type,
        train_out=train_out,
        test_scenario=test_scenario,
        calibration=calibration,
        tune_threshold=tune_threshold,
        threshold_metric=threshold_metric,
        threshold=threshold,
    )
    os.makedirs(output_root, exist_ok=True)

    X_test_full = np.load(Config.get_data_file(test_scenario))
    y_test_full = np.load(Config.get_labels_file(test_scenario)).astype(np.int64)
    groups_test_full = np.load(Config.get_groups_file(test_scenario))
    window_ids_test_path = os.path.join(os.path.dirname(Config.get_labels_file(test_scenario)), "window_ids.npy")
    window_ids_test_full = np.load(window_ids_test_path, allow_pickle=True) if os.path.exists(window_ids_test_path) else None

    X_train_full = np.load(Config.get_data_file(train_scenario))
    y_train_full = np.load(Config.get_labels_file(train_scenario)).astype(np.int64)
    groups_train_full = np.load(Config.get_groups_file(train_scenario))

    if no_mag:
        X_test_full = drop_mag_channels(X_test_full)
        X_train_full = drop_mag_channels(X_train_full)
    if only_mag:
        X_test_full = keep_only_mag_channels(X_test_full)
        X_train_full = keep_only_mag_channels(X_train_full)

    X_test_full = expand_to_canonical(X_test_full, test_scenario, target_sensors=train_sensors)
    X_train_full = expand_to_canonical(X_train_full, train_scenario, target_sensors=train_sensors)

    logo = LeaveOneGroupOut()
    rows = []

    run_config = {
        "model_type": model_type,
        "train_scenario": train_scenario,
        "test_scenario": test_scenario,
        "loss": loss,
        "inner_val_groups": int(inner_val_groups),
        "scale": bool(scale),
        "no_mag": bool(no_mag),
        "only_mag": bool(only_mag),
        "sensor_dropout": bool(sensor_dropout),
        "sensor_dropout_p": float(sensor_dropout_p),
        "sensor_dropout_max_off": int(sensor_dropout_max_off),
        "threshold": float(threshold),
        "tune_threshold": bool(tune_threshold),
        "threshold_metric": threshold_metric,
        "calibration": calibration,
        "train_sensors": list(train_sensors),
        "test_sensors": list(test_sensors),
    }
    with open(os.path.join(output_root, "fused_missing_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    for fold_idx, (_, test_idx) in enumerate(logo.split(X_test_full, y_test_full, groups_test_full)):
        left_out = groups_test_full[test_idx[0]]
        fold_label = f"s{left_out}"
        fold_dir = os.path.join(output_root, f"fold_{fold_label}")
        os.makedirs(fold_dir, exist_ok=True)

        X_test = X_test_full[test_idx]
        y_test = y_test_full[test_idx]
        sample_indices = test_idx
        group_ids = groups_test_full[test_idx]
        window_ids = window_ids_test_full[test_idx] if window_ids_test_full is not None else None

        train_mask = groups_train_full != left_out
        X_fit_all = X_train_full[train_mask]
        y_fit_all = y_train_full[train_mask]
        groups_fit_all = groups_train_full[train_mask]

        inner_subjects = np.unique(groups_fit_all)
        n_val_groups = min(int(inner_val_groups), len(inner_subjects) - 1)
        if n_val_groups <= 0:
            raise ValueError("Inner validation requires at least 2 training groups in each outer fold.")
        start_idx = fold_idx % len(inner_subjects)
        val_subjects = [inner_subjects[(start_idx + k) % len(inner_subjects)] for k in range(n_val_groups)]
        val_mask = np.isin(groups_fit_all, val_subjects)
        X_val = X_fit_all[val_mask]
        y_val = y_fit_all[val_mask]
        X_scale_fit = X_fit_all[~val_mask]

        if scale:
            if X_scale_fit.shape[1:] != X_test.shape[1:]:
                raise ValueError(
                    "Shape mismatch before scaling in fused-missing evaluation: "
                    f"X_fit={X_scale_fit.shape}, X_test={X_test.shape}"
                )
            ch_fit = X_scale_fit.shape[-1]
            scaler = StandardScaler()
            scaler.fit(X_scale_fit.reshape(-1, ch_fit))
            X_test = scaler.transform(X_test.reshape(-1, ch_fit)).reshape(X_test.shape)
            X_val = scaler.transform(X_val.reshape(-1, ch_fit)).reshape(X_val.shape)

        fold_threshold = float(threshold)
        calibration_info = {"method": calibration}

        if model_type in Config.CLASSICAL_MODELS:
            model_path = os.path.join(model_root, f"fold_{fold_label}", f"model_{fold_label}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            clf = joblib.load(model_path)
            X_test_flat = X_test.reshape(len(X_test), -1)
            X_val_flat = X_val.reshape(len(X_val), -1)
            expected_features = getattr(clf, "n_features_in_", X_test_flat.shape[1])
            if X_test_flat.shape[1] != expected_features or X_val_flat.shape[1] != expected_features:
                raise ValueError(
                    f"Feature mismatch for {fold_label}: model expects {expected_features}, "
                    f"but fused-missing data produced test={X_test_flat.shape[1]}, val={X_val_flat.shape[1]} features."
                )
            test_probs = clf.predict_proba(X_test_flat)
            val_probs = clf.predict_proba(X_val_flat)
            if calibration in {"platt", "isotonic"}:
                calibrator = _fit_probability_calibrator(val_probs[:, 1], y_val, calibration)
                test_probs = _apply_probability_calibrator(test_probs[:, 1], calibrator)
                val_probs = _apply_probability_calibrator(val_probs[:, 1], calibrator)
                calibration_info["details"] = calibration
            if tune_threshold:
                fold_threshold, threshold_score = _pick_best_threshold(y_val, val_probs[:, 1], metric=threshold_metric)
                calibration_info["threshold_tuning_score"] = threshold_score
            _save_eval_from_probs(
                y_true=y_test,
                y_probs=test_probs,
                decision_threshold=fold_threshold,
                output_dir=fold_dir,
                i=fold_label,
                sample_indices=sample_indices,
                group_ids=group_ids,
                window_ids=window_ids,
                scenario_name=test_scenario,
                sensor_status=transfer_sensor_status(train_scenario, test_scenario),
            )
        else:
            input_shape = _input_shape_from_data(X_test, model_type)
            model = create_model(model_type, Config.DEFAULT_PARAMS[model_type], input_shape, Config.NUM_LABELS)
            model_path = os.path.join(model_root, f"fold_{fold_label}", f"model_{fold_label}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            model = load_model_state(model, model_path, device=str(Config.DEVICE))
            model.to(Config.DEVICE)
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.long),
                ),
                batch_size=Config.TRAINING_CONFIG["batch_size"],
                shuffle=False,
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                ),
                batch_size=Config.TRAINING_CONFIG["batch_size"],
                shuffle=False,
            )
            test_logits, test_probs, _ = _collect_neural_outputs(model, test_loader, Config.DEVICE)
            val_logits, val_probs, _ = _collect_neural_outputs(model, val_loader, Config.DEVICE)
            if calibration == "temperature":
                temperature = _fit_temperature_from_logits(val_logits, y_val)
                test_probs = _softmax_np(test_logits / temperature)
                val_probs = _softmax_np(val_logits / temperature)
                calibration_info["temperature"] = temperature
            elif calibration in {"platt", "isotonic"}:
                calibrator = _fit_probability_calibrator(val_probs[:, 1], y_val, calibration)
                test_probs = _apply_probability_calibrator(test_probs[:, 1], calibrator)
                val_probs = _apply_probability_calibrator(val_probs[:, 1], calibrator)
                calibration_info["details"] = calibration
            if tune_threshold:
                fold_threshold, threshold_score = _pick_best_threshold(y_val, val_probs[:, 1], metric=threshold_metric)
                calibration_info["threshold_tuning_score"] = threshold_score
            _save_eval_from_probs(
                y_true=y_test,
                y_probs=test_probs,
                decision_threshold=fold_threshold,
                output_dir=fold_dir,
                i=fold_label,
                sample_indices=sample_indices,
                group_ids=group_ids,
                window_ids=window_ids,
                scenario_name=test_scenario,
                sensor_status=transfer_sensor_status(train_scenario, test_scenario),
            )

        with open(os.path.join(fold_dir, "fused_missing_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold": fold_label,
                    "left_out_subject": int(left_out) if isinstance(left_out, (int, np.integer)) else str(left_out),
                    "validation_subjects": [int(v) if isinstance(v, (int, np.integer)) else str(v) for v in val_subjects],
                    "threshold": float(fold_threshold),
                    "tune_threshold": bool(tune_threshold),
                    "threshold_metric": threshold_metric,
                    "calibration": calibration_info,
                    "train_scenario": train_scenario,
                    "test_scenario": test_scenario,
                },
                f,
                indent=2,
            )

        metrics_path = os.path.join(fold_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            row = pd.read_csv(metrics_path).iloc[0].to_dict()
            row["fold"] = fold_label
            row["threshold"] = float(fold_threshold)
            row["calibration"] = calibration
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_root, "summary_metrics.csv"), index=False)
    print(f"Padded fused evaluation saved to: {output_root}")
    return output_root


def save_results(
    model,
    val_loader,
    y_val_onehot,
    i,
    decision_threshold,
    output_dir,
    device,
    model_output_dir=None,
    save_model=True,
    sample_indices=None,
    group_ids=None,
    window_ids=None,
    scenario_name=None,
    sensor_status=None,
):
    """Persist model checkpoint and full evaluation artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    if model_output_dir is None:
        model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    if save_model:
        model_path = os.path.join(model_output_dir, f"model_{i}.pt")
        torch.save(model.state_dict(), model_path)

    model.eval()
    y_probs = []
    y_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(yb.numpy())


    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    # Filter out samples with NaN predictions
    nan_mask = ~np.isnan(y_probs).any(axis=1)
    n_skipped = np.sum(~nan_mask)
    if n_skipped > 0:
        print(f"[WARNING] {n_skipped} samples with NaN predictions skipped for {i}.")
    y_probs = y_probs[nan_mask]
    y_true = y_true[nan_mask]

    if len(y_probs) == 0:
        print(f"[WARNING] All predictions are NaN for {i}. Skipping metrics and plots.")
        return

    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir)
    plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)
    plot_precision_recall_curve(y_probs[:, 1], y_true, output_dir, i)

    metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
    record_metrics(metrics, tp, tn, fp, fn, i, output_dir)
    save_prediction_artifacts(
        output_dir=output_dir,
        y_true=y_true,
        y_probs=y_probs,
        y_pred=y_pred,
        sample_indices=sample_indices,
        group_ids=group_ids,
        scenario_name=scenario_name,
        sensor_status=sensor_status,
        window_ids=window_ids,
    )
    # save_prediction_artifacts(
    #     output_dir=output_dir,
    #     y_true=y_true,
    #     y_probs=y_probs,
    #     y_pred=y_pred,
    #     sample_indices=sample_indices,
    #     group_ids=group_ids,
    #     scenario_name=scenario_name,
    #     sensor_status=sensor_status,
    #     window_ids=window_ids,
    # )


def save_results_classical(
    clf,
    X_test_flat,
    y_test,
    decision_threshold,
    i,
    output_dir,
    model_output_dir=None,
    save_model=True,
    sample_indices=None,
    group_ids=None,
    scenario_name=None,
    sensor_status=None,
    window_ids=None,
):
    """Persist and evaluate classical sklearn/XGBoost/CatBoost models."""
    os.makedirs(output_dir, exist_ok=True)
    if model_output_dir is None:
        model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    if save_model:
        joblib.dump(clf, os.path.join(model_output_dir, f"model_{i}.pkl"))

    y_probs = clf.predict_proba(X_test_flat)
    y_true = y_test

    # Filter out samples with NaN predictions
    nan_mask = ~np.isnan(y_probs).any(axis=1)
    n_skipped = np.sum(~nan_mask)
    if n_skipped > 0:
        print(f"[WARNING] {n_skipped} samples with NaN predictions skipped for {i}.")
    y_probs = y_probs[nan_mask]
    y_true = y_true[nan_mask]

    if len(y_probs) == 0:
        print(f"[WARNING] All predictions are NaN for {i}. Skipping metrics and plots.")
        return

    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # Always use the same output structure as neural networks
    save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir)
    plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)
    plot_precision_recall_curve(y_probs[:, 1], y_true, output_dir, i)

    metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
    record_metrics(metrics, tp, tn, fp, fn, i, output_dir)
    save_prediction_artifacts(
        output_dir=output_dir,
        y_true=y_true,
        y_probs=y_probs,
        y_pred=y_pred,
        sample_indices=sample_indices,
        group_ids=group_ids,
        scenario_name=scenario_name,
        sensor_status=sensor_status,
        window_ids=window_ids,
    )


def save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir):
    """Save compact binary classification metrics as CSV."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=Config.METRICS_CONFIG["fall_class"], zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    path = os.path.join(output_dir, "metrics.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("prec,rec,f1,acc,tp,fp,tn,fn\n")
        f.write(
            f"{precision:.6f},{recall:.6f},{f1:.6f},{acc:.6f},"
            f"{int(tp)},{int(fp)},{int(tn)},{int(fn)}\n"
        )


def plot_roc_curve(y_score, y_true, output_dir, i):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Modelo {i}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_model_{i}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_precision_recall_curve(y_score, y_true, output_dir, i):
    """Save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - Modelo {i}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pr_curve_model_{i}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curve(train_losses, val_losses, output_dir, model_idx):
    """Save train/validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title(f"Training and Validation Loss - Modelo {model_idx}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_model_{model_idx}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def calculate_metrics(tp, tn, fp, fn, y_true, y_pred):
    """Calculate evaluation metrics from confusion matrix counts."""
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    f1 = f1_score(y_true, y_pred, pos_label=Config.METRICS_CONFIG["fall_class"], zero_division=0)

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "f1": f1,
    }


def record_metrics(metrics, tp, tn, fp, fn, i, output_dir):
    """Per-model metrics CSV output is intentionally disabled."""
    _ = (metrics, tp, tn, fp, fn, i, output_dir)


def load_model_state(model, path, device="cpu"):
    """Load a state dict and strip DataParallel prefix if needed."""
    state_dict = torch.load(path, map_location=device)

    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


def load_hyperparameters(output_dir):
    """Load best hyperparameters json from output directory."""
    results_file = os.path.join(output_dir, "best_hyperparameters.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Arquivo de hiperparametros nao encontrado: {results_file}")
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def load_test_data(output_dir):
    """Load test arrays from saved npz file."""
    test_data_file = os.path.join(output_dir, "test_data.npz")
    if not os.path.exists(test_data_file):
        raise FileNotFoundError(f"Arquivo de dados de teste nao encontrado: {test_data_file}")
    data = np.load(test_data_file)
    return data["X_test"], data["y_test"]


def plot_learning_curve(
    create_model_fn,
    X_full,
    y_full,
    groups_full,
    X_test,
    y_test,
    input_shape,
    num_labels,
    best_params,
    device,
    output_dir,
    fractions=None,
    epochs=None,
    seed=None,
    loss_type="weighted",
):
    """Generate group-aware learning curves by varying the number of train groups."""
    if fractions is None:
        fractions = Config.LEARNING_CURVE_CONFIG["fractions"]
    if epochs is None:
        epochs = Config.LEARNING_CURVE_CONFIG["epochs"]
    if seed is None:
        seed = Config.SEED

    rng = np.random.RandomState(seed)
    results = []

    print(f"\n{'=' * 50}")
    print("INICIANDO GERACAO DA LEARNING CURVE")
    print(f"{'=' * 50}")

    unique_groups = np.unique(groups_full)
    if len(unique_groups) < 2:
        raise ValueError("Sao necessarios pelo menos 2 grupos para learning curve baseada em grupos.")

    for frac in fractions:
        n_groups = int(round(len(unique_groups) * frac))
        n_groups = max(2, min(n_groups, len(unique_groups)))
        selected_groups = rng.choice(unique_groups, size=n_groups, replace=False)
        subset_mask = np.isin(groups_full, selected_groups)

        X_subset = X_full[subset_mask]
        y_subset = y_full[subset_mask]
        groups_subset = groups_full[subset_mask]

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + int(frac * 1000))
        tr_idx, vl_idx = next(splitter.split(X_subset, y_subset, groups_subset))
        X_tr, X_vl = X_subset[tr_idx], X_subset[vl_idx]
        y_tr, y_vl = y_subset[tr_idx], y_subset[vl_idx]

        print(
            f"\nTreinando com {len(np.unique(groups_subset))} grupos "
            f"({int(frac * 100)}% dos grupos, train={len(X_tr)}, val={len(X_vl)})"
        )

        # Always use config defaults for all hyperparameters
        model_type = best_params.get("model_type", "CNN1D")
        config_params = Config.DEFAULT_PARAMS[model_type].copy()
        config_params.update(best_params)  # allow best_params to override if present
        model = create_model_fn(config_params, input_shape, num_labels)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config_params["learning_rate"])

        if loss_type == "weighted":
            class_counts = np.bincount(y_tr, minlength=num_labels)
            class_counts = np.maximum(class_counts, 1)
            class_weights = len(y_tr) / (num_labels * class_counts.astype(float))
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        from src.train import train

        batch_size = Config.TRAINING_CONFIG.get("batch_size", 32)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_vl, dtype=torch.float32),
                torch.tensor(y_vl, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        _, _, val_losses, train_losses = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs=epochs,
            early_stopping=False,
            patience=Config.TRAINING_CONFIG.get("patience"),
            scaler=None,
        )

        model.eval()
        y_preds = []
        y_true_final = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb)
                y_preds.append(torch.argmax(preds, dim=1).cpu().numpy())
                y_true_final.append(yb.numpy())

        y_preds = np.concatenate(y_preds)
        y_true_final = np.concatenate(y_true_final)
        f1 = f1_score(y_true_final, y_preds, average="macro")
        acc = accuracy_score(y_true_final, y_preds)
        train_loss_mean = float(np.mean(train_losses))
        val_loss_mean = float(np.mean(val_losses))

        results.append(
            {
                "Fraction": frac,
                "Num_Groups": int(len(np.unique(groups_subset))),
                "f1": f1,
                "Accuracy": acc,
                "Train_Loss": train_loss_mean,
                "Val_Loss": val_loss_mean,
                "Loss_Type": loss_type,
            }
        )
        print(
            f"F1: {f1:.4f} | Acc: {acc:.4f} | "
            f"Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "learning_curve_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metricas da curva de aprendizado salvas em: {csv_path}")

    plt.figure(figsize=(10, 7))
    xvals = df["Num_Groups"] if "Num_Groups" in df.columns else (df["Fraction"] * 100)
    plt.plot(xvals, df["f1"], marker="o", label="F1-score")
    plt.plot(xvals, df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(xvals, df["Train_Loss"], marker="o", label="Train Loss")
    plt.plot(xvals, df["Val_Loss"], marker="o", label="Val Loss")
    plt.xlabel("Numero de Grupos de Treino")
    plt.ylabel("Valor da Metrica")
    plt.title("Curva de Aprendizado (por grupos)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    lc_plot_path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(lc_plot_path, dpi=300)
    plt.close()
    print(f"Curva de aprendizado salva em: {lc_plot_path}")
