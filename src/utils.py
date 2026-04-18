import os

import numpy as np

import src.config as config

ROOT_DIR = config.ROOT_DIR
DATA_PATH = config.DATA_PATH
OUTPUT_ROOT = config.OUTPUT_ROOT
MODELS_ROOT = config.MODELS_ROOT
INPUT_SIZE = config.INPUT_SIZE

CLASSICAL_MODELS = config.CLASSICAL_MODELS
SCENARIOS = {name: (name, "data_time_domain.npy", shape) for name, shape in INPUT_SIZE.items()}


def sensors_from_experiment(experiment):
    sensors = [p for p in experiment.split("_") if p in ("chest", "left", "right")]
    return sensors

def build_run_name(dataset_name, sensor_dropout=False, ablation=config.DEFAULT_ABLATION):
    name = str(dataset_name or "")
    if bool(sensor_dropout):
        name += "_SDP"
    ablation = str(ablation or "")
    if ablation and ablation != config.DEFAULT_ABLATION:
        name += f"_{ablation}"
    return name

def run_name_for_dataset(dataset_name, args):
    return build_run_name(
        dataset_name,
        sensor_dropout=getattr(args, "sensor_dropout", False),
        ablation=getattr(args, "ablation", config.DEFAULT_ABLATION),
    )

def model_input_shape(model_name, X):
    if model_name == "MLP":
        return int(X.shape[1] * X.shape[2])
    return (int(X.shape[1]), int(X.shape[2]))

def estimator_binary_prob_1(model, X):
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Estimator must provide predict_proba or decision_function for probability inference.")

def threshold_from_metrics_csv(metrics_path, default_threshold=config.DECISION_THRESHOLD):
    import pandas as pd

    threshold = float(default_threshold)
    if not os.path.exists(metrics_path):
        return threshold

    metrics_df = pd.read_csv(metrics_path)
    if "threshold" not in metrics_df.columns or len(metrics_df) == 0:
        return threshold

    thr = metrics_df.iloc[0]["threshold"]
    if pd.notna(thr):
        threshold = float(thr)
    return threshold

def make_tensor_loader(X, y=None, shuffle=False, generator=None, batch_size=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    if y is None:
        y_tensor = torch.zeros(len(X_tensor), dtype=torch.long)
    else:
        y_tensor = torch.as_tensor(y, dtype=torch.long)

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        shuffle=bool(shuffle),
        generator=generator,
        pin_memory=config.PIN_MEMORY,
        batch_size=int(batch_size or config.BATCH_SIZE),
        num_workers=config.NUM_WORKERS,
    )

def compute_channel_standardization_stats(X, prefer_gpu=False, eps=1e-12):
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_samples, timesteps, channels), got {X.shape}")

    if prefer_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                xt = torch.as_tensor(X, dtype=torch.float32, device="cuda")
                flat = xt.reshape(-1, xt.shape[2])
                mean_t = flat.mean(dim=0)
                var_t = flat.var(dim=0, unbiased=False)
                std_t = torch.sqrt(torch.clamp(var_t, min=0.0))
                std_t = torch.where(std_t > float(eps), std_t, torch.ones_like(std_t))
                return (
                    mean_t.detach().cpu().numpy().astype(np.float32, copy=False),
                    std_t.detach().cpu().numpy().astype(np.float32, copy=False),
                )
        except Exception:
            pass

    flat_np = X.reshape(-1, X.shape[2]).astype(np.float64, copy=False)
    mean = flat_np.mean(axis=0)
    std = flat_np.std(axis=0, ddof=0)
    std = np.where(std > float(eps), std, 1.0)
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)

def save_channel_standardization_stats(path, mean, std):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )

def load_channel_standardization_stats(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "mean" not in data or "std" not in data:
            return None
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
    return mean, std

def make_torch_standardizer(mean, std, device, eps=1e-12):
    import torch

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 1, -1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 1, -1)
    inv_std_t = torch.reciprocal(std_t.clamp_min(float(eps)))
    return mean_t, inv_std_t

def standardize_batch_torch(xb, mean_t, inv_std_t):
    if mean_t is None or inv_std_t is None:
        return xb
    return (xb - mean_t) * inv_std_t

def _ablation_channel_indices(channels, ablation):
    if channels % 8 != 0:
        raise ValueError(f"Expected channels multiple of 8 for ablation, got {channels}")
    n_sensors = channels // 8
    if not ablation:
        return list(range(channels))
    if ablation == "acc":
        return [s * 8 + offset for s in range(n_sensors) for offset in (1, 2, 3)]
    if ablation == "gyr":
        return [s * 8 + offset for s in range(n_sensors) for offset in (5, 6, 7)]
    if ablation == "acc_gyr":
        return [s * 8 + offset for s in range(n_sensors) for offset in (1, 2, 3, 5, 6, 7)]
    if ablation == "magacc_maggyr":
        return [s * 8 + offset for s in range(n_sensors) for offset in (0, 4)]
    if ablation == "acc_magacc":
        return [s * 8 + offset for s in range(n_sensors) for offset in (0, 1, 2, 3)]
    if ablation == "gyr_maggyr":
        return [s * 8 + offset for s in range(n_sensors) for offset in (4, 5, 6, 7)]
    if ablation == "magacc":
        return [s * 8 + 0 for s in range(n_sensors)]
    if ablation == "maggyr":
        return [s * 8 + 4 for s in range(n_sensors)]
    if ablation == "acc_gyr_magacc_maggyr":
        return list(range(channels))
    raise ValueError(f"Unknown ablation mode: {ablation}")


def resolve_available_sensor_runs(args, dataset_name=None, min_sensors=2, require_model_dir=False):
    dataset_name = dataset_name or getattr(args, "train_data", None) or getattr(args, "training_data", None) or ""
    candidates = list(sensors_from_experiment(dataset_name))
    if len(candidates) < 2:
        raise ValueError(
            "For bagging/stacking, dataset must include at least 2 sensors "
            "(e.g., chest_left, chest_right, left_right, chest_left_right)."
        )

    available = []
    trained = {}
    missing = []

    for sensor_name in candidates:
        sensor_run_name = run_name_for_dataset(sensor_name, args)
        out_dir = os.path.join(OUTPUT_ROOT, args.model, sensor_run_name)
        mdl_dir = os.path.join(MODELS_ROOT, args.model, sensor_run_name)
        has_output = os.path.exists(out_dir)
        has_model = os.path.exists(mdl_dir)
        if has_output and (has_model or not require_model_dir):
            available.append(sensor_name)
            trained[sensor_name] = {"output": out_dir, "model": mdl_dir}
        else:
            missing.append(sensor_name)

    if len(available) < int(min_sensors):
        raise FileNotFoundError(
            "Sensor ensemble requires at least 2 trained base sensors among "
            f"{candidates}. "
            f"Available: {available}. Missing: {missing}. "
            "Run individual_generalization first for the missing sensors/dataset/config."
        )

    return candidates, available, trained

def _normalize_prediction_df(df):
    out = df[["group_id", "window_id", "y_true", "y_prob_1"]].copy()
    out["window_id"] = out["window_id"].astype(str)
    return out

def _load_all_fold_predictions(run_output_dir):
    import pandas as pd

    rows = []
    if not os.path.isdir(run_output_dir):
        return pd.DataFrame(columns=["group_id", "window_id", "y_true", "y_prob_1"])

    for name in sorted(os.listdir(run_output_dir)):
        if not name.startswith("fold_"):
            continue
        pred_path = os.path.join(run_output_dir, name, "predictions.csv")
        if not os.path.exists(pred_path):
            continue
        rows.append(_normalize_prediction_df(pd.read_csv(pred_path)))

    if not rows:
        return pd.DataFrame(columns=["group_id", "window_id", "y_true", "y_prob_1"])

    return pd.concat(rows, ignore_index=True)

def _augment_meta_feature_dropout(X, y, p=0.5, max_off=1, copies=1, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D meta-features, got shape {X.shape}")
    if len(X) == 0 or X.shape[1] == 0 or copies <= 0 or p <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    X_parts = [X]
    y_parts = [y]
    max_off = min(max(int(max_off), 1), X.shape[1])

    for _ in range(int(copies)):
        X_copy = X.copy()
        for i in range(len(X_copy)):
            if rng.random() >= float(p):
                continue
            n_drop = int(rng.integers(1, max_off + 1))
            dropped = rng.choice(X.shape[1], size=n_drop, replace=False)
            X_copy[i, dropped] = 0.0
        X_parts.append(X_copy)
        y_parts.append(y)

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

def logits_to_binary_probs(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return np.empty((0, 2), dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.shape[1] == 1:
        prob_pos = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return np.column_stack([1.0 - prob_pos, prob_pos])
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_shift = np.exp(shift)
    return exp_shift / np.clip(np.sum(exp_shift, axis=1, keepdims=True), 1e-12, None)


def tune_threshold_f1(y_true, y_prob_1):
    y_true = np.asarray(y_true, dtype=int)
    y_prob_1 = np.asarray(y_prob_1, dtype=float)
    finite = np.isfinite(y_prob_1)
    y_true = y_true[finite]
    y_prob_1 = y_prob_1[finite]
    if y_true.size == 0:
        return 0.5, 0.0

    order = np.argsort(-y_prob_1, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob_1[order]

    tp_cum = np.cumsum(y_sorted == 1, dtype=np.int64)
    fp_cum = np.cumsum(y_sorted == 0, dtype=np.int64)
    n_pos = int(tp_cum[-1])

    # Evaluate only at unique score cutoffs where predictions change.
    cutoff = np.empty(len(p_sorted), dtype=bool)
    cutoff[:-1] = p_sorted[:-1] != p_sorted[1:]
    cutoff[-1] = True

    tp = tp_cum[cutoff].astype(float)
    fp = fp_cum[cutoff].astype(float)
    fn = float(n_pos) - tp
    denom = (2.0 * tp) + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0.0)
    thresholds = np.clip(p_sorted[cutoff].astype(float), 0.0, 1.0)

    best_score = float(np.max(f1)) if f1.size > 0 else 0.0
    best_idx = np.flatnonzero(np.abs(f1 - best_score) <= 1e-12)
    if best_idx.size == 0:
        return 0.5, best_score

    tie_thresholds = thresholds[best_idx]
    chosen_local = int(np.argmin(np.abs(tie_thresholds - 0.5)))
    best_thr = float(tie_thresholds[chosen_local])
    return best_thr, best_score


def _resolve_dataset_root(experiment):
    candidates = [
        os.path.join(DATA_PATH, experiment),
        experiment,
    ]
    for root in candidates:
        data_file = os.path.join(root, "data", "data_time_domain.npy")
        label_file = os.path.join(root, "labels", "labels.npy")
        if os.path.isfile(data_file) and os.path.isfile(label_file):
            return root
    raise FileNotFoundError(
        f"Could not find dataset '{experiment}'. Expected a folder with data/data_time_domain.npy and labels/labels.npy under {DATA_PATH}."
    )


def load_bundle(experiment, args):
    dataset_root = _resolve_dataset_root(experiment)
    X = np.load(os.path.join(dataset_root, "data", "data_time_domain.npy"))
    y = np.load(os.path.join(dataset_root, "labels", "labels.npy")).astype(np.int64)
    groups = np.load(os.path.join(dataset_root, "labels", "groups.npy"))
    window_ids_path = os.path.join(dataset_root, "labels", "window_ids.npy")
    window_ids = np.load(window_ids_path, allow_pickle=True) if os.path.exists(window_ids_path) else None
    if args.ablation:
        channels = X.shape[2]
        keep_cols = _ablation_channel_indices(channels, args.ablation)
        X = X[:, :, keep_cols]
    return {
        "experiment": experiment,
        "X": X, "y": y,
        "groups": groups, "window_ids": window_ids,
    }


def expand_to_train_layout(X, source_sensors, target_sensors):
    out = np.full((X.shape[0], X.shape[1], len(target_sensors) * 8), 0.0, dtype=X.dtype)
    source_map = {name: idx for idx, name in enumerate(source_sensors)}
    target_map = {name: idx for idx, name in enumerate(target_sensors)}
    for sensor in source_sensors:
        s0 = source_map[sensor] * 8
        t0 = target_map[sensor] * 8
        out[:, :, t0 : t0 + 8] = X[:, :, s0 : s0 + 8]
    return out


def augment_sensor_dropout(X, y, sensors, block_size=8, p=0.5, max_off=1, copies=1, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    sensors = list(sensors)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_samples, timesteps, channels), got {X.shape}")
    if not sensors or copies <= 0 or p <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    X_parts = [X]
    y_parts = [y]
    max_off = min(max(int(max_off), 1), len(sensors))
    sensor_offsets = [idx * block_size for idx in range(len(sensors))]

    for _ in range(int(copies)):
        X_copy = X.copy()
        for i in range(len(X_copy)):
            if rng.random() >= float(p):
                continue
            n_drop = int(rng.integers(1, max_off + 1))
            dropped = rng.choice(len(sensors), size=n_drop, replace=False)
            for sensor_idx in np.asarray(dropped, dtype=int):
                start = sensor_offsets[sensor_idx]
                X_copy[i, :, start : start + block_size] = 0.0
        X_parts.append(X_copy)
        y_parts.append(y)

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def apply_sensor_dropout_torch(xb, n_sensors, block_size=8, p=0.5, max_off=1):
    import torch

    if p <= 0 or max_off <= 0 or n_sensors <= 0:
        return xb

    B = xb.shape[0]
    n_sensors = int(n_sensors)
    max_off = min(max(int(max_off), 1), n_sensors)
    out = xb.clone()
    sensor_channels = n_sensors * int(block_size)
    if out.shape[2] < sensor_channels:
        return out

    apply_mask = torch.rand(B, device=xb.device) < float(p)
    active_rows = torch.nonzero(apply_mask, as_tuple=False).flatten()

    if active_rows.numel() == 0:
        return out

    out_blocks = out[:, :, :sensor_channels].view(B, out.shape[1], n_sensors, int(block_size))
    active_blocks = out_blocks[active_rows]
    n_active = active_blocks.shape[0]

    n_drop = torch.randint(1, max_off + 1, (n_active,), device=xb.device)
    scores = torch.rand(n_active, n_sensors, device=xb.device)
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(sorted_idx)
    sensor_order = torch.arange(n_sensors, device=xb.device).view(1, -1).expand(n_active, -1)
    ranks.scatter_(1, sorted_idx, sensor_order)
    drop_mask = ranks < n_drop.view(-1, 1)

    active_blocks.masked_fill_(drop_mask[:, None, :, None], 0.0)
    out_blocks[active_rows] = active_blocks

    return out


def save_loss_curve_plot(train_losses, out_path, val_losses=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_losses = np.asarray(train_losses if train_losses is not None else [], dtype=float)
    val_losses = np.asarray(val_losses if val_losses is not None else [], dtype=float)
    if train_losses.size == 0 and val_losses.size == 0:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    if train_losses.size > 0:
        epochs_train = np.arange(1, train_losses.size + 1)
        plt.plot(epochs_train, train_losses, label="train_loss", linewidth=2)

    if val_losses.size > 0:
        epochs_val = np.arange(1, val_losses.size + 1)
        plt.plot(epochs_val, val_losses, label="val_loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_roc_pr_curves(y_true, y_prob_1, fold_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    y_true = np.asarray(y_true, dtype=int)
    y_prob_1 = np.asarray(y_prob_1, dtype=float)

    roc_csv = os.path.join(fold_dir, "roc_curve.csv")
    roc_png = os.path.join(fold_dir, "roc_curve.png")
    pr_csv = os.path.join(fold_dir, "pr_curve.csv")
    pr_png = os.path.join(fold_dir, "pr_curve.png")

    if y_true.size == 0:
        pd.DataFrame(columns=["fpr", "tpr", "threshold"]).to_csv(roc_csv, index=False)
        pd.DataFrame(columns=["recall", "precision", "threshold"]).to_csv(pr_csv, index=False)
        return

    uniq = np.unique(y_true)

    if uniq.size >= 2:
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob_1)
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob_1))
        except Exception:
            roc_auc = float("nan")

        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(roc_csv, index=False)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.4f}" if np.isfinite(roc_auc) else "AUC=nan")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_png, dpi=180)
        plt.close()
    else:
        pd.DataFrame(columns=["fpr", "tpr", "threshold"]).to_csv(roc_csv, index=False)

    precision, recall, pr_thr = precision_recall_curve(y_true, y_prob_1)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob_1))
    except Exception:
        pr_auc = float("nan")

    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    pr_df["threshold"] = np.nan
    if pr_thr.size > 0:
        pr_df.loc[1:, "threshold"] = pr_thr
    pr_df.to_csv(pr_csv, index=False)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AP={pr_auc:.4f}" if np.isfinite(pr_auc) else "AP=nan")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_png, dpi=180)
    plt.close()


def score_and_save_fold_outputs(y_test, test_probs, threshold, fold_dir, test_bundle,
                                test_idx, experiment_for_outputs, sensor_status, save_arrays=False):
    import pandas as pd
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    os.makedirs(fold_dir, exist_ok=True)

    good = ~np.isnan(test_probs).any(axis=1)
    y_true = np.asarray(y_test, dtype=int)[good]
    y_probs = np.asarray(test_probs, dtype=float)[good]
    y_pred = (y_probs[:, 1] >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_probs[:, 1])
    except Exception:
        pr_auc = float("nan")

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, "done.csv"), index=False)

    data = {"y_true": y_true, "y_prob_0": y_probs[:, 0], "y_prob_1": y_probs[:, 1], "y_pred": y_pred}
    if test_bundle.get("groups") is not None:
        data["group_id"] = np.asarray(test_bundle["groups"][test_idx])[good]
    if test_bundle.get("window_ids") is not None:
        data["window_id"] = np.asarray(test_bundle["window_ids"][test_idx], dtype=object)[good]
    data["experiment"] = [experiment_for_outputs] * len(y_true)
    data["missing_sensors"] = [",".join(sensor_status.get("missing", []))] * len(y_true)
    data["available_sensors"] = [",".join(sensor_status.get("available", []))] * len(y_true)
    pd.DataFrame(data).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
    _save_roc_pr_curves(y_true=y_true, y_prob_1=y_probs[:, 1], fold_dir=fold_dir)

    if save_arrays:
        np.save(os.path.join(fold_dir, "y_true.npy"), y_true)
        np.save(os.path.join(fold_dir, "y_probs.npy"), y_probs)
        np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred)

    return metrics

def _log_header(title, **kwargs):
    print("=" * 90)
    print(f"[RUN] {title}")
    for key, value in kwargs.items():
        print(f"  - {key}: {value}")
    print("=" * 90)

def _log_fold(prefix, idx, total=None, **kwargs):
    head = f"[{prefix}] fold {idx + 1}/{total}" if total is not None else f"[{prefix}] fold {idx + 1}"
    tail = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"{head}{(' | ' + tail) if tail else ''}")