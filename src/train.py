import os
import glob

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import src.config as config
import src.utils as utils


def _predict_meta_prob_1(meta_model, X):
    X = np.asarray(X, dtype=np.float32)
    if hasattr(meta_model, "predict_proba") or hasattr(meta_model, "decision_function"):
        return utils.estimator_binary_prob_1(meta_model, X)

    import torch

    meta_model.eval()
    with torch.no_grad():
        logits = meta_model(torch.tensor(X, dtype=torch.float32)).squeeze(1)
        prob_1 = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(prob_1, dtype=float)


def _tune_meta_threshold_with_group_oof(X, y, groups):
    import src.models as models

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    if X.ndim != 2 or len(X) == 0:
        return float(config.DECISION_THRESHOLD), float("nan"), "default"

    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        logo = LeaveOneGroupOut()
        oof_prob_1 = np.full(len(y), np.nan, dtype=float)

        for fit_idx, val_idx in logo.split(X, y, groups):
            y_fit = y[fit_idx]
            if np.unique(y_fit).size < 2:
                continue

            inner_meta_model = models.make_classical_model("LogisticRegression", y_fit)
            inner_meta_model.fit(X[fit_idx], y_fit)
            oof_prob_1[val_idx] = _predict_meta_prob_1(inner_meta_model, X[val_idx])

        valid = np.isfinite(oof_prob_1)
        if np.any(valid) and np.unique(y[valid]).size >= 2:
            best_thr, best_score = utils.tune_threshold_f1(y[valid], oof_prob_1[valid])
            return float(best_thr), float(best_score), "group_oof_logo"

    return float(config.DECISION_THRESHOLD), float("nan"), "default"


def _save_meta_validation_curve(X, y, groups, out_dir, seed):
    import torch
    import torch.nn as nn

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    os.makedirs(out_dir, exist_ok=True)

    unique_groups = np.unique(groups)
    if X.ndim != 2 or len(X) == 0 or np.unique(y).size < 2 or unique_groups.size < 2:
        with open(os.path.join(out_dir, "meta_validation_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": "skipped",
                    "reason": "insufficient_samples_or_classes_or_groups",
                    "n_samples": int(len(X)),
                    "n_groups": int(unique_groups.size),
                    "n_classes": int(np.unique(y).size),
                },
                fh,
                indent=2,
            )
        return

    n_splits = min(int(config.INNER_FOLDS), int(unique_groups.size))
    if n_splits < 2:
        with open(os.path.join(out_dir, "meta_validation_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": "skipped",
                    "reason": "n_splits_less_than_2",
                    "n_splits": int(n_splits),
                },
                fh,
                indent=2,
            )
        return

    epochs = max(1, int(getattr(config, "STACKING_META_DIAG_EPOCHS", 40)))
    lr = float(getattr(config, "STACKING_META_DIAG_LR", 1e-2))

    train_sum = np.zeros(epochs, dtype=float)
    val_sum = np.zeros(epochs, dtype=float)
    used_splits = 0

    cv = GroupKFold(n_splits=n_splits)
    for split_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        y_tr = y[tr_idx]
        if np.unique(y_tr).size < 2:
            continue

        split_seed = int(seed + split_idx)
        torch.manual_seed(split_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(split_seed)

        x_tr = torch.as_tensor(X[tr_idx], dtype=torch.float32, device=config.DEVICE)
        y_tr_t = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=config.DEVICE).unsqueeze(1)
        x_val = torch.as_tensor(X[val_idx], dtype=torch.float32, device=config.DEVICE)
        y_val_t = torch.as_tensor(y[val_idx], dtype=torch.float32, device=config.DEVICE).unsqueeze(1)

        pos_count = max(int((y[tr_idx] == 1).sum()), 1)
        neg_count = max(int((y[tr_idx] == 0).sum()), 1)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=config.DEVICE)

        model = nn.Linear(x_tr.shape[1], 1).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")):
                logits_tr = model(x_tr)
                loss_tr = criterion(logits_tr, y_tr_t)
            loss_tr.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")):
                    tr_eval = criterion(model(x_tr), y_tr_t)
                    val_eval = criterion(model(x_val), y_val_t)
            train_sum[epoch] += float(tr_eval.detach().item())
            val_sum[epoch] += float(val_eval.detach().item())

        used_splits += 1

    if used_splits == 0:
        with open(os.path.join(out_dir, "meta_validation_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": "skipped",
                    "reason": "no_valid_inner_splits",
                },
                fh,
                indent=2,
            )
        return

    train_curve = train_sum / float(used_splits)
    val_curve = val_sum / float(used_splits)

    pd.DataFrame(
        {
            "epoch": np.arange(1, epochs + 1),
            "train_loss": train_curve,
            "val_loss": val_curve,
        }
    ).to_csv(os.path.join(out_dir, "meta_validation_loss_curve.csv"), index=False)

    utils.save_loss_curve_plot(
        train_losses=train_curve.tolist(),
        val_losses=val_curve.tolist(),
        out_path=os.path.join(out_dir, "meta_validation_loss_curve.png"),
    )

    with open(os.path.join(out_dir, "meta_validation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "status": "ok",
                "n_splits": int(n_splits),
                "used_splits": int(used_splits),
                "epochs": int(epochs),
                "learning_rate": float(lr),
            },
            fh,
            indent=2,
        )

def train_neural_model(
    args,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=None,
    normalizer=None,
    drop_sensors_override=None,
    seed=None,
    batch_size_override=None,
    preload_to_device=False,
    monitor_X=None,
    monitor_y=None,
):
    import src.models as models

    import torch
    import torch.nn as nn

    input_shape = utils.model_input_shape(args.model, X_train)

    model = models.create_model(args.model, input_shape, 1)
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=config.PATIENCE, min_lr=1e-6)
    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_seed = int(config.SEED if seed is None else seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    generator = torch.Generator()
    generator.manual_seed(train_seed)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(config.DEVICE.type == "cuda"))

    norm_mean_t, norm_inv_std_t = None, None
    if normalizer is not None:
        norm_mean, norm_std = normalizer
        norm_mean_t, norm_inv_std_t = utils.make_torch_standardizer(norm_mean, norm_std, config.DEVICE)

    effective_batch_size = int(batch_size_override or config.BATCH_SIZE)
    use_preloaded_path = bool(preload_to_device and config.DEVICE.type == "cuda" and X_val is None and y_val is None)

    train_loader = None
    val_loader = None
    monitor_loader = None
    if not use_preloaded_path:
        train_loader = utils.make_tensor_loader(
            X_train,
            y=y_train,
            shuffle=True,
            generator=generator,
            batch_size=effective_batch_size,
        )
        if X_val is not None and y_val is not None:
            val_loader = utils.make_tensor_loader(
                X_val,
                y=y_val,
                shuffle=False,
                batch_size=effective_batch_size,
            )

    if monitor_X is not None and monitor_y is not None:
        monitor_loader = utils.make_tensor_loader(
            monitor_X,
            y=monitor_y,
            shuffle=False,
            batch_size=effective_batch_size,
        )

    def _mean_loader_loss(loader):
        if loader is None:
            return float("nan")
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(config.DEVICE, non_blocking=True)
                yb = yb.to(config.DEVICE, non_blocking=True)
                xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)
                with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")):
                    out = model(xb)
                    target = yb.float().unsqueeze(1)
                    loss = criterion(out, target)
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses)) if losses else float("nan")

    train_losses, val_losses, monitor_losses = [], [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    if drop_sensors_override is not None:
        drop_sensors = list(drop_sensors_override)
    else:
        drop_sensors = utils.sensors_from_experiment(args.train_data)
    epochs_to_run = int(epochs or config.EPOCHS)

    if use_preloaded_path:
        x_train_gpu = torch.as_tensor(X_train, dtype=torch.float32, device=config.DEVICE)
        y_train_gpu = torch.as_tensor(y_train, dtype=torch.float32, device=config.DEVICE).unsqueeze(1)
        n_samples = int(x_train_gpu.shape[0])

        for _ in range(epochs_to_run):
            model.train()
            epoch_train = []

            perm = torch.randperm(n_samples, generator=generator)
            for start in range(0, n_samples, effective_batch_size):
                batch_idx = perm[start : start + effective_batch_size].to(config.DEVICE, non_blocking=True)
                xb = x_train_gpu.index_select(0, batch_idx)
                target = y_train_gpu.index_select(0, batch_idx)

                xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)
                if args.sensor_dropout:
                    xb = utils.apply_sensor_dropout_torch(
                        xb,
                        n_sensors=len(drop_sensors),
                        block_size=8,
                        p=config.SENSOR_DROPOUT_P,
                        max_off=config.SENSOR_DROPOUT_MAX_OFF,
                    )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    out = model(xb)
                    loss = criterion(out, target)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_train.append(float(loss.detach().item()))

            train_losses.append(float(np.mean(epoch_train)) if epoch_train else float("nan"))
            if monitor_loader is not None:
                monitor_losses.append(_mean_loader_loss(monitor_loader))

        epochs_ran = len(train_losses)
        return {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "monitor_losses": monitor_losses,
            "best_val_loss": float("nan"),
            "epochs_ran": epochs_ran,
        }

    for _ in range(epochs_to_run):
        model.train()
        epoch_train = []

        for xb, yb in train_loader:
            xb = xb.to(config.DEVICE, non_blocking=True)
            yb = yb.to(config.DEVICE, non_blocking=True)

            xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)

            if args.sensor_dropout:
                xb = utils.apply_sensor_dropout_torch(xb, n_sensors=len(drop_sensors), block_size=8, p=config.SENSOR_DROPOUT_P, max_off=config.SENSOR_DROPOUT_MAX_OFF)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")):
                out = model(xb); target = yb.float().unsqueeze(1); loss = criterion(out, target)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_train.append(float(loss.detach().item()))

        train_losses.append(float(np.mean(epoch_train)) if epoch_train else float("nan"))

        if monitor_loader is not None:
            monitor_losses.append(_mean_loader_loss(monitor_loader))

        if val_loader is None:
            continue

        model.eval()
        epoch_val = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(config.DEVICE, non_blocking=True)
                yb = yb.to(config.DEVICE, non_blocking=True)
                xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)
                with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")):
                    out = model(xb); target = yb.float().unsqueeze(1); loss = criterion(out, target)
                epoch_val.append(float(loss.detach().item()))

        avg_val_loss = float(np.mean(epoch_val)) if epoch_val else float("nan")
        val_losses.append(avg_val_loss)

        if np.isfinite(avg_val_loss):
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

    epochs_ran = len(train_losses)

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "monitor_losses": monitor_losses,
        "best_val_loss": best_val_loss if val_loader is not None else float("nan"),
        "epochs_ran": epochs_ran,
    }


def fit_and_eval_fold(args, train_bundle, train_idx, test_idx, fold_idx, fold_dir, fold_model_dir, 
                      experiment_for_outputs, sensor_status, test_bundle=None):
    import src.models as models

    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(fold_model_dir, exist_ok=True)
    metrics_path = os.path.join(fold_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        if args.model in models.CLASSICAL_MODELS:
            return pd.read_csv(metrics_path).iloc[0].to_dict()

        inner_curve_csvs = glob.glob(os.path.join(fold_dir, "inner_loss_curves", "inner_fold_*_loss_curve.csv"))
        inner_curve_pngs = glob.glob(os.path.join(fold_dir, "inner_loss_curves", "inner_fold_*_loss_curve.png"))
        if inner_curve_csvs and inner_curve_pngs:
            return pd.read_csv(metrics_path).iloc[0].to_dict()

        print(f"[REBUILD] Missing inner CV loss artifacts in {fold_dir}; recomputing fold outputs.")

    test_bundle = test_bundle or train_bundle
    X_train_all = train_bundle["X"][train_idx]
    y_train_all = train_bundle["y"][train_idx]
    groups_train = train_bundle["groups"][train_idx]
    X_test = test_bundle["X"][test_idx]
    y_test = test_bundle["y"][test_idx]

    threshold = config.DECISION_THRESHOLD
    fold_label = os.path.basename(fold_dir).replace("fold_", "")
    final_epochs = config.EPOCHS
    threshold_score = float("nan")
    inner_loss_rows = []

    unique_groups = np.unique(groups_train)
    n_splits = min(int(config.INNER_FOLDS), len(unique_groups))
    if n_splits < 2: raise ValueError("Inner GroupCV requires at least 2 groups in the outer-train set.")
    inner_cv = GroupKFold(n_splits=n_splits)
    utils._log_fold("INNER-CV", fold_idx, None, n_splits=n_splits, train_samples=len(train_idx), test_samples=len(test_idx))
    oof_prob_1 = np.full(len(y_train_all), np.nan, dtype=float)
    inner_epoch_counts = []

    for inner_fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_all, y_train_all, groups=groups_train)):
        utils._log_fold(
            "INNER",
            inner_fold_idx,
            n_splits,
            train=len(inner_tr_idx),
            val=len(inner_val_idx),
        )
        X_train = X_train_all[inner_tr_idx]
        y_train = y_train_all[inner_tr_idx]
        X_val = X_train_all[inner_val_idx]
        y_val = y_train_all[inner_val_idx]

        if args.model in models.CLASSICAL_MODELS:
            if args.sensor_dropout:
                drop_sensors = utils.sensors_from_experiment(args.train_data)
                X_train_fit_src, y_train_fit = utils.augment_sensor_dropout(
                    X_train,
                    y_train,
                    sensors=drop_sensors,
                    block_size=8,
                    p=config.SENSOR_DROPOUT_P,
                    max_off=config.SENSOR_DROPOUT_MAX_OFF,
                    copies=1,
                    seed=config.SEED + int(int(fold_idx) * 100 + int(inner_fold_idx)),
                )
            else:
                X_train_fit_src, y_train_fit = X_train, y_train

            X_train_fit = X_train_fit_src.reshape(len(X_train_fit_src), -1)
            X_val_fit = X_val.reshape(len(X_val), -1)
            inner_model = models.make_classical_model(args.model, y_train_fit)
            inner_model.fit(X_train_fit, y_train_fit)

            prob_1 = utils.estimator_binary_prob_1(inner_model, X_val_fit)
            val_probs = np.column_stack([1.0 - prob_1, prob_1])
            oof_prob_1[inner_val_idx] = val_probs[:, 1]
        else:
            import torch

            train_result = train_neural_model(args=args, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            inner_curve_dir = os.path.join(fold_dir, "inner_loss_curves")
            os.makedirs(inner_curve_dir, exist_ok=True)
            inner_curve_csv = os.path.join(inner_curve_dir, f"inner_fold_{int(inner_fold_idx)}_loss_curve.csv")
            inner_curve_png = os.path.join(inner_curve_dir, f"inner_fold_{int(inner_fold_idx)}_loss_curve.png")

            train_hist = list(train_result["train_losses"])
            val_hist = list(train_result["val_losses"])
            max_len = max(len(train_hist), len(val_hist))
            if max_len > 0:
                train_pad = train_hist + [np.nan] * (max_len - len(train_hist))
                val_pad = val_hist + [np.nan] * (max_len - len(val_hist))
                pd.DataFrame(
                    {
                        "epoch": np.arange(1, max_len + 1),
                        "train_loss": train_pad,
                        "val_loss": val_pad,
                    }
                ).to_csv(inner_curve_csv, index=False)
                utils.save_loss_curve_plot(
                    train_losses=train_hist,
                    val_losses=val_hist,
                    out_path=inner_curve_png,
                )

            val_loader = utils.make_tensor_loader(X_val, y=None, shuffle=False)
            train_result["model"].eval()
            val_logits = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(config.DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")): out = train_result["model"](xb)
                    val_logits.append(out.detach().cpu().numpy())
            val_logits = np.concatenate(val_logits, axis=0) if val_logits else np.empty((0, 1), dtype=float)
            val_probs = utils.logits_to_binary_probs(val_logits)
            oof_prob_1[inner_val_idx] = val_probs[:, 1]
            inner_epoch_counts.append(int(train_result["epochs_ran"]))
            inner_loss_rows.append({"inner_fold": int(inner_fold_idx), "best_val_loss": float(train_result["best_val_loss"]), "epochs_ran": int(train_result["epochs_ran"] )})

    valid = np.isfinite(oof_prob_1)
    if not np.any(valid): raise ValueError("No valid OOF probabilities were produced for threshold tuning.")

    y_true_thr = np.asarray(y_train_all[valid]).astype(int)
    y_prob_thr = np.asarray(oof_prob_1[valid], dtype=float)
    best_threshold, best_score = utils.tune_threshold_f1(y_true_thr, y_prob_thr)

    threshold = best_threshold
    threshold_score = best_score

    pd.DataFrame({"y_true": y_train_all, "y_prob_1": oof_prob_1, "group_id": groups_train, "window_id": train_bundle["window_ids"][train_idx]}).to_csv(os.path.join(fold_dir, "inner_oof_predictions.csv"), index=False)

    if inner_loss_rows:
        pd.DataFrame(inner_loss_rows).to_csv(os.path.join(fold_dir, "inner_cv_summary.csv"), index=False)
        final_epochs = max(1, int(round(float(np.mean(inner_epoch_counts)))))

    X_fit = X_train_all
    y_fit = y_train_all

    if args.model in models.CLASSICAL_MODELS:
        _, _, n_channels = X_fit.shape
        final_scaler = StandardScaler()
        final_scaler.fit(X_fit.reshape(-1, n_channels))
        X_fit_scaled = final_scaler.transform(X_fit.reshape(-1, n_channels)).reshape(X_fit.shape)
        X_test_scaled = final_scaler.transform(X_test.reshape(-1, n_channels)).reshape(X_test.shape)
        joblib.dump(final_scaler, os.path.join(fold_model_dir, "scaler.joblib"))

        if args.sensor_dropout:
            drop_sensors = utils.sensors_from_experiment(args.train_data)
            X_fit_src, y_fit_aug = utils.augment_sensor_dropout(
                X_fit_scaled,
                y_fit,
                sensors=drop_sensors,
                block_size=8,
                p=config.SENSOR_DROPOUT_P,
                max_off=config.SENSOR_DROPOUT_MAX_OFF,
                copies=1,
                seed=config.SEED + int(4242 + int(fold_idx)),
            )
        else:
            X_fit_src, y_fit_aug = X_fit_scaled, y_fit

        X_fit_flat = X_fit_src.reshape(len(X_fit_src), -1)
        X_test_flat = X_test_scaled.reshape(len(X_test_scaled), -1)
        final_model = models.make_classical_model(args.model, y_fit_aug)
        final_model.fit(X_fit_flat, y_fit_aug)
        prob_1 = utils.estimator_binary_prob_1(final_model, X_test_flat)
        test_probs = np.column_stack([1.0 - prob_1, prob_1])
        joblib.dump(final_model, os.path.join(fold_model_dir, f"{fold_label}.joblib"))
    else:
        import torch

        norm_mean, norm_std = utils.compute_channel_standardization_stats(
            X_fit,
            prefer_gpu=(config.GPU_NORMALIZATION and config.DEVICE.type == "cuda"),
        )
        utils.save_channel_standardization_stats(
            os.path.join(fold_model_dir, "scaler_stats.npz"),
            norm_mean,
            norm_std,
        )

        final_result = train_neural_model(
            args=args,
            X_train=X_fit,
            y_train=y_fit,
            X_val=None,
            y_val=None,
            epochs=final_epochs,
            normalizer=(norm_mean, norm_std),
        )
        final_model = final_result["model"]
        test_loader = utils.make_tensor_loader(X_test, y=None, shuffle=False)
        norm_mean_t, norm_inv_std_t = utils.make_torch_standardizer(norm_mean, norm_std, config.DEVICE)
        final_model.eval()
        test_logits = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(config.DEVICE, non_blocking=True)
                xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)
                with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")): out = final_model(xb)
                test_logits.append(out.detach().cpu().numpy())
        test_logits = np.concatenate(test_logits, axis=0) if test_logits else np.empty((0, 1), dtype=float)
        test_probs = utils.logits_to_binary_probs(test_logits)

        pd.DataFrame({"epoch": np.arange(1, len(final_result["train_losses"]) + 1), "train_loss": final_result["train_losses"]}).to_csv(os.path.join(fold_dir, "final_fit_loss_curve.csv"), index=False)
        utils.save_loss_curve_plot(train_losses=final_result["train_losses"], out_path=os.path.join(fold_dir, "final_fit_loss_curve.png"))

        torch.save(final_model.state_dict(), os.path.join(fold_model_dir, f"{fold_label}.pt"))

    metrics = utils.score_and_save_fold_outputs(y_test=y_test, test_probs=test_probs, threshold=threshold, fold_dir=fold_dir, test_bundle=test_bundle, test_idx=test_idx, experiment_for_outputs=experiment_for_outputs, sensor_status=sensor_status, save_arrays=False)

    metrics["threshold_tuning_score"] = float(threshold_score)
    metrics["inner_cv_folds"] = int(min(int(config.INNER_FOLDS), len(np.unique(groups_train))))
    if args.model not in models.CLASSICAL_MODELS:
        metrics["final_fit_epochs"] = int(final_epochs)
    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, "done.csv"), index=False)

    return metrics


def _bundle_key_strings(bundle, indices=None):
    window_ids = bundle.get("window_ids")
    if window_ids is None:
        raise ValueError("window_ids are required for stacking alignment.")

    groups = bundle["groups"] if indices is None else bundle["groups"][indices]
    windows = window_ids if indices is None else window_ids[indices]
    return np.asarray([f"{str(g)}::{str(w)}" for g, w in zip(groups, windows)], dtype=object)


def _build_bundle_key_index(bundle, context="bundle"):
    keys = _bundle_key_strings(bundle)
    key_to_idx = {}
    for idx, key in enumerate(keys.tolist()):
        if key in key_to_idx:
            raise ValueError(f"Duplicate (group_id, window_id) key in {context}: {key}")
        key_to_idx[key] = int(idx)
    return key_to_idx


def _prediction_frame_from_bundle(bundle, indices, prob_1, sensor_name):
    idx = np.asarray(indices, dtype=int)
    prob_1 = np.asarray(prob_1, dtype=float)
    if len(idx) != len(prob_1):
        raise ValueError(
            f"Prediction size mismatch for {sensor_name}: indices={len(idx)}, probs={len(prob_1)}"
        )

    return pd.DataFrame(
        {
            "group_id": bundle["groups"][idx],
            "window_id": np.asarray(bundle["window_ids"][idx], dtype=object).astype(str),
            "y_true": np.asarray(bundle["y"][idx], dtype=int),
            f"prob_{sensor_name}": prob_1,
        }
    )


def _merge_sensor_probability_frames(sensor_frames, sensor_order, context):
    present = [sensor for sensor in sensor_order if sensor in sensor_frames]
    if len(present) < 2:
        return pd.DataFrame()

    first = present[0]
    merged = sensor_frames[first].rename(columns={"y_true": f"y_true_{first}"})
    for sensor in present[1:]:
        merged = merged.merge(
            sensor_frames[sensor].rename(columns={"y_true": f"y_true_{sensor}"}),
            on=["group_id", "window_id"],
            how="inner",
        )

    if merged.empty:
        return merged

    y_cols = [f"y_true_{sensor}" for sensor in present if f"y_true_{sensor}" in merged.columns]
    ref_col = y_cols[0]
    mismatch = np.zeros(len(merged), dtype=bool)
    for col in y_cols[1:]:
        mismatch |= merged[col].to_numpy() != merged[ref_col].to_numpy()

    if np.any(mismatch):
        sample_cols = ["group_id", "window_id", *y_cols]
        sample = merged.loc[mismatch, sample_cols].head(10)
        raise ValueError(
            f"Label mismatch across sensors while {context}. Examples:\n{sample.to_string(index=False)}"
        )

    merged["y_true"] = merged[ref_col].astype(int)
    prob_cols = [f"prob_{sensor}" for sensor in present if f"prob_{sensor}" in merged.columns]
    return merged[["group_id", "window_id", "y_true", *prob_cols]]


def _save_stacking_submodel_fit_artifacts(train_result, fit_dir):
    os.makedirs(fit_dir, exist_ok=True)

    train_hist = list(train_result.get("train_losses") or [])
    val_hist = list(train_result.get("val_losses") or [])
    monitor_hist = list(train_result.get("monitor_losses") or [])
    val_source = "none"
    if len(val_hist) > 0:
        val_source = "validation_loader"
    elif len(monitor_hist) > 0:
        val_hist = monitor_hist
        val_source = "monitor_loader"

    max_len = max(len(train_hist), len(val_hist))

    if max_len > 0:
        train_pad = train_hist + [np.nan] * (max_len - len(train_hist))
        val_pad = val_hist + [np.nan] * (max_len - len(val_hist))
        pd.DataFrame(
            {
                "epoch": np.arange(1, max_len + 1),
                "train_loss": train_pad,
                "val_loss": val_pad,
            }
        ).to_csv(os.path.join(fit_dir, "loss_curve.csv"), index=False)
        utils.save_loss_curve_plot(
            train_losses=train_hist,
            val_losses=(val_hist if len(val_hist) > 0 else None),
            out_path=os.path.join(fit_dir, "loss_curve.png"),
        )

    best_val_loss = train_result.get("best_val_loss", float("nan"))
    best_val_out = float(best_val_loss) if np.isfinite(best_val_loss) else None
    with open(os.path.join(fit_dir, "fit_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "epochs_ran": int(train_result.get("epochs_ran", len(train_hist))),
                "best_val_loss": best_val_out,
                "validation_source": val_source,
            },
            fh,
            indent=2,
        )


def _save_stacking_submodel_eval(sensor_bundle, indices, prob_1, out_dir, sensor_name, fold_label, phase, inner_fold=None):
    idx = np.asarray(indices, dtype=int)
    prob_1 = np.asarray(prob_1, dtype=float)
    if idx.size == 0 or prob_1.size == 0:
        return None
    if idx.size != prob_1.size:
        raise ValueError(
            f"Submodel eval size mismatch for {sensor_name}/{phase}: "
            f"indices={idx.size}, prob_1={prob_1.size}"
        )

    probs = np.column_stack([1.0 - prob_1, prob_1])
    metrics = utils.score_and_save_fold_outputs(
        y_test=np.asarray(sensor_bundle["y"][idx], dtype=int),
        test_probs=probs,
        threshold=float(config.DECISION_THRESHOLD),
        fold_dir=out_dir,
        test_bundle=sensor_bundle,
        test_idx=idx,
        experiment_for_outputs=f"stacking_submodel_{sensor_name}",
        sensor_status={"missing": [], "available": [sensor_name]},
        save_arrays=False,
    )

    row = dict(metrics)
    row.update(
        {
            "fold": str(fold_label),
            "sensor": str(sensor_name),
            "phase": str(phase),
            "inner_fold": int(inner_fold) if inner_fold is not None else -1,
            "n_samples": int(idx.size),
        }
    )
    return row


def _fit_predict_base_sensor_prob_1(
    args,
    sensor_name,
    sensor_bundle,
    fit_idx,
    pred_idx,
    seed_offset=0,
    diagnostics_fit_dir=None,
    monitor_X=None,
    monitor_y=None,
):
    import src.models as models

    fit_idx = np.asarray(fit_idx, dtype=int)
    pred_idx = np.asarray(pred_idx, dtype=int)
    if fit_idx.size == 0 or pred_idx.size == 0:
        return np.empty(pred_idx.size, dtype=float)

    X_fit = sensor_bundle["X"][fit_idx]
    y_fit = np.asarray(sensor_bundle["y"][fit_idx], dtype=int)
    X_pred = sensor_bundle["X"][pred_idx]

    if np.unique(y_fit).size < 2:
        constant_prob = float(np.clip(np.mean(y_fit.astype(float)), 0.0, 1.0))
        return np.full(len(pred_idx), constant_prob, dtype=float)

    if args.model in models.CLASSICAL_MODELS:
        _, _, n_channels = X_fit.shape
        scaler = StandardScaler()
        scaler.fit(X_fit.reshape(-1, n_channels))
        X_fit_scaled = scaler.transform(X_fit.reshape(-1, n_channels)).reshape(X_fit.shape)
        X_pred_scaled = scaler.transform(X_pred.reshape(-1, n_channels)).reshape(X_pred.shape)

        if args.sensor_dropout:
            X_fit_src, y_fit_aug = utils.augment_sensor_dropout(
                X_fit_scaled,
                y_fit,
                sensors=[sensor_name],
                block_size=8,
                p=config.SENSOR_DROPOUT_P,
                max_off=config.SENSOR_DROPOUT_MAX_OFF,
                copies=1,
                seed=int(config.SEED + int(seed_offset)),
            )
        else:
            X_fit_src, y_fit_aug = X_fit_scaled, y_fit

        model = models.make_classical_model(args.model, y_fit_aug)
        model.fit(X_fit_src.reshape(len(X_fit_src), -1), y_fit_aug)
        if diagnostics_fit_dir:
            os.makedirs(diagnostics_fit_dir, exist_ok=True)
            with open(os.path.join(diagnostics_fit_dir, "fit_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "model_family": "classical",
                        "model": str(args.model),
                        "train_samples": int(len(y_fit_aug)),
                    },
                    fh,
                    indent=2,
                )
        return utils.estimator_binary_prob_1(model, X_pred_scaled.reshape(len(X_pred_scaled), -1))

    import torch

    norm_mean, norm_std = utils.compute_channel_standardization_stats(
        X_fit,
        prefer_gpu=(config.GPU_NORMALIZATION and config.DEVICE.type == "cuda"),
    )
    train_seed = int(config.SEED + int(seed_offset))
    train_result = train_neural_model(
        args=args,
        X_train=X_fit,
        y_train=y_fit,
        X_val=None,
        y_val=None,
        epochs=config.EPOCHS,
        normalizer=(norm_mean, norm_std),
        drop_sensors_override=[sensor_name],
        seed=train_seed,
        batch_size_override=config.STACKING_BATCH_SIZE,
        preload_to_device=bool(config.STACKING_PRELOAD_TO_GPU),
        monitor_X=monitor_X,
        monitor_y=monitor_y,
    )

    if diagnostics_fit_dir:
        _save_stacking_submodel_fit_artifacts(train_result, diagnostics_fit_dir)

    model = train_result["model"]
    model.eval()
    norm_mean_t, norm_inv_std_t = utils.make_torch_standardizer(norm_mean, norm_std, config.DEVICE)
    pred_loader = utils.make_tensor_loader(
        X_pred,
        y=None,
        shuffle=False,
        batch_size=config.STACKING_BATCH_SIZE,
    )

    logits = []
    with torch.no_grad():
        for xb, _ in pred_loader:
            xb = xb.to(config.DEVICE, non_blocking=True)
            xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)
            with torch.amp.autocast(device_type="cuda", enabled=(config.DEVICE.type == "cuda")):
                out = model(xb)
            logits.append(out.detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0) if logits else np.empty((0, 1), dtype=float)
    probs = utils.logits_to_binary_probs(logits)
    return probs[:, 1].astype(float, copy=False)

def train_experiment(args):
    import src.models as models

    if args.model not in models.CLASSICAL_MODELS:
        config.seed_setup()
    run_name = utils.run_name_for_dataset(args.train_data, args)
    output_dir = os.path.join(config.OUTPUT_ROOT, args.model, run_name)
    model_dir = os.path.join(config.MODELS_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)
    done_flag = os.path.exists(os.path.join(output_dir, "DONE"))
    summary_flag = os.path.exists(os.path.join(output_dir, "summary_metrics.csv"))
    if done_flag or summary_flag:
        if args.model in models.CLASSICAL_MODELS:
            print(f"Run already complete at: {output_dir} - skipping.")
            return output_dir

        existing_inner_curves = glob.glob(os.path.join(output_dir, "fold_*", "inner_loss_curves", "inner_fold_*_loss_curve.csv"))
        if existing_inner_curves:
            print(f"Run already complete at: {output_dir} - skipping.")
            return output_dir

        print(f"[REBUILD] Existing run found at {output_dir}, but inner CV loss curves are missing. Recomputing neural fold artifacts.")

    bundle = utils.load_bundle(args.train_data, args)
    logo = LeaveOneGroupOut()
    logo_splits = list(logo.split(bundle["X"], bundle["y"], bundle["groups"]))
    utils._log_header(
        "individual_generalization::train",
        model=args.model,
        dataset=args.train_data,
        run_name=run_name,
        sensor_dropout=bool(args.sensor_dropout),
        ablation=(args.ablation or "none"),
        logo_folds=len(logo_splits),
    )
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(logo_splits):
        left_out = bundle["groups"][test_idx[0]]
        fold_label = f"s{left_out}"
        utils._log_fold("LOGO", fold_idx, len(logo_splits), left_out=left_out, train=len(train_idx), test=len(test_idx))
        fold_dir = os.path.join(output_dir, f"fold_{fold_label}")
        fold_model_dir = os.path.join(model_dir, f"fold_{fold_label}")
        row = fit_and_eval_fold(args, bundle, train_idx, test_idx, fold_idx, fold_dir, fold_model_dir, bundle["experiment"], {"missing": [], "available": utils.sensors_from_experiment(bundle["experiment"])})
        row["fold"] = fold_label
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    open(os.path.join(output_dir, "DONE"), "w").close()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": "train", "n_folds": len(rows)}, fh, indent=2)
    return output_dir

def train_stacking_experiment(args):
    import src.models as models

    if args.model not in models.CLASSICAL_MODELS:
        config.seed_setup()

    source_bundle = utils.load_bundle(args.train_data, args)
    if source_bundle.get("window_ids") is None:
        raise ValueError("Stacking requires window_ids in the source dataset.")

    source_sensors = list(utils.sensors_from_experiment(args.train_data))
    if len(source_sensors) < 2:
        raise ValueError(
            "Stacking requires a fused training dataset with at least 2 sensors "
            "(e.g., chest_left, chest_right, left_right, chest_left_right)."
        )
    ensemble_sensors = list(source_sensors)

    run_name = f"stacking_{utils.run_name_for_dataset(args.train_data, args)}"
    output_dir = os.path.join(config.OUTPUT_ROOT, args.model, run_name)
    model_dir = os.path.join(config.MODELS_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    done_flag = os.path.exists(os.path.join(output_dir, "DONE"))
    summary_flag = os.path.exists(os.path.join(output_dir, "summary_metrics.csv"))
    if done_flag or summary_flag:
        rebuild_reasons = []
        submodel_summary = os.path.join(output_dir, "summary_metrics_submodels.csv")
        if bool(config.STACKING_SAVE_SUBMODEL_ARTIFACTS) and not os.path.exists(submodel_summary):
            rebuild_reasons.append("submodel diagnostics are missing")

        if bool(getattr(config, "STACKING_META_SAVE_VALIDATION_CURVE", True)):
            meta_curve_files = glob.glob(
                os.path.join(
                    output_dir,
                    "fold_*",
                    "stacking_train",
                    "meta_diagnostics",
                    "meta_validation_loss_curve.csv",
                )
            )
            if len(meta_curve_files) == 0:
                rebuild_reasons.append("meta validation curves are missing")

        if not rebuild_reasons:
            print(f"Run already complete at: {output_dir} - skipping.")
            return output_dir

        print(
            f"[REBUILD] Existing stacking run found at {output_dir}, but "
            + "; ".join(rebuild_reasons)
            + ". Recomputing stacking outputs."
        )

    source_keys_all = _bundle_key_strings(source_bundle)

    sensor_bundles = {}
    sensor_key_maps = {}
    for sensor_name in ensemble_sensors:
        sensor_bundle = utils.load_bundle(sensor_name, args)
        if sensor_bundle.get("window_ids") is None:
            raise ValueError(f"Stacking requires window_ids for sensor dataset '{sensor_name}'.")
        sensor_bundles[sensor_name] = sensor_bundle
        sensor_key_maps[sensor_name] = _build_bundle_key_index(sensor_bundle, context=f"sensor={sensor_name}")

    rows = []
    submodel_rows = []
    logo = LeaveOneGroupOut()
    splits = list(logo.split(source_bundle["X"], source_bundle["y"], source_bundle["groups"]))
    utils._log_header(
        "stacking::meta_train",
        model=args.model,
        source_dataset=args.train_data,
        ensemble_sensors=",".join(ensemble_sensors),
        run_name=run_name,
        sensor_dropout=bool(args.sensor_dropout),
        logo_folds=len(splits),
    )
    for fold_idx, (_, test_idx_source) in enumerate(splits):
        left_out = source_bundle["groups"][test_idx_source[0]]
        fold_label = f"s{left_out}"
        utils._log_fold("LOGO", fold_idx, len(splits), left_out=left_out, source_test=len(test_idx_source))

        outer_test_keys = set(source_keys_all[test_idx_source].tolist())
        outer_train_keys = set(source_keys_all.tolist()) - outer_test_keys

        sensor_train_frames = {}
        sensor_test_frames = {}

        for sensor_pos, sensor_name in enumerate(ensemble_sensors):
            sensor_bundle = sensor_bundles[sensor_name]
            key_to_idx = sensor_key_maps[sensor_name]

            sensor_train_idx = np.asarray(
                sorted(key_to_idx[key] for key in outer_train_keys if key in key_to_idx),
                dtype=int,
            )
            sensor_test_idx = np.asarray(
                sorted(key_to_idx[key] for key in outer_test_keys if key in key_to_idx),
                dtype=int,
            )

            if sensor_train_idx.size == 0 or sensor_test_idx.size == 0:
                continue

            groups_outer_train = sensor_bundle["groups"][sensor_train_idx]
            unique_outer_train_groups = np.unique(groups_outer_train)
            n_inner = min(int(config.INNER_FOLDS), len(unique_outer_train_groups))
            if n_inner < 2:
                continue

            X_outer_train = sensor_bundle["X"][sensor_train_idx]
            y_outer_train = sensor_bundle["y"][sensor_train_idx]
            inner_cv = GroupKFold(n_splits=n_inner)
            oof_prob_1 = np.full(len(sensor_train_idx), np.nan, dtype=float)

            for inner_idx, (inner_fit_local, inner_val_local) in enumerate(
                inner_cv.split(X_outer_train, y_outer_train, groups=groups_outer_train)
            ):
                fit_idx_sensor = sensor_train_idx[inner_fit_local]
                val_idx_sensor = sensor_train_idx[inner_val_local]
                seed_offset = int((fold_idx * 10000) + (sensor_pos * 1000) + inner_idx)

                submodel_base_dir = None
                diagnostics_fit_dir = None
                if bool(config.STACKING_SAVE_SUBMODEL_ARTIFACTS):
                    submodel_base_dir = os.path.join(
                        output_dir,
                        f"fold_{fold_label}",
                        "submodels",
                        f"sensor_{sensor_name}",
                        f"inner_{int(inner_idx)}",
                    )
                    diagnostics_fit_dir = os.path.join(submodel_base_dir, "fit")

                probs_val = _fit_predict_base_sensor_prob_1(
                    args=args,
                    sensor_name=sensor_name,
                    sensor_bundle=sensor_bundle,
                    fit_idx=fit_idx_sensor,
                    pred_idx=val_idx_sensor,
                    seed_offset=seed_offset,
                    diagnostics_fit_dir=diagnostics_fit_dir,
                    monitor_X=(sensor_bundle["X"][val_idx_sensor] if bool(config.STACKING_SAVE_SUBMODEL_ARTIFACTS) else None),
                    monitor_y=(sensor_bundle["y"][val_idx_sensor] if bool(config.STACKING_SAVE_SUBMODEL_ARTIFACTS) else None),
                )
                oof_prob_1[inner_val_local] = probs_val

                if submodel_base_dir is not None:
                    row = _save_stacking_submodel_eval(
                        sensor_bundle=sensor_bundle,
                        indices=val_idx_sensor,
                        prob_1=probs_val,
                        out_dir=os.path.join(submodel_base_dir, "eval_inner_val"),
                        sensor_name=sensor_name,
                        fold_label=fold_label,
                        phase="inner_val",
                        inner_fold=inner_idx,
                    )
                    if row is not None:
                        row["seed_offset"] = int(seed_offset)
                        row["fit_samples"] = int(len(fit_idx_sensor))
                        row["pred_samples"] = int(len(val_idx_sensor))
                        submodel_rows.append(row)

            valid_oof = np.isfinite(oof_prob_1)
            if np.any(valid_oof):
                sensor_train_frames[sensor_name] = _prediction_frame_from_bundle(
                    sensor_bundle,
                    sensor_train_idx[valid_oof],
                    oof_prob_1[valid_oof],
                    sensor_name,
                )

            seed_offset_test = int((fold_idx * 10000) + (sensor_pos * 1000) + 999)

            submodel_outer_dir = None
            diagnostics_outer_fit_dir = None
            if bool(config.STACKING_SAVE_SUBMODEL_ARTIFACTS):
                submodel_outer_dir = os.path.join(
                    output_dir,
                    f"fold_{fold_label}",
                    "submodels",
                    f"sensor_{sensor_name}",
                    "outer_train_fit",
                )
                diagnostics_outer_fit_dir = os.path.join(submodel_outer_dir, "fit")

            probs_test = _fit_predict_base_sensor_prob_1(
                args=args,
                sensor_name=sensor_name,
                sensor_bundle=sensor_bundle,
                fit_idx=sensor_train_idx,
                pred_idx=sensor_test_idx,
                seed_offset=seed_offset_test,
                diagnostics_fit_dir=diagnostics_outer_fit_dir,
            )
            sensor_test_frames[sensor_name] = _prediction_frame_from_bundle(
                sensor_bundle,
                sensor_test_idx,
                probs_test,
                sensor_name,
            )

            if submodel_outer_dir is not None:
                row = _save_stacking_submodel_eval(
                    sensor_bundle=sensor_bundle,
                    indices=sensor_test_idx,
                    prob_1=probs_test,
                    out_dir=os.path.join(submodel_outer_dir, "eval_outer_test"),
                    sensor_name=sensor_name,
                    fold_label=fold_label,
                    phase="outer_test",
                    inner_fold=None,
                )
                if row is not None:
                    row["seed_offset"] = int(seed_offset_test)
                    row["fit_samples"] = int(len(sensor_train_idx))
                    row["pred_samples"] = int(len(sensor_test_idx))
                    submodel_rows.append(row)

        if len(sensor_train_frames) < 2 or len(sensor_test_frames) < 2:
            continue

        meta_train = _merge_sensor_probability_frames(
            sensor_train_frames,
            sensor_order=ensemble_sensors,
            context=f"building stacking meta-train fold {fold_label}",
        )
        meta_test = _merge_sensor_probability_frames(
            sensor_test_frames,
            sensor_order=ensemble_sensors,
            context=f"building stacking meta-test fold {fold_label}",
        )

        if meta_train.empty or meta_test.empty:
            continue

        prob_cols = [c for c in meta_train.columns if c.startswith("prob_") and c in meta_test.columns]
        if len(prob_cols) < 2:
            continue

        X_meta_train = meta_train[prob_cols].to_numpy(dtype=float)
        y_meta_train = meta_train["y_true"].to_numpy(dtype=int)
        g_meta_train = meta_train["group_id"].to_numpy()
        X_meta_test = meta_test[prob_cols].to_numpy(dtype=float)
        y_meta_test = meta_test["y_true"].to_numpy(dtype=int)
        if len(np.unique(y_meta_train)) < 2:
            continue

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_label}", "stacking_train")
        os.makedirs(fold_output_dir, exist_ok=True)
        fold_model_dir = os.path.join(model_dir, f"fold_{fold_label}")
        os.makedirs(fold_model_dir, exist_ok=True)

        if bool(getattr(config, "STACKING_META_SAVE_VALIDATION_CURVE", True)):
            _save_meta_validation_curve(
                X=X_meta_train,
                y=y_meta_train,
                groups=g_meta_train,
                out_dir=os.path.join(fold_output_dir, "meta_diagnostics"),
                seed=int(config.SEED + int(fold_idx) * 1000),
            )

        X_meta_train_fit = np.asarray(X_meta_train, dtype=float)
        y_meta_train_fit = np.asarray(y_meta_train, dtype=int)

        meta_dropout_applied = False
        meta_dropout_p = None
        meta_dropout_max_off = None
        if args.sensor_dropout:
            meta_dropout_applied = True
            meta_dropout_p = float(config.SENSOR_DROPOUT_P)
            meta_dropout_max_off = int(min(config.SENSOR_DROPOUT_MAX_OFF, len(prob_cols)))
            X_meta_train_fit, y_meta_train_fit = utils._augment_meta_feature_dropout(
                X=X_meta_train_fit,
                y=y_meta_train_fit,
                p=meta_dropout_p,
                max_off=meta_dropout_max_off,
                copies=1,
                seed=config.SEED + int(fold_idx),
            )

        meta_model = models.make_classical_model("LogisticRegression", y_meta_train_fit)
        meta_model.fit(X_meta_train_fit, y_meta_train_fit)
        meta_model_type = "sklearn_logistic_regression_lbfgs"

        best_thr, best_score, threshold_tuning_strategy = _tune_meta_threshold_with_group_oof(
            X=X_meta_train,
            y=y_meta_train,
            groups=g_meta_train,
        )
        p_test = _predict_meta_prob_1(meta_model, X_meta_test)
        stacking_probs = np.column_stack([1.0 - p_test, p_test])

        joblib.dump(meta_model, os.path.join(fold_model_dir, "meta_stacking.joblib"))
        with open(os.path.join(fold_output_dir, "meta_config.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "threshold": float(best_thr),
                "threshold_tuning_score": float(best_score),
                "threshold_tuning_strategy": threshold_tuning_strategy,
                "features": prob_cols,
                "meta_model_type": meta_model_type,
                "meta_feature_sensor_dropout": {
                    "enabled": bool(meta_dropout_applied),
                    "p": meta_dropout_p,
                    "max_off": meta_dropout_max_off,
                },
            }, fh, indent=2)

        stacking_bundle = {
            "groups": meta_test["group_id"].to_numpy(),
            "window_ids": meta_test["window_id"].astype(object).to_numpy(),
        }
        metrics = utils.score_and_save_fold_outputs(
            y_test=y_meta_test,
            test_probs=stacking_probs,
            threshold=best_thr,
            fold_dir=fold_output_dir,
            test_bundle=stacking_bundle,
            test_idx=np.arange(len(meta_test), dtype=int),
            experiment_for_outputs=args.train_data,
            sensor_status={"missing": [], "available": ensemble_sensors},
            save_arrays=False,
        )
        metrics["fold"] = fold_label
        metrics["method"] = "stacking_train"
        metrics["threshold_tuning_score"] = float(best_score)
        rows.append(metrics)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary_metrics_stacking_train.csv"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    if submodel_rows:
        pd.DataFrame(submodel_rows).to_csv(os.path.join(output_dir, "summary_metrics_submodels.csv"), index=False)
    open(os.path.join(output_dir, "DONE"), "w").close()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": "stacking_train",
                "n_folds": len(rows),
                "source_sensors": source_sensors,
                "sensors_used": ensemble_sensors,
                "source_dataset": args.train_data,
            },
            fh,
            indent=2,
        )

    return output_dir