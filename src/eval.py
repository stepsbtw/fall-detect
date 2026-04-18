import json
import os

import joblib
import numpy as np

import src.config as config
import src.utils as utils

OUTPUT_ROOT = config.OUTPUT_ROOT
MODELS_ROOT = config.MODELS_ROOT


def _toggle_with_rifle_window_id(window_id):
    text = str(window_id)
    parts = text.split("|")
    if len(parts) < 5:
        return None
    if parts[2] not in {"0", "1"}:
        return None
    parts[2] = "1" if parts[2] == "0" else "0"
    return "|".join(parts)

def _predict_with_trained_fold(args, train_bundle, test_bundle, fit_idx, test_idx, fold_label,
                               trained_fold_output_dir, trained_fold_model_dir):
    from sklearn.preprocessing import StandardScaler

    X_test = test_bundle["X"][test_idx]
    y_test = test_bundle["y"][test_idx]

    threshold = float(config.DECISION_THRESHOLD)
    metrics_path = os.path.join(trained_fold_output_dir, "metrics.csv")
    threshold = utils.threshold_from_metrics_csv(metrics_path, default_threshold=threshold)

    if args.model in config.CLASSICAL_MODELS:
        scaler_path = os.path.join(trained_fold_model_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            X_fit = train_bundle["X"][fit_idx]
            _, _, n_channels = X_fit.shape
            scaler = StandardScaler()
            scaler.fit(X_fit.reshape(-1, n_channels))

        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
        model_path = os.path.join(trained_fold_model_dir, f"{fold_label}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing trained classical model: {model_path}")
        model = joblib.load(model_path)
        X_test_fit = X_test_scaled.reshape(len(X_test_scaled), -1)
        prob_1 = utils.estimator_binary_prob_1(model, X_test_fit)
        test_probs = np.column_stack([1.0 - prob_1, prob_1])
    else:
        import torch

        import src.models as models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        norm_mean_t, norm_inv_std_t = None, None
        X_test_for_model = X_test

        norm_stats = utils.load_channel_standardization_stats(
            os.path.join(trained_fold_model_dir, "scaler_stats.npz")
        )
        if norm_stats is None:
            # Backward compatibility with older neural runs that only saved scaler.joblib.
            scaler_path = os.path.join(trained_fold_model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                X_fit = train_bundle["X"][fit_idx]
                _, _, n_channels = X_fit.shape
                scaler = StandardScaler()
                scaler.fit(X_fit.reshape(-1, n_channels))
            X_test_for_model = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
        else:
            norm_mean, norm_std = norm_stats
            norm_mean_t, norm_inv_std_t = utils.make_torch_standardizer(norm_mean, norm_std, device)

        input_shape = utils.model_input_shape(args.model, X_test_for_model)
        model = models.create_model(args.model, input_shape, 1)
        model_path = os.path.join(trained_fold_model_dir, f"{fold_label}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing trained neural model: {model_path}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        test_loader = utils.make_tensor_loader(X_test_for_model, y=None, shuffle=False)
        logits = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=True)
                xb = utils.standardize_batch_torch(xb, norm_mean_t, norm_inv_std_t)
                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    out = model(xb)
                logits.append(out.detach().cpu().numpy())
        logits = np.concatenate(logits, axis=0) if logits else np.empty((0, 1), dtype=float)
        test_probs = utils.logits_to_binary_probs(logits)

    return y_test, test_probs, float(threshold)


def infer_with_trained_fold(args, train_bundle, test_bundle, fit_idx, test_idx, fold_label,
                            trained_fold_output_dir, trained_fold_model_dir, out_fold_dir, experiment_for_outputs, sensor_status):
    import pandas as pd

    if os.path.exists(os.path.join(out_fold_dir, "metrics.csv")):
        return pd.read_csv(os.path.join(out_fold_dir, "metrics.csv")).iloc[0].to_dict()

    y_test, test_probs, threshold = _predict_with_trained_fold(
        args=args,
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        fit_idx=fit_idx,
        test_idx=test_idx,
        fold_label=fold_label,
        trained_fold_output_dir=trained_fold_output_dir,
        trained_fold_model_dir=trained_fold_model_dir,
    )

    return utils.score_and_save_fold_outputs(y_test=y_test, test_probs=test_probs, threshold=threshold, fold_dir=out_fold_dir,
        test_bundle=test_bundle, test_idx=test_idx, experiment_for_outputs=experiment_for_outputs, sensor_status=sensor_status, save_arrays=False)


def eval_cross_sensor_experiment(args):
    import pandas as pd
    from sklearn.model_selection import LeaveOneGroupOut

    if args.experiment != "cross_sensor":
        raise ValueError("test must be cross_sensor for cross_sensor_experiment")
    source_bundle = utils.load_bundle(args.train_data, args)
    target_bundle = utils.load_bundle(args.test_data, args)
    trained_run_name = utils.run_name_for_dataset(args.train_data, args)
    trained_output_dir = os.path.join(OUTPUT_ROOT, args.model, trained_run_name)
    trained_model_dir = os.path.join(MODELS_ROOT, args.model, trained_run_name)
    if not os.path.exists(trained_output_dir) or not os.path.exists(trained_model_dir):
        raise FileNotFoundError(
            f"Missing trained run for source dataset {args.train_data}: {trained_output_dir}. "
            "Run individual_generalization first for this dataset/config."
        )

    run_name = f"cross_sensor_{utils.run_name_for_dataset(args.train_data, args)}_to_{args.test_data}"
    output_dir = os.path.join(OUTPUT_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    logo = LeaveOneGroupOut()
    splits = list(logo.split(source_bundle["X"], source_bundle["y"], source_bundle["groups"]))
    target_pair_to_indices = {}
    for idx, pair in enumerate(zip(target_bundle["groups"], target_bundle["window_ids"])):
        group_id, window_id = pair
        target_pair_to_indices.setdefault((group_id, str(window_id)), []).append(idx)
    utils._log_header(
        "cross_sensor::eval",
        model=args.model,
        train_dataset=args.train_data,
        test_dataset=args.test_data,
        run_name=run_name,
        logo_folds=len(splits),
    )

    for fold_idx, (train_idx, test_idx_source) in enumerate(splits):
        left_out = source_bundle["groups"][test_idx_source[0]]
        utils._log_fold("LOGO", fold_idx, len(splits), left_out=left_out, train=len(train_idx), source_test=len(test_idx_source))
        wanted = set(
            zip(
                source_bundle["groups"][test_idx_source],
                np.asarray(source_bundle["window_ids"][test_idx_source], dtype=object).astype(str),
            )
        )

        aligned_exact = sorted(
            i
            for pair in wanted
            for i in target_pair_to_indices.get(pair, [])
        )

        if aligned_exact:
            aligned = aligned_exact
        else:
            aligned_fallback = sorted(
                i
                for group_id, window_id in wanted
                for i in target_pair_to_indices.get(
                    (group_id, _toggle_with_rifle_window_id(window_id)),
                    [],
                )
            )
            aligned = aligned_fallback
            if aligned:
                print(
                    f"[INFO] fold s{left_out}: using with_rifle-toggled window_id alignment "
                    f"({len(aligned)} samples)."
                )

        if not aligned:
            continue
        fold_label = f"s{left_out}"
        fold_dir = os.path.join(output_dir, f"fold_{fold_label}")
        trained_fold_output_dir = os.path.join(trained_output_dir, f"fold_{fold_label}")
        trained_fold_model_dir = os.path.join(trained_model_dir, f"fold_{fold_label}")
        row = infer_with_trained_fold(args, train_bundle=source_bundle, test_bundle=target_bundle, fit_idx=train_idx,
            test_idx=np.asarray(aligned), fold_label=fold_label, trained_fold_output_dir=trained_fold_output_dir, 
            trained_fold_model_dir=trained_fold_model_dir, out_fold_dir=fold_dir, experiment_for_outputs=args.test_data, 
            sensor_status={"missing": [s for s in utils.sensors_from_experiment(args.train_data) if s not in utils.sensors_from_experiment(args.test_data)], 
                           "available": utils.sensors_from_experiment(args.test_data)})
        row["fold"] = fold_label
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    open(os.path.join(output_dir, "DONE"), "w").close()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": "cross_sensor", "n_folds": len(rows), "target_dataset": args.test_data}, fh, indent=2)
    return output_dir


def eval_missing_sensor_experiment(args):
    import pandas as pd
    from sklearn.model_selection import LeaveOneGroupOut

    if args.experiment != "missing_sensor":
        raise ValueError("test must be missing_sensor for missing_sensor_experiment")

    train_sensors = tuple(utils.sensors_from_experiment(args.train_data))
    test_sensors = tuple(utils.sensors_from_experiment(args.test_data))
    train_shape = utils.SCENARIOS[args.train_data][2]

    allowed_targets = [name for name, (_, _, shape) in utils.SCENARIOS.items() 
                       if shape[0] == train_shape[0] and tuple(utils.sensors_from_experiment(name)) != train_sensors 
                       and set(utils.sensors_from_experiment(name)).issubset(set(train_sensors))]
    if args.test_data not in allowed_targets:
        raise ValueError(f"Invalid fused-missing pair: {args.train_data} -> {args.test_data}. " f"Allowed test experiments: {allowed_targets}")

    train_run_name = utils.run_name_for_dataset(args.train_data, args)
    trained_output_dir = os.path.join(OUTPUT_ROOT, args.model, train_run_name)
    trained_model_dir = os.path.join(MODELS_ROOT, args.model, train_run_name)
    if not os.path.exists(trained_output_dir) or not os.path.exists(trained_model_dir):
        raise FileNotFoundError(
            f"Missing trained run for source dataset {args.train_data}: {trained_output_dir}. "
            "Run individual_generalization first for this dataset/config."
        )

    run_name = f"missing_sensor_{train_run_name}_on_{args.test_data}"
    output_dir = os.path.join(OUTPUT_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    train_bundle = utils.load_bundle(args.train_data, args)
    test_bundle = utils.load_bundle(args.test_data, args)

    train_bundle["X"] = utils.expand_to_train_layout(train_bundle["X"], train_sensors, train_sensors)
    test_bundle["X"] = utils.expand_to_train_layout(test_bundle["X"], test_sensors, train_sensors)

    rows = []
    logo = LeaveOneGroupOut()
    splits = list(logo.split(test_bundle["X"], test_bundle["y"], test_bundle["groups"]))
    utils._log_header(
        "missing_sensor::single_model_eval",
        model=args.model,
        train_dataset=args.train_data,
        test_dataset=args.test_data,
        run_name=run_name,
        source_sensors=",".join(train_sensors),
        target_sensors=",".join(test_sensors),
        logo_folds=len(splits),
    )

    for fold_idx, (_, test_idx) in enumerate(splits):
        left_out = test_bundle["groups"][test_idx[0]]
        fold_label = f"s{left_out}"
        utils._log_fold("LOGO", fold_idx, len(splits), left_out=left_out, target_test=len(test_idx))

        fold_dir = os.path.join(output_dir, f"fold_{fold_label}")
        os.makedirs(fold_dir, exist_ok=True)

        fit_idx = np.where(train_bundle["groups"] != left_out)[0]
        sensor_status = {
            "missing": [s for s in train_sensors if s not in test_sensors],
            "available": list(test_sensors),
        }

        trained_fold_output_dir = os.path.join(trained_output_dir, f"fold_{fold_label}")
        trained_fold_model_dir = os.path.join(trained_model_dir, f"fold_{fold_label}")
        row = infer_with_trained_fold(args=args, train_bundle=train_bundle, test_bundle=test_bundle, fit_idx=fit_idx,
            test_idx=test_idx, fold_label=fold_label, trained_fold_output_dir=trained_fold_output_dir, trained_fold_model_dir=trained_fold_model_dir, out_fold_dir=fold_dir, experiment_for_outputs=args.test_data, sensor_status=sensor_status)
        row["fold"] = fold_label
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    open(os.path.join(output_dir, "DONE"), "w").close()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": "missing_sensor", "n_folds": len(rows), "target_dataset": args.test_data}, fh, indent=2)

    return output_dir

def eval_bagging_experiment(args):
    import pandas as pd

    if args.experiment != "bagging":
        raise ValueError("test must be bagging for bagging_experiment")
    if args.test_data is None:
        raise ValueError("--test_data is required for bagging")

    target_sensors = list(utils.sensors_from_experiment(args.test_data))
    if len(target_sensors) < 2:
        raise ValueError("Bagging requires a fused test dataset with at least 2 sensors.")

    _, ensemble_sensors, trained = utils.resolve_available_sensor_runs(
        args,
        dataset_name=args.test_data,
        min_sensors=2,
        require_model_dir=False,
    )

    run_name = f"bagging_{utils.run_name_for_dataset(args.test_data, args)}"
    output_dir = os.path.join(OUTPUT_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    if os.path.exists(os.path.join(output_dir, "DONE")) or os.path.exists(os.path.join(output_dir, "summary_metrics.csv")):
        print(f"Run already complete at: {output_dir} - skipping.")
        return output_dir

    fold_sets = []
    for sensor_name in ensemble_sensors:
        sensor_output = trained[sensor_name]["output"]
        folds = {
            name
            for name in os.listdir(sensor_output)
            if name.startswith("fold_") and os.path.isdir(os.path.join(sensor_output, name))
        }
        fold_sets.append(folds)

    common_folds = sorted(set.intersection(*fold_sets) if fold_sets else set())
    utils._log_header(
        "bagging::eval_from_sensor_outputs",
        model=args.model,
        test_dataset=args.test_data,
        ensemble_sensors=",".join(ensemble_sensors),
        run_name=run_name,
        folds=len(common_folds),
    )

    fold_bag_data = {}
    for fold_idx, fold_name in enumerate(common_folds):
        fold_label = fold_name.replace("fold_", "")
        utils._log_fold("LOGO", fold_idx, len(common_folds), fold=fold_label)

        sensor_frames = {}
        for sensor_name in ensemble_sensors:
            pred_path = os.path.join(trained[sensor_name]["output"], fold_name, "predictions.csv")
            if not os.path.exists(pred_path):
                continue

            frame = utils._normalize_prediction_df(pd.read_csv(pred_path)).rename(
                columns={"y_true": f"y_true_{sensor_name}", "y_prob_1": f"prob_{sensor_name}"}
            )
            sensor_frames[sensor_name] = frame

        if len(sensor_frames) < 2:
            continue

        present_sensors = [sensor for sensor in ensemble_sensors if sensor in sensor_frames]
        first_sensor = present_sensors[0]
        merged = sensor_frames[first_sensor]
        for sensor_name in present_sensors[1:]:
            merged = merged.merge(sensor_frames[sensor_name], on=["group_id", "window_id"], how="inner")

        if merged.empty:
            continue

        y_cols = [f"y_true_{sensor_name}" for sensor_name in present_sensors if f"y_true_{sensor_name}" in merged.columns]
        ref_col = y_cols[0]
        mismatch = np.zeros(len(merged), dtype=bool)
        for col in y_cols[1:]:
            mismatch |= merged[col].to_numpy() != merged[ref_col].to_numpy()
        if np.any(mismatch):
            sample_cols = ["group_id", "window_id", *y_cols]
            sample = merged.loc[mismatch, sample_cols].head(10)
            raise ValueError(
                "Label mismatch across sensors while building bagging features. "
                f"Examples:\n{sample.to_string(index=False)}"
            )

        merged["y_true"] = merged[ref_col].astype(int)

        prob_cols = [c for c in merged.columns if c.startswith("prob_")]
        if len(prob_cols) < 2:
            continue

        bag_prob_1 = merged[prob_cols].mean(axis=1).to_numpy(dtype=float)
        fold_bag_data[fold_name] = pd.DataFrame(
            {
                "group_id": merged["group_id"].to_numpy(),
                "window_id": merged["window_id"].astype(object).to_numpy(),
                "y_true": merged["y_true"].to_numpy(dtype=int),
                "bag_prob_1": bag_prob_1,
            }
        )

    rows_bagging = []
    for fold_name in common_folds:
        fold_label = fold_name.replace("fold_", "")
        fold_df = fold_bag_data.get(fold_name)
        if fold_df is None or fold_df.empty:
            continue

        threshold = float(config.DECISION_THRESHOLD)
        threshold_score = float("nan")

        train_parts = [df for name, df in fold_bag_data.items() if name != fold_name and len(df) > 0]
        if train_parts:
            tune_df = pd.concat(train_parts, ignore_index=True)
            y_tune = tune_df["y_true"].to_numpy(dtype=int)
            p_tune = tune_df["bag_prob_1"].to_numpy(dtype=float)
            if np.unique(y_tune).size >= 2:
                threshold, threshold_score = utils.tune_threshold_f1(y_tune, p_tune)

        bag_prob_1 = fold_df["bag_prob_1"].to_numpy(dtype=float)
        bag_probs = np.column_stack([1.0 - bag_prob_1, bag_prob_1])

        bag_fold_dir = os.path.join(output_dir, fold_name, "bagging")
        os.makedirs(bag_fold_dir, exist_ok=True)
        bag_test_bundle = {
            "groups": fold_df["group_id"].to_numpy(),
            "window_ids": fold_df["window_id"].astype(object).to_numpy(),
        }

        bag_metrics = utils.score_and_save_fold_outputs(
            y_test=fold_df["y_true"].to_numpy(dtype=int),
            test_probs=bag_probs,
            threshold=float(threshold),
            fold_dir=bag_fold_dir,
            test_bundle=bag_test_bundle,
            test_idx=np.arange(len(fold_df), dtype=int),
            experiment_for_outputs=args.test_data,
            sensor_status={"missing": [], "available": ensemble_sensors},
            save_arrays=False,
        )
        bag_metrics["fold"] = fold_label
        bag_metrics["method"] = "bagging"
        bag_metrics["threshold_tuning_score"] = float(threshold_score)
        rows_bagging.append(bag_metrics)

    if rows_bagging:
        pd.DataFrame(rows_bagging).to_csv(os.path.join(output_dir, "summary_metrics_bagging.csv"), index=False)
        pd.DataFrame(rows_bagging).to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    open(os.path.join(output_dir, "DONE"), "w").close()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": "bagging",
                "n_folds": len(rows_bagging),
                "sensors_used": ensemble_sensors,
                "target_dataset": args.test_data,
            },
            fh,
            indent=2,
        )

    return output_dir