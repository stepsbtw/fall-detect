#!/usr/bin/env python3
"""Evaluate a fused model on smaller sensor scenarios by zero-padding missing sensor blocks.

Example:
    python fused_missing_eval.py --model CNN1D --train-scenario chest_left_right_T --test-scenario chest_T --sensor-dropout
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from config import Config
from sensor_fusion import expand_to_canonical, sensors_from_scenario
from test import save_results, save_results_classical, load_model_state
from training import create_model, _input_shape_from_data, _make_classical_model, drop_mag_channels, keep_only_mag_channels


def scenario_output_name(model, scenario, loss="weighted", inner_val_groups=3, scale=False, no_mag=False, only_mag=False, sensor_dropout=False, sensor_dropout_p=0.5, sensor_dropout_max_off=1):
    scenario_out = scenario if loss == "weighted" else scenario + "_NW"
    if model not in Config.CLASSICAL_MODELS:
        scenario_out = f"{scenario_out}_IVG{max(int(inner_val_groups), 1)}"
    if scale:
        scenario_out = f"{scenario_out}_SC"
    if no_mag:
        scenario_out = f"{scenario_out}_NM"
    if only_mag:
        scenario_out = f"{scenario_out}_OM"
    if sensor_dropout:
        scenario_out = f"{scenario_out}_SDP{str(sensor_dropout_p).replace('.', 'p')}_M{int(sensor_dropout_max_off)}"
    return scenario_out


def evaluate(args):
    train_out = scenario_output_name(
        args.model, args.train_scenario, args.loss, args.inner_val_groups, args.scale, args.no_mag, args.only_mag,
        args.sensor_dropout, args.sensor_dropout_p, args.sensor_dropout_max_off,
    )
    model_root = Path(Config.get_models_dir(args.model, train_out))
    output_root = Path(Config.get_output_dir(args.model, f"padded_eval_{train_out}_on_{args.test_scenario}"))
    output_root.mkdir(parents=True, exist_ok=True)

    X = np.load(Config.get_data_file(args.test_scenario))
    y = np.load(Config.get_labels_file(args.test_scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(args.test_scenario))
    window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(scenario)), "window_ids.npy")
    window_ids = np.load(window_ids_path, allow_pickle=True) if os.path.exists(window_ids_path) else None

    if args.no_mag:
        X = drop_mag_channels(X)
    if args.only_mag:
        X = keep_only_mag_channels(X)

    X = expand_to_canonical(X, args.test_scenario)
    logo = LeaveOneGroupOut()
    threshold = Config.DEFAULT_PARAMS[args.model].get("decision_threshold", 0.5)
    rows = []

    for _, (_, test_idx) in enumerate(logo.split(X, y, groups)):
        left_out = groups[test_idx[0]]
        fold_label = f"s{left_out}"
        fold_dir = output_root / f"fold_{fold_label}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        X_test = X[test_idx]
        y_test = y[test_idx]

        if args.scale:
            train_X = np.load(Config.get_data_file(args.train_scenario))
            train_y = np.load(Config.get_labels_file(args.train_scenario)).astype(np.int64)
            train_groups = np.load(Config.get_groups_file(args.train_scenario))
            train_window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(args.train_scenario)), "window_ids.npy")
            train_window_ids = np.load(train_window_ids_path, allow_pickle=True) if os.path.exists(train_window_ids_path) else None
            if args.no_mag:
                train_X = drop_mag_channels(train_X)
            if args.only_mag:
                train_X = keep_only_mag_channels(train_X)
            train_mask = train_groups != left_out
            X_fit = train_X[train_mask]
            n_tr, t_steps, n_ch = X_fit.shape
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_fit.reshape(-1, n_ch)).reshape(n_tr, t_steps, n_ch)
            X_test = scaler.transform(X_test.reshape(-1, n_ch)).reshape(X_test.shape[0], t_steps, n_ch)

        if args.model in Config.CLASSICAL_MODELS:
            model_path = model_root / f"fold_{fold_label}" / f"model_{fold_label}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            clf = joblib.load(model_path)
            save_results_classical(
                clf=clf,
                X_test_flat=X_test.reshape(len(X_test), -1),
                y_test=y_test,
                decision_threshold=threshold,
                i=fold_label,
                output_dir=str(fold_dir),
                save_model=False,
                sample_indices=test_idx,
                group_ids=groups[test_idx],
                window_ids=window_ids[test_idx] if window_ids is not None else None,
                scenario_name=args.test_scenario,
                sensor_status={"missing": [s for s in sensors_from_scenario(args.train_scenario) if s not in sensors_from_scenario(args.test_scenario)], "available": sensors_from_scenario(args.test_scenario)},
            )
        else:
            input_shape = _input_shape_from_data(X_test, args.model)
            model = create_model(args.model, Config.DEFAULT_PARAMS[args.model], input_shape, Config.NUM_LABELS)
            model_path = model_root / f"fold_{fold_label}" / f"model_{fold_label}.pt"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = load_model_state(model, str(model_path), device=str(Config.DEVICE))
            model.to(Config.DEVICE)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
                batch_size=Config.TRAINING_CONFIG["batch_size"],
                shuffle=False,
            )
            save_results(
                model=model,
                val_loader=loader,
                y_val_onehot=y_test,
                i=fold_label,
                decision_threshold=threshold,
                output_dir=str(fold_dir),
                device=Config.DEVICE,
                save_model=False,
                sample_indices=test_idx,
                group_ids=groups[test_idx],
                window_ids=window_ids[test_idx] if window_ids is not None else None,
                scenario_name=args.test_scenario,
                sensor_status={"missing": [s for s in sensors_from_scenario(args.train_scenario) if s not in sensors_from_scenario(args.test_scenario)], "available": sensors_from_scenario(args.test_scenario)},
            )

        metrics_path = fold_dir / "metrics.csv"
        if metrics_path.exists():
            row = pd.read_csv(metrics_path).iloc[0].to_dict()
            row["fold"] = fold_label
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(output_root / "summary_metrics.csv", index=False)
    print(f"Padded fused evaluation saved to: {output_root}")



def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a fused model on a smaller scenario by zero-padding missing sensors.")
    parser.add_argument("--model", required=True, choices=list(Config.DEFAULT_PARAMS.keys()))
    parser.add_argument("--train-scenario", default="chest_left_right_T")
    parser.add_argument("--test-scenario", required=True, choices=list(Config.SCENARIOS.keys()))
    parser.add_argument("--loss", choices=["weighted", "unweighted"], default="weighted")
    parser.add_argument("--inner-val-groups", type=int, default=1)
    parser.add_argument("--scale", action="store_true", default=False)
    parser.add_argument("--no-mag", dest="no_mag", action="store_true", default=False)
    parser.add_argument("--only-mag", dest="only_mag", action="store_true", default=False)
    parser.add_argument("--sensor-dropout", action="store_true", default=False)
    parser.add_argument("--sensor-dropout-p", type=float, default=0.5)
    parser.add_argument("--sensor-dropout-max-off", type=int, default=1)
    return parser


if __name__ == "__main__":
    Config.setup_device()
    Config.set_seed()
    evaluate(build_parser().parse_args())
