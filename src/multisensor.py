#!/usr/bin/env python3
"""Ensembling and stacking utilities for per-sensor models.

Examples:
    python multisensor.py ensemble --model CNN1D
    python multisensor.py stacking --model CNN1D
    python multisensor.py all --model CNN1D --inner-val-groups 3
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut

from config import Config
from sensor_fusion import CANONICAL_SENSORS

BASE_SCENARIOS = {
    "chest": "chest_T",
    "left": "left_T",
    "right": "right_T",
}


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

    base_dir = Path(Config.get_output_dir(model, scenario_out))
    fold_files = sorted(base_dir.glob("fold_s*/predictions.csv"))
    if not fold_files:
        raise FileNotFoundError(f"No predictions.csv files found for {sensor_name} at {base_dir}")

    frames = []
    for fp in fold_files:
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
    out = out.sort_values(["group_id", "window_id"]).drop_duplicates(
        ["group_id", "window_id"],
        keep="last",
    )

    return out, scenario_out


def build_meta_dataframe(model, args):
    merged = None
    scenario_tags = {}
    sensor_frames = {}

    for sensor in CANONICAL_SENSORS:
        df, tag = load_predictions_for_sensor(model, sensor, args)
        scenario_tags[sensor] = tag
        sensor_frames[sensor] = df
        df = df.rename(columns={"y_true": f"y_true_{sensor}"})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["window_id", "group_id"], how="inner")
    report_window_id_overlap(sensor_frames)

    if merged is None or merged.empty:
        raise ValueError("No aligned per-sensor predictions were found.")

    # validate labels instead of using y_true as part of the join key
    ref_sensor = CANONICAL_SENSORS[0]
    ref_col = f"y_true_{ref_sensor}"

    for sensor in CANONICAL_SENSORS[1:]:
        col = f"y_true_{sensor}"
        mismatch = merged[ref_col].to_numpy() != merged[col].to_numpy()
        if mismatch.any():
            bad = merged.loc[mismatch, ["group_id", "window_id", ref_col, col]].head(20)
            raise ValueError(
                f"y_true mismatch between {ref_sensor} and {sensor} after window_id alignment.\n"
                f"Examples:\n{bad.to_string(index=False)}"
            )

    merged["y_true"] = merged[ref_col].astype(int)

    sort_cols = ["group_id", "window_id"]
    if "sample_index_chest" in merged.columns:
        sort_cols.append("sample_index_chest")

    return merged.sort_values(sort_cols).reset_index(drop=True), scenario_tags


def available_sensor_conditions():
    return {
        "all_present": ["chest", "left", "right"],
        "missing_chest": ["left", "right"],
        "missing_left": ["chest", "right"],
        "missing_right": ["chest", "left"],
    }

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
            print(
                f"[window_id overlap] {a} vs {b}: "
                f"common={common} only_{a}={only_a} only_{b}={only_b}"
            )

def compute_metrics(y_true, y_prob, threshold=0.5):
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


def save_condition_outputs(output_dir, condition_name, df, y_prob, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    metrics, y_pred = compute_metrics(df["y_true"].to_numpy(), y_prob, threshold=threshold)
    per_sample = df.copy()
    per_sample["y_prob_fused"] = y_prob
    per_sample["y_pred_fused"] = y_pred
    per_sample.to_csv(os.path.join(output_dir, f"predictions_{condition_name}.csv"), index=False)
    pd.DataFrame([{"condition": condition_name, **metrics}]).to_csv(
        os.path.join(output_dir, f"metrics_{condition_name}.csv"), index=False
    )
    return metrics


def run_ensemble(args):
    df, scenario_tags = build_meta_dataframe(args.model, args)
    threshold = float(args.threshold)
    output_dir = Path(Config.get_output_dir(args.model, f"multisensor_ensemble_{args.tag}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition_name, available in available_sensor_conditions().items():
        probs = df[[f"p_{sensor}" for sensor in available]].mean(axis=1).to_numpy()
        metrics = save_condition_outputs(output_dir, condition_name, df, probs, threshold=threshold)
        rows.append({"method": "ensemble", "condition": condition_name, "available_sensors": ",".join(available), **metrics})

    pd.DataFrame(rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "threshold": threshold, "source_scenarios": scenario_tags}, f, indent=2)
    print(f"Ensemble results saved to: {output_dir}")



def prepare_stacking_features(df, available):
    X = np.column_stack([
        df["p_chest"].to_numpy() if "chest" in available else np.zeros(len(df), dtype=float),
        df["p_left"].to_numpy() if "left" in available else np.zeros(len(df), dtype=float),
        df["p_right"].to_numpy() if "right" in available else np.zeros(len(df), dtype=float),
        np.full(len(df), 1.0 if "chest" in available else 0.0),
        np.full(len(df), 1.0 if "left" in available else 0.0),
        np.full(len(df), 1.0 if "right" in available else 0.0),
    ])
    return X



def run_stacking(args):
    df, scenario_tags = build_meta_dataframe(args.model, args)
    output_dir = Path(Config.get_output_dir(args.model, f"multisensor_stacking_{args.tag}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(args.threshold)

    logo = LeaveOneGroupOut()
    groups = df["group_id"].to_numpy()
    y = df["y_true"].to_numpy().astype(int)
    summary_rows = []

    for condition_name, available in available_sensor_conditions().items():
        X = prepare_stacking_features(df, available)
        all_probs = np.zeros(len(df), dtype=float)
        all_preds = np.zeros(len(df), dtype=int)
        fold_rows = []

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=Config.SEED + fold_idx)
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            preds = (probs >= threshold).astype(int)
            all_probs[test_idx] = probs
            all_preds[test_idx] = preds
            metrics, _ = compute_metrics(y[test_idx], probs, threshold=threshold)
            fold_rows.append({"fold": fold_idx + 1, "left_out_group": int(groups[test_idx][0]), **metrics})

        out_df = df.copy()
        out_df["y_prob_stacked"] = all_probs
        out_df["y_pred_stacked"] = all_preds
        out_df.to_csv(output_dir / f"predictions_{condition_name}.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / f"fold_metrics_{condition_name}.csv", index=False)
        metrics, _ = compute_metrics(y, all_probs, threshold=threshold)
        summary_rows.append({"method": "stacking", "condition": condition_name, "available_sensors": ",".join(available), **metrics})

        final_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=Config.SEED)
        final_clf.fit(X, y)
        joblib.dump(final_clf, output_dir / f"stacker_{condition_name}.pkl")

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "threshold": threshold, "source_scenarios": scenario_tags}, f, indent=2)
    print(f"Stacking results saved to: {output_dir}")



def build_parser():
    parser = argparse.ArgumentParser(description="Per-sensor ensemble and stacking runner")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("--model", required=True, choices=list(Config.DEFAULT_PARAMS.keys()))
        p.add_argument("--loss", choices=["weighted", "unweighted"], default="weighted")
        p.add_argument("--inner-val-groups", type=int, default=1)
        p.add_argument("--scale", action="store_true", default=False)
        p.add_argument("--no-mag", dest="no_mag", action="store_true", default=False)
        p.add_argument("--only-mag", dest="only_mag", action="store_true", default=False)
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--tag", default="default", help="Suffix used in the output folder name.")

    add_common(sub.add_parser("ensemble"))
    add_common(sub.add_parser("stacking"))
    add_common(sub.add_parser("all"))
    return parser



def main():
    args = build_parser().parse_args()
    if args.mode in {"ensemble", "all"}:
        run_ensemble(args)
    if args.mode in {"stacking", "all"}:
        run_stacking(args)


if __name__ == "__main__":
    main()
