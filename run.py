#!/usr/bin/env python3

import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np

# Project imports
from src.config import Config
from src.train import main as train_main
from src.validation import main as validation_main
from src.test import run_cross_sensor_eval, evaluate_padded_fused_model
from src.validation import run_ensemble, run_stacking
from src.analysis import main as analysis_main


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
CLASSICAL_MODELS = sorted(Config.CLASSICAL_MODELS)
ALL_MODELS = list(Config.DEFAULT_PARAMS.keys())
SCENARIOS = list(Config.SCENARIOS.keys())


def is_classical(model: str) -> bool:
    return model in CLASSICAL_MODELS


def expected_folds(scenario: str) -> int:
    groups_file = Path(Config.DATA_PATH) / Config.SCENARIOS[scenario][0] / "labels" / "groups.npy"
    if groups_file.exists():
        return int(np.unique(np.load(groups_file)).size)
    return Config.N_INDIVIDUALS


def require_args(args, *names):
    missing = [name for name in names if getattr(args, name, None) in (None, "")]
    if missing:
        formatted = ", ".join(f"--{name}" for name in missing)
        raise ValueError(f"Missing required argument(s) for this mode: {formatted}")




def _train_output_dir(args):
    from src.sensor_fusion import scenario_output_name
    return Config.get_output_dir(
        args.model,
        scenario_output_name(
            args.model,
            args.scenario,
            loss=args.loss,
            inner_val_groups=args.inner_val_groups,
            scale=args.scale,
            no_mag=args.no_mag,
            only_mag=args.only_mag,
            sensor_dropout=args.sensor_dropout,
            sensor_dropout_p=args.sensor_dropout_p,
            sensor_dropout_max_off=args.sensor_dropout_max_off,
        ),
    )


def _cross_sensor_output_dirs(args):
    from src.sensor_fusion import scenario_output_name
    train_tag = scenario_output_name(
        args.model,
        args.scenario,
        loss=args.loss,
        inner_val_groups=args.inner_val_groups,
        scale=args.scale,
        no_mag=args.no_mag,
        only_mag=args.only_mag,
    )
    allowed_pairs = {
        "left_T": ["chest_T", "right_T"],
        "right_T": ["chest_T", "left_T"],
        "chest_T": ["left_T", "right_T"],
    }
    targets = []
    for test_scenario in Config.SCENARIOS:
        if test_scenario == args.scenario:
            continue
        if args.scenario in allowed_pairs and test_scenario not in allowed_pairs[args.scenario]:
            continue
        targets.append(Config.get_output_dir(args.model, f"cross_sensor_{train_tag}_to_{test_scenario}"))
    return targets


def _fused_missing_output_dir(args):
    from src.sensor_fusion import scenario_output_name
    train_out = scenario_output_name(
        args.model,
        args.scenario,
        loss=args.loss,
        inner_val_groups=args.inner_val_groups,
        scale=args.scale,
        no_mag=args.no_mag,
        only_mag=args.only_mag,
        sensor_dropout=args.sensor_dropout,
        sensor_dropout_p=args.sensor_dropout_p,
        sensor_dropout_max_off=args.sensor_dropout_max_off,
    )
    return Config.get_output_dir(args.model, f"padded_eval_{train_out}_on_{args.test_scenario}")


def _multisensor_output_dirs(args):
    modes = [args.mode] if args.mode != "all" else ["ensemble", "stacking"]
    return [Config.get_output_dir(args.model, f"multisensor_{mode}_{args.tag}") for mode in modes]

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def run_train(args):
    require_args(args, "model", "scenario")
    output_dir = _train_output_dir(args)
    if Config.is_run_complete(output_dir):
        print(f">> TRAIN already complete at {output_dir} - skipping")
        return
    print(f">> TRAIN | model={args.model} | scenario={args.scenario}")
    train_main(args)


# ------------------------------------------------------------------
# Cross-sensor
# ------------------------------------------------------------------
def run_cross_sensor(args):
    require_args(args, "model", "scenario")
    output_dirs = _cross_sensor_output_dirs(args)
    if output_dirs and all(Config.is_run_complete(path) for path in output_dirs):
        print(">> CROSS SENSOR already complete for all target scenarios - skipping")
        return
    print(f">> CROSS SENSOR | model={args.model} | scenario={args.scenario}")

    run_cross_sensor_eval(
        train_scenario=args.scenario,
        model_type=args.model,
        loss_type=args.loss,
        epochs=args.epochs,
        scale=args.scale,
        no_mag=args.no_mag,
        only_mag=args.only_mag,
        inner_val_groups=args.inner_val_groups,
    )


# ------------------------------------------------------------------
# Fused missing
# ------------------------------------------------------------------
def run_fused_missing(args):
    require_args(args, "model", "scenario", "test_scenario")
    output_dir = _fused_missing_output_dir(args)
    if Config.is_run_complete(output_dir):
        print(f">> FUSED MISSING already complete at {output_dir} - skipping")
        return
    print(f">> FUSED MISSING | model={args.model}")

    evaluate_padded_fused_model(
        model_type=args.model,
        train_scenario=args.scenario,
        test_scenario=args.test_scenario,
        loss=args.loss,
        inner_val_groups=args.inner_val_groups,
        scale=args.scale,
        no_mag=args.no_mag,
        only_mag=args.only_mag,
        sensor_dropout=args.sensor_dropout,
        sensor_dropout_p=args.sensor_dropout_p,
        sensor_dropout_max_off=args.sensor_dropout_max_off,
    )


# ------------------------------------------------------------------
# Multisensor (ensemble / stacking)
# ------------------------------------------------------------------
def run_multisensor(args):
    require_args(args, "model")
    output_dirs = _multisensor_output_dirs(args)
    if output_dirs and all(Config.is_run_complete(path) for path in output_dirs):
        print(f">> MULTISENSOR already complete for mode={args.mode} - skipping")
        return
    print(f">> MULTISENSOR | mode={args.mode} | model={args.model}")

    common_kwargs = dict(
        model=args.model,
        loss=args.loss,
        inner_val_groups=args.inner_val_groups,
        scale=args.scale,
        no_mag=args.no_mag,
        only_mag=args.only_mag,
        threshold=args.threshold,
        tag=args.tag,
    )

    if args.mode == "ensemble":
        run_ensemble(**common_kwargs)
    elif args.mode == "stacking":
        run_stacking(**common_kwargs)
    elif args.mode == "all":
        run_ensemble(**common_kwargs)
        run_stacking(**common_kwargs)


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
def run_analysis(args):
    print(f">> ANALYZE | base_dir={args.base_dir} | output_dir={args.output_dir}")
    analysis_args = Namespace(
        mode="analyze",
        base_dir=args.base_dir,
        output_dir=args.output_dir,
    )
    analysis_main(analysis_args)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser()

    # Core
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--nested", action="store_true")
    parser.add_argument("--cross_sensor", action="store_true")
    parser.add_argument("--fused_missing", action="store_true")
    parser.add_argument("--analyze", action="store_true")

    # Multisensor
    parser.add_argument("--multisensor", action="store_true")
    parser.add_argument("--mode", choices=["ensemble", "stacking", "all"], default="all")
    parser.add_argument("--tag", type=str, default="default")

    # Filters
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--test_scenario", choices=SCENARIOS)

    # Analysis config
    parser.add_argument("--base_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="output/analysis")

    # Training config
    parser.add_argument("--epochs", type=int, default=Config.TRAINING_CONFIG["epochs"])
    parser.add_argument("--loss", type=str, default="weighted")
    parser.add_argument("--inner_val_groups", type=int, default=1)

    # Data flags
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--no_mag", action="store_true")
    parser.add_argument("--only_mag", action="store_true")

    # Sensor dropout
    parser.add_argument("--sensor_dropout", action="store_true")
    parser.add_argument("--sensor_dropout_p", type=float, default=0.5)
    parser.add_argument("--sensor_dropout_max_off", type=int, default=1)

    parser.add_argument("--threshold", type=float, default=0.5)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    selected_actions = [
        args.train,
        args.nested,
        args.cross_sensor,
        args.fused_missing,
        args.multisensor,
        args.analyze,
    ]
    if not any(selected_actions):
        parser.error("select at least one action: --train, --nested, --cross_sensor, --fused_missing, --multisensor, or --analyze")

    # --------------------------------------------------------------
    # Dispatch
    # --------------------------------------------------------------
    if args.train:
        run_train(args)

    if args.cross_sensor:
        run_cross_sensor(args)

    if args.fused_missing:
        run_fused_missing(args)

    if args.multisensor:
        run_multisensor(args)

    if args.nested:
        require_args(args, "model", "scenario")
        print(">> NESTED validation")
        validation_main(args)

    if args.analyze:
        run_analysis(args)


if __name__ == "__main__":
    main()
