#!/usr/bin/env python3
"""
run.py - Batch runner: model training and evaluation for all models and scenarios.

Usage:
    python run.py [--train] [--nested] [--analyze]
                  [--model <MODEL>] [--scenario <SCENARIO>]
                  [--n_trials <N>] [--epochs <N>] [--inner {kfold,holdout,none}]

Modes (combinable; default is --train):
    --train      Outer LOGO using Config.DEFAULT_PARAMS — no HP search, zero leakage
    --nested     Nested LOGO (outer LOGO over all subjects; inner Optuna per fold — gold standard)
    --analyze    Aggregate per-fold metrics then run global analysis for all completed combos

Filters (optional):
    --model <MODEL>        One of: CNN1D MLP LSTM GRU RF SVM XGBoost CatBoost
    --scenario <SCENARIO>  e.g. chest_T, left_T ...

Evaluation strategies
---------------------
  --train:
    Outer LOGO over all subjects using Config.DEFAULT_PARAMS.
    No HP search — zero leakage by design.
    Each fold holds out one training subject for early-stopping validation,
    trains on the remaining N-2 subjects, and evaluates on the single left-out
    test subject.  The test subject never influences training or stopping.

  --nested:
    Outer LOGO over all subjects.
    For each outer fold, a fresh inner Optuna with --inner CV strategy runs on the
    N-1 remaining subjects to select HPs, then trains on all N-1 and evaluates on
    the single left-out subject.  Zero HP leakage.
    N× more compute than --train.

Examples:
    python run.py                                              # train all (DEFAULT_PARAMS)
    python run.py --nested                                     # nested LOGO, all combos
    python run.py --train --model CNN1D --scenario chest_T     # train one combo
    python run.py --nested --model CNN1D --n_trials 15         # nested, CNN1D, 15 inner trials
    python run.py --nested --inner holdout                     # nested, holdout inner CV
    python run.py --nested --model GRU --n_trials 50 --epochs 300
    python run.py --analyze                                    # aggregate + global analysis
    python run.py --train --analyze                            # train all then aggregate + analyze
    python run.py --analyze --model CNN1D                      # aggregate + analyze for CNN1D only

Logs: logs/<model>_<scenario>_{train,nested}.log
"""

import argparse
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if not SRC_DIR.exists():
    SRC_DIR = SCRIPT_DIR

sys.path.insert(0, str(SRC_DIR))
from config import Config

# Add cross-sensor eval support
from test import run_cross_sensor_eval

# ---------------------------------------------------------------------------
# Defaults — derived from Config so there is a single source of truth
# ---------------------------------------------------------------------------
CLASSICAL_MODELS = sorted(Config.CLASSICAL_MODELS)
NEURAL_MODELS    = [m for m in Config.DEFAULT_PARAMS if m not in Config.CLASSICAL_MODELS]
ALL_MODELS       = NEURAL_MODELS + CLASSICAL_MODELS
SCENARIOS        = list(Config.SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_classical(model: str) -> bool:
    return model in CLASSICAL_MODELS


def _scenario_dir(scenario: str) -> str:
    return Config.SCENARIOS[scenario][0]


def _expected_folds(scenario: str) -> int:
    """Number of unique subjects in this scenario (from groups.npy)."""
    groups_file = SCRIPT_DIR / "dataset" / _scenario_dir(scenario) / "labels" / "groups.npy"
    if groups_file.exists():
        return int(np.unique(np.load(groups_file)).size)
    return 15  # safe fallback


def _train_scenario_out(
    model: str,
    scenario: str,
    loss: str = "weighted",
    inner_val_groups: int = 1,
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
) -> str:
    scenario_out = scenario if loss == "weighted" else scenario + "_NW"
    if not is_classical(model):
        scenario_out = f"{scenario_out}_IVG{max(int(inner_val_groups), 1)}"
    if scale:
        scenario_out = f"{scenario_out}_SC"
    if no_mag:
        scenario_out = f"{scenario_out}_NM"
    if only_mag:
        scenario_out = f"{scenario_out}_OM"
    return scenario_out


def is_train_done(
    model: str,
    scenario: str,
    loss: str = "weighted",
    inner_val_groups: int = 1,
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
    sensor_dropout: bool = False,
    sensor_dropout_p: float = 0.5,
    sensor_dropout_max_off: int = 1,
) -> bool:
    """Return True if every LOGO fold for this combo has a completed metrics file."""
    scenario_out = _train_scenario_out(model, scenario, loss, inner_val_groups, scale, no_mag, only_mag)
    if sensor_dropout:
        scenario_out = f"{scenario_out}_SDP{str(sensor_dropout_p).replace('.', 'p')}_M{int(sensor_dropout_max_off)}"
    output_dir = SCRIPT_DIR / "output" / model / scenario_out
    done = list(output_dir.glob("fold_s*/metrics.csv"))
    return len(done) >= _expected_folds(scenario)


def is_aggregate_done(model: str, scenario_out: str) -> bool:
    """Return True when aggregate outputs already exist for this variant."""
    output_dir = SCRIPT_DIR / "output" / model / scenario_out
    all_metrics = output_dir / "all_metrics.csv"
    summary_standard = output_dir / "summary_metrics_standard.csv"
    summary_confusion = output_dir / "summary_metrics_confusion.csv"
    return (
        all_metrics.exists()
        and summary_standard.exists()
        and summary_confusion.exists()
    )


def run_command(cmd: list[str]) -> None:
    """Run *cmd*, print stdout+stderr to console, and raise on failure."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=SRC_DIR,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_train(
    model: str,
    scenario: str,
    epochs: int,
    loss: str = "weighted",
    inner_val_groups: int = 1,
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
    sensor_dropout: bool = False,
    sensor_dropout_p: float = 0.5,
    sensor_dropout_max_off: int = 1,
    evaluate_missing: bool = False,
) -> None:
    scenario_out = _train_scenario_out(model, scenario, loss, inner_val_groups, scale, no_mag, only_mag)
    if sensor_dropout:
        scenario_out = f"{scenario_out}_SDP{str(sensor_dropout_p).replace('.', 'p')}_M{int(sensor_dropout_max_off)}"
    print(
        f">>  train  model={model}  scenario={scenario}  loss={loss}  "
        f"inner_val_groups={inner_val_groups}  scale={scale}  no_mag={no_mag}  only_mag={only_mag}  "
        f"sensor_dropout={sensor_dropout}  sensor_dropout_p={sensor_dropout_p}  "
        f"sensor_dropout_max_off={sensor_dropout_max_off}  evaluate_missing={evaluate_missing}"
    )
    cmd = [
        sys.executable,
        "-u",
        "training.py",
        "-scenario",
        scenario,
        "--model",
        model,
        "--loss",
        loss,
        "--inner-val-groups",
        str(inner_val_groups),
    ]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    if scale:
        cmd += ["--scale"]
    if no_mag:
        cmd += ["--no-mag"]
    if only_mag:
        cmd += ["--only-mag"]
    if sensor_dropout:
        cmd += ["--sensor-dropout", "--sensor-dropout-p", str(sensor_dropout_p), "--sensor-dropout-max-off", str(sensor_dropout_max_off)]
    if evaluate_missing:
        cmd += ["--evaluate-missing"]
    run_command(cmd)
    print(f"    done.")


def run_nested(
    model: str,
    scenario: str,
    n_trials: int,
    epochs: int,
    inner: str,
    loss: str = "weighted",
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
) -> None:
    print(
        f">>  nested  model={model}  scenario={scenario}  n_trials={n_trials}  "
        f"inner={inner}  loss={loss}  scale={scale}  no_mag={no_mag}  only_mag={only_mag}"
    )
    cmd = [
        sys.executable,
        "-u",
        "validation.py",
        "-scenario",
        scenario,
        "--model",
        model,
        "--n_trials",
        str(n_trials),
        "--inner",
        inner,
        "--loss",
        loss,
    ]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    if scale:
        cmd += ["--scale"]
    if no_mag:
        cmd += ["--no-mag"]
    if only_mag:
        cmd += ["--only-mag"]
    run_command(cmd)
    print(f"    done.")


def run_ensemble(
    model: str,
    loss: str = "weighted",
    inner_val_groups: int = 1,
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
    tag: str = "default",
) -> None:
    print(
        f">>  ensemble  model={model}  loss={loss}  inner_val_groups={inner_val_groups}  "
        f"scale={scale}  no_mag={no_mag}  only_mag={only_mag}  tag={tag}"
    )
    cmd = [
        sys.executable, "-u", "multisensor.py", "ensemble",
        "--model", model,
        "--loss", loss,
        "--inner-val-groups", str(inner_val_groups),
        "--tag", tag,
    ]
    if scale:
        cmd += ["--scale"]
    if no_mag:
        cmd += ["--no-mag"]
    if only_mag:
        cmd += ["--only-mag"]
    run_command(cmd)
    print("    done.")


def run_stacking(
    model: str,
    loss: str = "weighted",
    inner_val_groups: int = 1,
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
    tag: str = "default",
) -> None:
    print(
        f">>  stacking  model={model}  loss={loss}  inner_val_groups={inner_val_groups}  "
        f"scale={scale}  no_mag={no_mag}  only_mag={only_mag}  tag={tag}"
    )
    cmd = [
        sys.executable, "-u", "multisensor.py", "stacking",
        "--model", model,
        "--loss", loss,
        "--inner-val-groups", str(inner_val_groups),
        "--tag", tag,
    ]
    if scale:
        cmd += ["--scale"]
    if no_mag:
        cmd += ["--no-mag"]
    if only_mag:
        cmd += ["--only-mag"]
    run_command(cmd)
    print("    done.")


def run_fused_missing_eval(
    model: str,
    train_scenario: str,
    test_scenario: str,
    loss: str = "weighted",
    inner_val_groups: int = 1,
    scale: bool = False,
    no_mag: bool = False,
    only_mag: bool = False,
    sensor_dropout: bool = False,
    sensor_dropout_p: float = 0.5,
    sensor_dropout_max_off: int = 1,
) -> None:
    print(
        f">>  fused_missing_eval  model={model}  train={train_scenario}  test={test_scenario}  "
        f"loss={loss}  inner_val_groups={inner_val_groups}  scale={scale}  no_mag={no_mag}  "
        f"only_mag={only_mag}  sensor_dropout={sensor_dropout}"
    )
    cmd = [
        sys.executable, "-u", "fused_missing_eval.py",
        "--model", model,
        "--train-scenario", train_scenario,
        "--test-scenario", test_scenario,
        "--loss", loss,
        "--inner-val-groups", str(inner_val_groups),
    ]
    if scale:
        cmd += ["--scale"]
    if no_mag:
        cmd += ["--no-mag"]
    if only_mag:
        cmd += ["--only-mag"]
    if sensor_dropout:
        cmd += ["--sensor-dropout", "--sensor-dropout-p", str(sensor_dropout_p), "--sensor-dropout-max-off", str(sensor_dropout_max_off)]
    run_command(cmd)
    print("    done.")


def run_aggregate(model: str, scenario_out: str) -> None:
    print(f">>  aggregate  model={model}  scenario_out={scenario_out}")
    cmd = [sys.executable, "-u", "analysis.py", "aggregate",
           "-scenario", scenario_out,
           "--model", model]
    run_command(cmd)
    print(f"    done.")


def run_analyze(output_dir: str = "output/analysis") -> None:
    print(f">>  analyze  output_dir={output_dir}")
    cmd = [sys.executable, "-u", "analysis.py", "analyze",
           "--base_dir", str(SCRIPT_DIR / "output"),
           "--output_dir", str(SCRIPT_DIR / output_dir)]
    run_command(cmd)
    print(f"    done.")


def is_global_analysis_done(output_dir: str = "output/analysis") -> bool:
    """Return True when global analysis artifacts already exist."""
    analysis_root = SCRIPT_DIR / output_dir
    return (analysis_root / "csv" / "summary_final_models.csv").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fall-Detect batch runner (Optuna search and/or final training).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cross_sensor", action="store_true", help="Run cross-sensor evaluation after training")
    parser.add_argument("--train_scenario", type=str, help="Training scenario for cross-sensor eval (deprecated; --scenario also works)")
    parser.add_argument("--train",      action="store_true", help="Outer LOGO eval with default HPs — no search, zero leakage")
    parser.add_argument("--nested",     action="store_true", help="Nested LOGO (outer LOGO / inner Optuna per fold)")
    parser.add_argument("--analyze",    action="store_true", help="Aggregate per-fold metrics then run global analysis for all completed combos")
    parser.add_argument("--model",      metavar="MODEL",    help="One of: " + " ".join(ALL_MODELS))
    parser.add_argument("--scenario",   metavar="SCENARIO", help="e.g. chest_T, left_T ...")
    parser.add_argument("--n_trials",   type=int, default=Config.OPTUNA_CONFIG["n_trials"],    metavar="N")
    parser.add_argument("--epochs",     type=int, default=Config.TRAINING_CONFIG["epochs"],    metavar="N")
    parser.add_argument("--inner",      choices=["kfold", "holdout", "none"], default="kfold",
                        help="Inner CV for --nested: kfold=GroupKFold(k=3), holdout=GroupShuffleSplit(n=1), none=in-sample (default=kfold)")
    parser.add_argument(
        "--loss",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Loss weighting for neural models: 'weighted' uses inverse-frequency class weights (default); "
             "'unweighted' uses plain CrossEntropyLoss. Results are saved to <scenario>_NW dirs.",
    )
    parser.add_argument(
        "--inner-val-groups",
        dest="inner_val_groups",
        type=int,
        default=1,
        metavar="N",
        help="Group-wise inner validation size for --train (held-out training subjects per outer fold, default=1).",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Fit a StandardScaler on the training split of each LOGO fold and apply it "
             "to validation and test. Scaled runs are saved to <scenario>_SC directories.",
    )
    parser.add_argument(
        "--no-mag",
        dest="no_mag",
        action="store_true",
        default=False,
        help="Drop the engineered magnitude channels (mag_acc, mag_gyr) from every sensor "
             "block before training. Results are saved to <scenario>_NM directories.",
    )
    parser.add_argument(
        "--only-mag",
        dest="only_mag",
        action="store_true",
        default=False,
        help="Keep only the engineered magnitude channels (mag_acc, mag_gyr), dropping raw axes. "
             "Results are saved to <scenario>_OM directories.",
    )
    parser.add_argument("--ensemble", action="store_true", help="Run multisensor late-fusion ensemble from saved fold predictions")
    parser.add_argument("--stacking", action="store_true", help="Run multisensor stacking from saved fold predictions")
    parser.add_argument("--fused_missing_eval", action="store_true", help="Evaluate a fused model on a smaller scenario by zero-padding missing sensors")
    parser.add_argument("--test_scenario", type=str, help="Target scenario for --fused_missing_eval")
    parser.add_argument("--sensor-dropout", action="store_true", default=False, help="Enable structured sensor dropout during training")
    parser.add_argument("--sensor-dropout-p", type=float, default=0.5, metavar="P", help="Probability of masking one or more sensor blocks during training")
    parser.add_argument("--sensor-dropout-max-off", type=int, default=1, metavar="N", help="Maximum number of sensor blocks to mask during training")
    parser.add_argument("--evaluate-missing", action="store_true", default=False, help="After each fold, also evaluate deterministic missing-sensor conditions")
    parser.add_argument("--tag", default="default", help="Optional suffix for ensemble/stacking outputs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.cross_sensor:
        train_scenario = args.train_scenario or args.scenario
        if not (train_scenario and args.model):
            print("--model and one of --scenario/--train_scenario are required for cross-sensor eval.")
            return
        run_cross_sensor_eval(
            train_scenario,
            args.model,
            loss_type=args.loss,
            epochs=args.epochs,
            scale=args.scale,
            no_mag=args.no_mag,
            only_mag=args.only_mag,
        )
        return

    if args.ensemble:
        if not args.model:
            print("--model is required for --ensemble.")
            return
        run_ensemble(
            model=args.model,
            loss=args.loss,
            inner_val_groups=args.inner_val_groups,
            scale=args.scale,
            no_mag=args.no_mag,
            only_mag=args.only_mag,
            tag=args.tag,
        )
        return

    if args.stacking:
        if not args.model:
            print("--model is required for --stacking.")
            return
        run_stacking(
            model=args.model,
            loss=args.loss,
            inner_val_groups=args.inner_val_groups,
            scale=args.scale,
            no_mag=args.no_mag,
            only_mag=args.only_mag,
            tag=args.tag,
        )
        return

    if args.fused_missing_eval:
        train_scenario = args.train_scenario or args.scenario
        if not (args.model and train_scenario and args.test_scenario):
            print("--model, one of --scenario/--train_scenario, and --test_scenario are required for --fused_missing_eval.")
            return
        run_fused_missing_eval(
            model=args.model,
            train_scenario=train_scenario,
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
        return

    # Default: train only (unless --analyze is the sole flag)
    if not args.train and not args.nested and not args.analyze:
        args.train = True

    models    = [args.model]    if args.model    else list(ALL_MODELS)
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)

    total = len(models) * len(scenarios)
    mode_str = " ".join(filter(None, [
        "train"   if args.train   else "",
        "nested"  if args.nested  else "",
        "analyze" if args.analyze else "",
    ]))

    print("=" * 56)
    print("  Fall-Detect -- batch runner")
    print(f"  Mode     : {mode_str}")
    print(f"  Models   : {' '.join(models)}")
    print(f"  Scenarios: {' '.join(scenarios)}")
    print(f"  Combos   : {total}")
    if args.train:
        print(f"  epochs    : {args.epochs} (NN only)")
        print(f"  loss      : {args.loss} (NN only)")
        print(f"  inner val : {args.inner_val_groups} groups (NN only)")
        print(f"  scale     : {args.scale}")
        print(f"  no_mag    : {args.no_mag}")
        print(f"  only_mag  : {args.only_mag}")
        print(f"  sensor_dropout        : {args.sensor_dropout}")
        print(f"  sensor_dropout_p      : {args.sensor_dropout_p}")
        print(f"  sensor_dropout_max_off: {args.sensor_dropout_max_off}")
        print(f"  evaluate_missing      : {args.evaluate_missing}")
    if args.nested:
        print(f"  n_trials  : {args.n_trials} (inner, per outer fold) | epochs: {args.epochs} (NN only)")
        print(f"  inner     : {args.inner}")
        print(f"  loss      : {args.loss} (NN only)")
        print(f"  scale     : {args.scale}")
        print(f"  no_mag    : {args.no_mag}")
        print(f"  only_mag  : {args.only_mag}")
    print("=" * 56)
    print()

    # --analyze only: skip the training loop entirely
    if args.analyze and not args.train and not args.nested:
        _run_full_analysis(models, scenarios)
        return

    count = 0
    for model in models:
        for scenario in scenarios:
            count += 1
            print(f"-- [{count}/{total}]  {model} / {scenario} --")

            if args.train:
                if is_train_done(model, scenario, args.loss, args.inner_val_groups, args.scale, args.no_mag, args.only_mag, args.sensor_dropout, args.sensor_dropout_p, args.sensor_dropout_max_off):
                    print(f"   [skip] train {model}/{scenario} — all folds already done.")
                else:
                    run_train(
                        model,
                        scenario,
                        args.epochs,
                        args.loss,
                        args.inner_val_groups,
                        args.scale,
                        args.no_mag,
                        args.only_mag,
                        args.sensor_dropout,
                        args.sensor_dropout_p,
                        args.sensor_dropout_max_off,
                        args.evaluate_missing,
                    )

            if args.nested:
                run_nested(
                    model,
                    scenario,
                    args.n_trials,
                    args.epochs,
                    args.inner,
                    args.loss,
                    args.scale,
                    args.no_mag,
                    args.only_mag,
                )

            print()

    if args.analyze:
        _run_full_analysis(models, scenarios)

    print("=" * 56)
    print(f"  All done ({total} combos).")
    print("=" * 56)


def _discover_variant_dirs(models: list, scenarios: list) -> list[tuple[str, str]]:
    """Return (model, scenario_out) for every variant directory that has fold data.

    A variant directory matches when its name equals a base scenario name or starts
    with '<base_scenario>_' (e.g. chest_T, chest_T_IVG1, chest_T_IVG1_SC_NM).
    """
    combos = []
    for model in models:
        model_dir = SCRIPT_DIR / "output" / model
        if not model_dir.exists():
            continue
        for variant_dir in sorted(model_dir.iterdir()):
            if not variant_dir.is_dir() or variant_dir.name == "analysis":
                continue
            if not any(
                variant_dir.name == s or variant_dir.name.startswith(s + "_")
                for s in scenarios
            ):
                continue
            if list(variant_dir.glob("fold_s*/metrics.csv")):
                combos.append((model, variant_dir.name))
    return combos


def _run_full_analysis(models: list, scenarios: list) -> None:
    """Aggregate per-fold metrics for every completed combo, then run global analysis."""
    print()
    print("=" * 56)
    print("  Aggregate + Analyze phase")
    print("=" * 56)

    aggregated = 0
    skipped = 0
    for model, scenario_out in _discover_variant_dirs(models, scenarios):
        if is_aggregate_done(model, scenario_out):
            print(f"   [skip] aggregate {model}/{scenario_out} — metrics already aggregated.")
            skipped += 1
            continue
        try:
            run_aggregate(model, scenario_out)
            aggregated += 1
        except subprocess.CalledProcessError as exc:
            print(f"   [error] aggregate {model}/{scenario_out} failed (exit {exc.returncode}), continuing.")

    print()
    print(f"  Aggregate complete — {aggregated} combos processed, {skipped} skipped.")
    print()

    if is_global_analysis_done("output/analysis"):
        print("   [skip] global analyze — analysis output already exists (output/analysis/).")
        return

    try:
        run_analyze()
    except subprocess.CalledProcessError as exc:
        print(f"   [error] global analyze failed (exit {exc.returncode}).")


if __name__ == "__main__":
    main()
