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
    --model <MODEL>        One of: CNN1D MLP LSTM RF SVM XGBoost CatBoost
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
    python run.py --nested --model LSTM --n_trials 50 --epochs 300
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
LOG_DIR    = SCRIPT_DIR / "logs"
SRC_DIR    = SCRIPT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
from config import Config

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
    """Map 'chest_T' -> 'chest', 'chest_left_T' -> 'chest_left', etc."""
    return scenario.rsplit("_", 1)[0]


def _expected_folds(scenario: str) -> int:
    """Number of unique subjects in this scenario (from groups.npy)."""
    groups_file = SCRIPT_DIR / "dataset" / _scenario_dir(scenario) / "labels" / "groups.npy"
    if groups_file.exists():
        return int(np.unique(np.load(groups_file)).size)
    return 15  # safe fallback


def is_train_done(model: str, scenario: str) -> bool:
    """Return True if every LOGO fold for this combo has a completed metrics file."""
    output_dir = SCRIPT_DIR / "output" / model / scenario
    done = list(output_dir.glob("fold_s*/metrics_model_s*.csv"))
    return len(done) >= _expected_folds(scenario)


def is_aggregate_done(model: str, scenario: str) -> bool:
    """Return True when aggregate outputs already exist for this combo."""
    output_dir = SCRIPT_DIR / "output" / model / scenario
    all_metrics = output_dir / "all_metrics.csv"
    summary_metrics = output_dir / "summary_metrics.csv"
    return all_metrics.exists() and summary_metrics.exists()


def run_command(cmd: list[str], log_path: Path) -> None:
    """Run *cmd*, tee stdout+stderr to *log_path*, and raise on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
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
            log_fh.write(line)
        process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_train(model: str, scenario: str, epochs: int) -> None:
    log = LOG_DIR / f"{model}_{scenario}_train.log"
    print(f">>  train  model={model}  scenario={scenario}")
    cmd = [
        sys.executable,
        "-u",
        "training.py",
        "-scenario",
        scenario,
        "--model",
        model,
    ]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


def run_nested(model: str, scenario: str, n_trials: int, epochs: int, inner: str) -> None:
    log = LOG_DIR / f"{model}_{scenario}_nested.log"
    print(f">>  nested  model={model}  scenario={scenario}  n_trials={n_trials}  inner={inner}")
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
    ]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


def run_aggregate(model: str, scenario: str) -> None:
    log = LOG_DIR / f"{model}_{scenario}_aggregate.log"
    print(f">>  aggregate  model={model}  scenario={scenario}")
    cmd = [sys.executable, "-u", "analysis.py", "aggregate",
           "-scenario", scenario,
           "--model", model]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


def run_analyze(output_dir: str = "output/analysis") -> None:
    log = LOG_DIR / "analyze_global.log"
    print(f">>  analyze  output_dir={output_dir}")
    cmd = [sys.executable, "-u", "analysis.py", "analyze",
           "--base_dir", "../output",
           "--output_dir", f"../{output_dir}"]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


def is_global_analysis_done(output_dir: str = "output/analysis") -> bool:
    """Return True when global analysis artifacts already exist."""
    analysis_root = SCRIPT_DIR / output_dir
    return (analysis_root / "summary_final_models.csv").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fall-Detect batch runner (Optuna search and/or final training).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",      action="store_true", help="Outer LOGO eval with default HPs — no search, zero leakage")
    parser.add_argument("--nested",     action="store_true", help="Nested LOGO (outer LOGO / inner Optuna per fold)")
    parser.add_argument("--analyze",    action="store_true", help="Aggregate per-fold metrics then run global analysis for all completed combos")
    parser.add_argument("--model",      metavar="MODEL",    help="One of: " + " ".join(ALL_MODELS))
    parser.add_argument("--scenario",   metavar="SCENARIO", help="e.g. chest_T, left_T ...")
    parser.add_argument("--n_trials",   type=int, default=Config.OPTUNA_CONFIG["n_trials"],    metavar="N")
    parser.add_argument("--epochs",     type=int, default=Config.TRAINING_CONFIG["epochs"],    metavar="N")
    parser.add_argument("--inner",      choices=["kfold", "holdout", "none"], default="kfold",
                        help="Inner CV for --nested: kfold=GroupKFold(k=3), holdout=GroupShuffleSplit(n=1), none=in-sample (default=kfold)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

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
    if args.nested:
        print(f"  n_trials  : {args.n_trials} (inner, per outer fold) | epochs: {args.epochs} (NN only)")
        print(f"  inner     : {args.inner}")
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
                if is_train_done(model, scenario):
                    print(f"   [skip] train {model}/{scenario} — all folds already done.")
                else:
                    run_train(model, scenario, args.epochs)

            if args.nested:
                run_nested(model, scenario, args.n_trials, args.epochs, args.inner)

            print()

    if args.analyze:
        _run_full_analysis(models, scenarios)

    print("=" * 56)
    print(f"  All done ({total} combos).")
    print(f"  Logs saved to: {LOG_DIR}/")
    print("=" * 56)


def _run_full_analysis(models: list, scenarios: list) -> None:
    """Aggregate per-fold metrics for every completed combo, then run global analysis."""
    print()
    print("=" * 56)
    print("  Aggregate + Analyze phase")
    print("=" * 56)

    aggregated = 0
    skipped = 0
    for model in models:
        for scenario in scenarios:
            output_dir = SCRIPT_DIR / "output" / model / scenario
            fold_csvs = list(output_dir.glob("fold_s*/metrics_model_s*.csv"))
            if not fold_csvs:
                print(f"   [skip] aggregate {model}/{scenario} — no fold data found.")
                skipped += 1
                continue
            if is_aggregate_done(model, scenario):
                print(f"   [skip] aggregate {model}/{scenario} — metrics already aggregated.")
                skipped += 1
                continue
            try:
                run_aggregate(model, scenario)
                aggregated += 1
            except subprocess.CalledProcessError as exc:
                print(f"   [error] aggregate {model}/{scenario} failed (exit {exc.returncode}), continuing.")

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
