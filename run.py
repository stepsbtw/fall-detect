#!/usr/bin/env python3
"""
run.py - Batch runner: model training and evaluation for all models and scenarios.

Usage:
    python run.py [--train] [--nested]
                  [--model <MODEL>] [--scenario <SCENARIO>]
                  [--n_trials <N>] [--epochs <N>] [--inner {kfold,holdout,none}]

Modes (combinable; default is --train):
    --train      Outer LOGO using Config.DEFAULT_PARAMS — no HP search, zero leakage
    --nested     Nested LOGO (outer LOGO over all subjects; inner Optuna per fold — gold standard)

Filters (optional):
    --model <MODEL>        One of: CNN1D MLP LSTM RF SVM XGBoost
    --scenario <SCENARIO>  e.g. chest_T, left_T ...

Evaluation strategies
---------------------
  --train:
    Outer LOGO over all subjects using Config.DEFAULT_PARAMS.
    No HP search — zero leakage by design.
    Each fold trains on N-1 subjects and evaluates on the single left-out subject
    (which also acts as the early-stopping val set).

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

Logs: logs/<model>_<scenario>_{train,nested}.log
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NEURAL_MODELS    = ["CNN1D", "MLP", "LSTM"]
CLASSICAL_MODELS = ["RF", "SVM", "XGBoost"]
ALL_MODELS       = NEURAL_MODELS + CLASSICAL_MODELS

SCENARIOS = [
    "chest_T",
    # "chest_F",
    "left_T",
    # "left_F",
    "right_T",
    # "right_F",
    "chest_left_T",
    # "chest_left_F",
    "chest_right_T",
    # "chest_right_F",
    # "chest_left_right_T",
    # "chest_left_right_F",
]

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR    = SCRIPT_DIR / "logs"
SRC_DIR    = SCRIPT_DIR / "src"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_classical(model: str) -> bool:
    return model in CLASSICAL_MODELS


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
    cmd = [sys.executable, "-u", "pipeline.py", "train",
           "-scenario", scenario,
           "--model", model]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


def run_nested(model: str, scenario: str, n_trials: int, epochs: int, inner: str) -> None:
    log = LOG_DIR / f"{model}_{scenario}_nested.log"
    print(f">>  nested  model={model}  scenario={scenario}  n_trials={n_trials}  inner={inner}")
    cmd = [sys.executable, "-u", "pipeline.py", "nested",
           "-scenario", scenario,
           "--model", model,
           "--n_trials", str(n_trials),
           "--inner", inner]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


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
    parser.add_argument("--model",      metavar="MODEL",    help="One of: " + " ".join(ALL_MODELS))
    parser.add_argument("--scenario",   metavar="SCENARIO", help="e.g. chest_T, left_T ...")
    parser.add_argument("--n_trials",   type=int, default=30,  metavar="N")
    parser.add_argument("--epochs",     type=int, default=200, metavar="N")
    parser.add_argument("--inner",      choices=["kfold", "holdout", "none"], default="kfold",
                        help="Inner CV for --nested: kfold=GroupKFold(k=3), holdout=GroupShuffleSplit(n=1), none=in-sample (default=kfold)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Default: train only
    if not args.train and not args.nested:
        args.train = True

    models    = [args.model]    if args.model    else list(ALL_MODELS)
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)

    total = len(models) * len(scenarios)
    mode_str = " ".join(filter(None, [
        "train"  if args.train  else "",
        "nested" if args.nested else "",
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

    for count, model in enumerate(models, 1):
        for scenario in scenarios:
            print(f"-- [{count}/{total}]  {model} / {scenario} --")

            if args.train:
                run_train(model, scenario, args.epochs)

            if args.nested:
                run_nested(model, scenario, args.n_trials, args.epochs, args.inner)

            print()

    print("=" * 56)
    print(f"  All done ({total} combos).")
    print(f"  Logs saved to: {LOG_DIR}/")
    print("=" * 56)


if __name__ == "__main__":
    main()
