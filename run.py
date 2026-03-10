#!/usr/bin/env python3
"""
run.py - Batch runner: Optuna search and/or model evaluation for all models and scenarios.

Usage:
    python run.py [--search] [--train] [--nested]
                  [--model <MODEL>] [--scenario <SCENARIO>]
                  [--n_trials <N>] [--num_models <N>] [--epochs <N>]

Modes (combinable; default is --train only):
    --search     Run Optuna hyperparameter search (LOGO over 12 trainval subjects)
    --train      LOGO evaluation using fixed 3-subject holdout
    --nested     Nested LOGO (outer LOGO over 15; inner Optuna per fold — gold standard)

Filters (optional):
    --model <MODEL>        One of: CNN1D MLP LSTM RF SVM XGBoost
    --scenario <SCENARIO>  e.g. chest_T, left_T ...

Evaluation strategies
---------------------
  --train:
    Requires a prior --search run.
    Locks 3 subjects away as an untouched test set.
    Optuna searches over LOGO on the remaining 12.
    Each LOGO fold (11 train / 1 val for early stopping) is evaluated on those fixed 3.
    Reports mean ± std across 12 folds.

  --nested:
    Self-contained; no prior --search needed.
    Outer LOGO over all 15 subjects.
    For each outer fold, a fresh inner Optuna runs on the 14 remaining subjects.
    Evaluates on the single left-out subject.
    Reports mean ± std across 15 folds.
    15× more compute than --train, but zero HP leakage and uses all subjects.

Examples:
    python run.py                                              # train all (DEFAULT_PARAMS)
    python run.py --search --train                             # search then train, all combos
    python run.py --nested                                     # nested LOGO, all combos
    python run.py --search --model RF                          # search only, RF, all scenarios
    python run.py --train --model CNN1D --scenario chest_T     # train one combo
    python run.py --nested --model CNN1D --n_trials 15         # nested, CNN1D, 15 inner trials
    python run.py --search --train --model LSTM --n_trials 50 --epochs 300

Logs: logs/<model>_<scenario>_{search,train,nested}.log
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


def run_search(model: str, scenario: str, n_trials: int) -> None:
    log = LOG_DIR / f"{model}_{scenario}_search.log"
    print(f">>  search  model={model}  scenario={scenario}  n_trials={n_trials}")
    run_command(
        [sys.executable, "-u", "pipeline.py", "search",
         "-scenario", scenario,
         "--model", model,
         "--n_trials", str(n_trials)],
        log,
    )
    print(f"    done  (log: {log})")


def run_train(model: str, scenario: str, num_models: int, epochs: int) -> None:
    log = LOG_DIR / f"{model}_{scenario}_train.log"
    print(f">>  train  model={model}  scenario={scenario}  num_models={num_models}")
    cmd = [sys.executable, "-u", "pipeline.py", "train",
           "-scenario", scenario,
           "--model", model,
           "--num_models", str(num_models)]
    if not is_classical(model):
        cmd += ["--epochs", str(epochs)]
    run_command(cmd, log)
    print(f"    done  (log: {log})")


def run_nested(model: str, scenario: str, n_trials: int, epochs: int) -> None:
    log = LOG_DIR / f"{model}_{scenario}_nested.log"
    print(f">>  nested  model={model}  scenario={scenario}  n_trials={n_trials}")
    cmd = [sys.executable, "-u", "pipeline.py", "nested",
           "-scenario", scenario,
           "--model", model,
           "--n_trials", str(n_trials)]
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
    parser.add_argument("--search",     action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--train",      action="store_true", help="Run LOGO eval with fixed 3-subject holdout")
    parser.add_argument("--nested",     action="store_true", help="Run nested LOGO (outer 15 / inner Optuna per fold)")
    parser.add_argument("--model",      metavar="MODEL",    help="One of: " + " ".join(ALL_MODELS))
    parser.add_argument("--scenario",   metavar="SCENARIO", help="e.g. chest_T, left_T ...")
    parser.add_argument("--n_trials",   type=int, default=30,  metavar="N")
    parser.add_argument("--num_models", type=int, default=1,   metavar="N",
                        help="LOGO repetitions for --train (default=1)")
    parser.add_argument("--epochs",     type=int, default=200, metavar="N")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Default: train only
    if not args.search and not args.train and not args.nested:
        args.train = True

    models    = [args.model]    if args.model    else list(ALL_MODELS)
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)

    total = len(models) * len(scenarios)
    mode_str = " ".join(filter(None, [
        "search" if args.search else "",
        "train"  if args.train  else "",
        "nested" if args.nested else "",
    ]))

    print("=" * 56)
    print("  Fall-Detect -- batch runner")
    print(f"  Mode     : {mode_str}")
    print(f"  Models   : {' '.join(models)}")
    print(f"  Scenarios: {' '.join(scenarios)}")
    print(f"  Combos   : {total}")
    if args.search:
        print(f"  n_trials  : {args.n_trials}")
    if args.train:
        print(f"  num_models: {args.num_models} LOGO reps | epochs: {args.epochs} (NN only)")
    if args.nested:
        print(f"  n_trials  : {args.n_trials} (inner, per outer fold) | epochs: {args.epochs} (NN only)")
    print("=" * 56)
    print()

    for count, model in enumerate(models, 1):
        for scenario in scenarios:
            print(f"-- [{count}/{total}]  {model} / {scenario} --")

            if args.search:
                run_search(model, scenario, args.n_trials)

            if args.train:
                run_train(model, scenario, args.num_models, args.epochs)

            if args.nested:
                run_nested(model, scenario, args.n_trials, args.epochs)

            print()

    print("=" * 56)
    print(f"  All done ({total} combos).")
    print(f"  Logs saved to: {LOG_DIR}/")
    print("=" * 56)


if __name__ == "__main__":
    main()
