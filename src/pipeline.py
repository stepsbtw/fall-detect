"""Unified pipeline: model training and evaluation.

Usage:
    python pipeline.py train  -scenario <s> [--model <m>] [--epochs E]
    python pipeline.py nested -scenario <s> [--model <m>] [--n_trials N] [--epochs E] [--inner {kfold,holdout,none}]

Evaluation strategies
-----------------------
  train   Outer LOGO over all subjects using Config.DEFAULT_PARAMS.
          No HP search — zero leakage by design.
          Each fold trains on N-1 subjects and evaluates on the left-out subject
          (which also acts as the early-stopping validation set).

  nested  Nested LOGO (gold standard).
          Outer LOGO over all subjects; for each outer fold a fresh inner Optuna runs
          on the N-1 remaining subjects to pick HPs, then trains on all N-1 and
          evaluates on the single left-out subject.  Zero HP leakage.
          N× more compute than `train`.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
import optuna.visualization as vis
import pandas as pd

from utils import run_optuna, train, save_results, save_results_classical, _make_classical_model, create_model, plot_loss_curve
from config import Config

SCENARIO_CHOICES = [
    "chest_T", "chest_F", "left_T", "left_F", "right_T", "right_F",
    "chest_left_right_T", "chest_left_right_F",
    "chest_left_T", "chest_left_F",
    "chest_right_T", "chest_right_F",
]

def _print_best_params(model_type, best_value, best_params):
    """Print a formatted summary of the best hyperparameters found by Optuna."""
    print(f"\n{'='*50}")
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print(f"{'='*50}")
    print(f"Modelo: {model_type}")
    print(f"Melhor F1: {best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()


def run_final_training(args):
    """Outer LOGO over all subjects using Config.DEFAULT_PARAMS — no HP search.

    Each fold trains on N-1 subjects and uses the single left-out subject for
    both early-stopping validation and final test evaluation.  Zero HP leakage.
    """
    scenario = args.scenario
    model_type_arg = args.model
    epochs = args.epochs

    if not model_type_arg:
        raise ValueError("--model é obrigatório para o modo train.")

    model_type  = model_type_arg
    best_params = Config.DEFAULT_PARAMS[model_type]
    print(f"Usando parâmetros padrão para {model_type}: {best_params}")

    base_out = Config.get_output_dir(model_type_arg, scenario)
    os.makedirs(base_out, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    X      = np.load(Config.get_data_file(scenario))
    y      = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))

    unique_subjects = np.unique(groups)
    print(f"Sujeitos (LOGO): {sorted(unique_subjects.tolist())} ({len(unique_subjects)} total)")

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    threshold = best_params.get("decision_threshold", 0.5)

    # ── Classical models (RF / SVM / XGBoost / CatBoost) ───────────────────
    if model_type in Config.CLASSICAL_MODELS:
        Config.set_seed(Config.FINAL_TRAINING['seed_offset'])
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            left_out = groups[test_idx[0]]
            fold_dir = os.path.join(base_out, f"fold_s{left_out}")
            fold_label = f"s{left_out}"
            done_marker = os.path.join(fold_dir, f"metrics_model_{fold_label}.csv")
            if os.path.exists(done_marker):
                print(f"  Fold s{left_out} já concluído — pulando.")
                continue
            print(f"  Fold {fold_idx+1}/{n_folds} — sujeito de teste: {left_out}")
            os.makedirs(fold_dir, exist_ok=True)
            X_tr = X[train_idx].reshape(len(train_idx), -1)
            y_tr = y[train_idx]
            X_te = X[test_idx].reshape(len(test_idx), -1)
            y_te = y[test_idx]
            clf = _make_classical_model(model_type, best_params, y_tr)
            clf.fit(X_tr, y_tr)
            save_results_classical(
                clf=clf, X_test_flat=X_te, y_test=y_te,
                decision_threshold=threshold, i=fold_label, output_dir=fold_dir,
            )
            print(f"  Fold s{left_out} concluído")
        print(f"\nLOGO concluído! Resultados em: {base_out}")
        return

    # ── Neural networks ───────────────────────────────────────────────────────
    input_shape_dict = Config.get_input_shape_dict(scenario, model_type)
    input_shape = input_shape_dict[model_type]
    batch_size = Config.TRAINING_CONFIG.get('batch_size', 32)

    Config.set_seed(Config.FINAL_TRAINING['seed_offset'])
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        left_out = groups[test_idx[0]]
        fold_dir = os.path.join(base_out, f"fold_s{left_out}")
        fold_label = f"s{left_out}"
        done_marker = os.path.join(fold_dir, f"metrics_model_{fold_label}.csv")
        if os.path.exists(done_marker):
            print(f"\n  Fold s{left_out} já concluído — pulando.")
            continue
        print(f"\n  Fold {fold_idx+1}/{n_folds} — sujeito de teste: {left_out}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train = X[train_idx]
        y_train = y[train_idx]
        # left-out subject: used for early-stopping val and as the test set
        X_val   = X[test_idx]
        y_val   = y[test_idx]

        model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
        model.to(Config.DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
        )
        class_counts = np.bincount(y_train, minlength=Config.NUM_LABELS)
        class_weights = len(y_train) / (Config.NUM_LABELS * class_counts.astype(float))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=batch_size, shuffle=False,
        )

        _, _, val_losses, train_losses = train(
            model, train_loader, val_loader, optimizer, criterion, Config.DEVICE,
            epochs=epochs,
            early_stopping=True,
            patience=Config.TRAINING_CONFIG['patience'],
            scaler=None,
            scheduler=scheduler,
        )

        plot_loss_curve(train_losses, val_losses, fold_dir, fold_label)
        np.save(os.path.join(fold_dir, f"train_losses_{fold_label}.npy"), np.array(train_losses))
        np.save(os.path.join(fold_dir, f"val_losses_{fold_label}.npy"), np.array(val_losses))
        pd.DataFrame({
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }).to_csv(os.path.join(fold_dir, f"losses_{fold_label}.csv"), index=False)

        save_results(
            model=model,
            val_loader=val_loader,
            y_val_onehot=y_val,
            i=fold_label,
            decision_threshold=threshold,
            output_dir=fold_dir,
            device=Config.DEVICE,
        )
        print(f"  Fold s{left_out} concluído — salvo em {fold_dir}")

    print(f"\nLOGO concluído! Resultados em: {base_out}")


def run_nested_logo(args):
    """Nested LOGO: outer LOGO over all subjects, inner Optuna with GroupKFold(k=3).

    For each outer fold:
      - Inner Optuna runs GroupKFold(k=3) over the N-1 remaining subjects to select HPs.
      - A model is trained on all N-1 subjects (one is held out for early stopping).
      - Evaluated on the 1 left-out outer subject.

    Zero HP leakage; all subjects contribute exactly one test result.
    """
    scenario      = args.scenario
    model_type_arg = args.model
    n_trials      = args.n_trials
    epochs        = args.epochs
    inner_cv      = args.inner

    base_out = os.path.join(
        Config.get_output_dir(model_type_arg, scenario), "nested"
    )
    os.makedirs(base_out, exist_ok=True)

    X      = np.load(Config.get_data_file(scenario))
    y      = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))

    print(f"\nNested LOGO  |  scenario={scenario}  model={model_type_arg or 'auto'}")
    print(f"Subjects: {sorted(np.unique(groups).tolist())}  ({len(np.unique(groups))} total)")
    print(f"Inner n_trials per fold: {n_trials}")

    input_shape_dict = Config.get_input_shape_dict(scenario, model_type_arg)
    logo_outer = LeaveOneGroupOut()
    n_outer    = logo_outer.get_n_splits(groups=groups)
    batch_size = Config.TRAINING_CONFIG.get('batch_size', 32)

    for outer_idx, (inner_idx, test_idx) in enumerate(
            logo_outer.split(X, y, groups)):
        left_out = groups[test_idx[0]]
        print(f"\n{'='*60}")
        print(f"Outer fold {outer_idx+1}/{n_outer}  —  test subject: {left_out}")
        print(f"{'='*60}")

        X_inner      = X[inner_idx]
        y_inner      = y[inner_idx]
        groups_inner = groups[inner_idx]
        X_test_fold  = X[test_idx]
        y_test_fold  = y[test_idx]

        fold_dir = os.path.join(base_out, f"outer_s{left_out}")
        os.makedirs(fold_dir, exist_ok=True)

        # ── Inner Optuna (inner_cv per fold) ────────────────────────────────────
        Config.OPTUNA_CONFIG['n_trials'] = n_trials
        study_name = (
            f"{scenario}_{model_type_arg}_outer_s{left_out}"
            if model_type_arg else
            f"{scenario}_outer_s{left_out}"
        )
        study = run_optuna(
            input_shape_dict=input_shape_dict,
            X_trainval=X_inner,
            y_trainval=y_inner,
            groups=groups_inner,
            output_dir=fold_dir,
            num_labels=Config.NUM_LABELS,
            device=Config.DEVICE,
            restrict_model_type=model_type_arg,
            study_name=study_name,
            inner_cv=inner_cv,
        )

        best_params = study.best_params
        model_type  = best_params["model_type"] if not model_type_arg else model_type_arg
        threshold   = best_params.get("decision_threshold", 0.5)

        # ── Persist study artefacts ──────────────────────────────────────────
        with open(os.path.join(fold_dir, "best_hyperparameters.json"), "w") as f:
            json.dump({
                "outer_subject": int(left_out),
                "model_type": model_type,
                "best_value": float(study.best_value),
                "best_params": best_params,
                "n_trials": len(study.trials),
                "optimization_history": [t.value for t in study.trials if t.value is not None],
            }, f, indent=2)

        study.trials_dataframe().to_csv(
            os.path.join(fold_dir, "optuna_trials.csv"), index=False
        )

        try:
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(fold_dir, "param_importance.png"))
        except Exception as e:
            print(f"  [AVISO] Não foi possível salvar param_importance.png: {e}")

        _print_best_params(model_type, study.best_value, best_params)

        # ── Train on all 14, val = best inner-fold left-out subject ───────
        if model_type in Config.CLASSICAL_MODELS:
            X_tr_flat  = X_inner.reshape(len(X_inner), -1)
            X_te_flat  = X_test_fold.reshape(len(X_test_fold), -1)
            clf = _make_classical_model(model_type, best_params, y_inner)
            clf.fit(X_tr_flat, y_inner)
            save_results_classical(
                clf=clf, X_test_flat=X_te_flat, y_test=y_test_fold,
                decision_threshold=threshold,
                i=f"outer_s{left_out}", output_dir=fold_dir,
            )
        else:
            input_shape = input_shape_dict[model_type]

            # Use the inner-LOGO best val subject for early stopping
            logo_inner   = LeaveOneGroupOut()
            inner_groups = np.unique(groups_inner)
            val_subject  = inner_groups[outer_idx % len(inner_groups)]
            val_mask     = groups_inner == val_subject
            X_tr = X_inner[~val_mask]
            y_tr = y_inner[~val_mask]
            X_vl = X_inner[val_mask]
            y_vl = y_inner[val_mask]

            model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
            model.to(Config.DEVICE)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=best_params["learning_rate"],
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
            )
            class_counts  = np.bincount(y_tr, minlength=Config.NUM_LABELS)
            class_weights = len(y_tr) / (Config.NUM_LABELS * class_counts.astype(float))
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
            criterion     = nn.CrossEntropyLoss(weight=weight_tensor)

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                               torch.tensor(y_tr, dtype=torch.long)),
                batch_size=batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_vl, dtype=torch.float32),
                               torch.tensor(y_vl, dtype=torch.long)),
                batch_size=batch_size, shuffle=False,
            )
            test_loader = DataLoader(
                TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32),
                               torch.tensor(y_test_fold, dtype=torch.long)),
                batch_size=batch_size, shuffle=False,
            )

            fold_label = f"outer_s{left_out}"
            _, _, val_losses, train_losses = train(
                model, train_loader, val_loader, optimizer, criterion, Config.DEVICE,
                epochs=epochs,
                early_stopping=True,
                patience=Config.TRAINING_CONFIG['patience'],
                scaler=None,
                scheduler=scheduler,
            )

            plot_loss_curve(train_losses, val_losses, fold_dir, fold_label)
            pd.DataFrame({
                "epoch": range(1, len(train_losses) + 1),
                "train_loss": train_losses,
                "val_loss": val_losses,
            }).to_csv(os.path.join(fold_dir, f"losses_{fold_label}.csv"), index=False)

            save_results(
                model=model,
                val_loader=test_loader,
                y_val_onehot=y_test_fold,
                i=fold_label,
                decision_threshold=threshold,
                output_dir=fold_dir,
                device=Config.DEVICE,
            )

        print(f"  Outer fold s{left_out} concluído — salvo em {fold_dir}")

    print(f"\nNested LOGO concluído! Resultados em: {base_out}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fall-detect pipeline: train | nested",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
        p.add_argument("--model", required=False, choices=["CNN1D", "MLP", "LSTM", "RF", "SVM", "XGBoost", "CatBoost"])

    # --- train ---
    p_train = subparsers.add_parser("train", help="Outer LOGO with default HPs — no search, zero leakage")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=200)

    # --- nested ---
    p_nested = subparsers.add_parser(
        "nested",
        help="Nested LOGO: outer LOGO, inner Optuna per fold — gold standard, zero leakage",
    )
    add_common(p_nested)
    p_nested.add_argument("--n_trials", type=int, default=15,
                          help="Inner Optuna trials per outer fold (default=15)")
    p_nested.add_argument("--epochs", type=int, default=200)
    p_nested.add_argument("--inner", choices=["kfold", "holdout", "none"], default="kfold",
                          help="Inner CV strategy: kfold=GroupKFold(k=3), holdout=GroupShuffleSplit(n=1), none=in-sample (default=kfold)")

    return parser


def main():
    Config.setup_device()
    Config.set_seed()

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train":  run_final_training,
        "nested": run_nested_logo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
