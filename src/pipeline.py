"""Unified pipeline: hyperparameter search, post-trials analysis, and final training.

Usage:
    python pipeline.py search      -scenario <s> [--nn <m>] [--n_trials N] [--timeout T]
    python pipeline.py post_trials -scenario <s> [--nn <m>] [--n_trials N] [--timeout T]
    python pipeline.py train       -scenario <s> [--nn <m>] [--num_models N] [--epochs E]
    python pipeline.py nested      -scenario <s> [--nn <m>] [--n_trials N] [--epochs E]

Two evaluation strategies
--------------------------
  train   Fixed 3-subject holdout.
          Optuna ran on 12 subjects; each LOGO fold trains on 11 and evals on fixed 3.

  nested  Nested LOGO (gold standard, 15x more compute).
          Outer LOGO over all 15 subjects; for each outer fold a fresh inner Optuna
          runs on the remaining 14 subjects to pick HPs, then evals on the left-out 1.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
import optuna
import optuna.visualization as vis
import pandas as pd

from utils import run_optuna, train, save_results, save_results_classical, _make_classical_model, load_hyperparameters, load_test_data, create_model, plot_loss_curve
from config import Config

SCENARIO_CHOICES = [
    "chest_T", "chest_F", "left_T", "left_F", "right_T", "right_F",
    "chest_left_right_T", "chest_left_right_F",
    "chest_left_T", "chest_left_F",
    "chest_right_T", "chest_right_F",
]

def _split_and_report(X, y, groups):
    """Hold out the last N_TEST_INDIVIDUALS groups as the test set.

    All samples from the same individual stay together — no individual
    ever appears in both train/val and test.
    """
    unique_groups = np.unique(groups)  # sorted ascending
    assert len(unique_groups) == Config.N_INDIVIDUALS, (
        f"Expected {Config.N_INDIVIDUALS} individuals, found {len(unique_groups)}"
    )

    test_groups = unique_groups[-Config.N_TEST_INDIVIDUALS:]
    test_mask   = np.isin(groups, test_groups)
    trainval_mask = ~test_mask

    X_trainval     = X[trainval_mask]
    y_trainval     = y[trainval_mask]
    groups_trainval = groups[trainval_mask]
    X_test         = X[test_mask]
    y_test         = y[test_mask]

    print(f"Individuals in train/val: {sorted(np.unique(groups_trainval).tolist())}  ({X_trainval.shape[0]} samples)")
    print(f"Individuals in test:      {sorted(test_groups.tolist())}  ({X_test.shape[0]} samples)")
    return X_trainval, X_test, y_trainval, y_test, groups_trainval


def _print_best_params(model_type, best_value, best_params):
    """Print a summary of the best hyperparameters."""
    print(f"\n{'='*50}")
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print(f"{'='*50}")
    print(f"Modelo: {model_type}")
    print(f"Melhor valor: {best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")


def _save_study_results(base_out, scenario, model_type, study, best_params,
                        X_trainval, y_trainval, X_test, y_test, groups_trainval):
    """Persist best_hyperparameters.json and test_data.npz."""
    results_file = os.path.join(base_out, "best_hyperparameters.json")
    results = {
        "scenario": scenario,
        "model_type": model_type,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": len(study.trials),
        "optimization_history": [t.value for t in study.trials if t.value is not None],
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {results_file}")

    test_data_file = os.path.join(base_out, "test_data.npz")
    np.savez(test_data_file, X_trainval=X_trainval, y_trainval=y_trainval,
             X_test=X_test, y_test=y_test, groups_trainval=groups_trainval)
    print(f"Dados salvos em: {test_data_file}")
    print("\nBusca de hiperparâmetros concluída!")
    print("Próximo passo: executar treinamento LOGO com os melhores parâmetros")

def run_hyperparameter_search(args):
    """Run Optuna hyperparameter search and persist results."""
    scenario = args.scenario
    model_type_arg = args.model

    Config.OPTUNA_CONFIG['n_trials'] = args.n_trials
    Config.OPTUNA_CONFIG['timeout'] = args.timeout

    base_out = Config.get_output_dir(model_type_arg, scenario)

    print(f"\nCarregando dados para cenário: {scenario}")
    print(f"Modelo: {model_type_arg if model_type_arg else 'Todos'}")

    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))

    print(f"Shape dos dados: {X.shape}")
    print(f"Shape dos labels: {y.shape}")

    X_trainval, X_test, y_trainval, y_test, groups_trainval = _split_and_report(X, y, groups)

    input_shape_dict = Config.get_input_shape_dict(scenario, model_type_arg)
    print(f"Input shapes: {input_shape_dict}")

    print(f"\nIniciando Otimização com Optuna (LOGO sobre {len(np.unique(groups_trainval))} indivíduos de treino)...")
    print(f"Número de trials: {Config.OPTUNA_CONFIG['n_trials']}")
    print(f"Timeout: {Config.OPTUNA_CONFIG.get('timeout', 'N/A')} segundos")

    study_name = f"{scenario}_{model_type_arg}" if model_type_arg else scenario

    study = run_optuna(
        input_shape_dict=input_shape_dict,
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        groups=groups_trainval,
        output_dir=base_out,
        num_labels=Config.NUM_LABELS,
        device=Config.DEVICE,
        restrict_model_type=model_type_arg,
        study_name=study_name,
    )

    best_params = study.best_params
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg

    _print_best_params(model_type, study.best_value, best_params)
    _save_study_results(base_out, scenario, model_type, study, best_params,
                        X_trainval, y_trainval, X_test, y_test, groups_trainval)

def _load_optuna_study(output_dir, study_name):
    """Load an existing Optuna study and export reports."""
    db_path = os.path.join(output_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"Estudo existente carregado de: {db_path}")
    print("Melhor MCC:", study.best_value)
    print("Melhores hiperparâmetros:", study.best_params)

    os.makedirs(output_dir, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    try:
        fig = vis.plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, "param_importance.png"))
    except Exception as e:
        print(f"Could not save importance plot: {e}")

    return study


def run_post_trials(args):
    """Load a completed Optuna study and persist best parameters + test data."""
    scenario = args.scenario
    model_type_arg = args.model

    Config.OPTUNA_CONFIG['n_trials'] = args.n_trials
    Config.OPTUNA_CONFIG['timeout'] = args.timeout

    base_out = Config.get_output_dir(model_type_arg, scenario)

    print(f"\nCarregando dados para cenário: {scenario}")
    print(f"Modelo: {model_type_arg if model_type_arg else 'Todos'}")

    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))

    print(f"Shape dos dados: {X.shape}")
    print(f"Shape dos labels: {y.shape}")

    X_trainval, X_test, y_trainval, y_test, groups_trainval = _split_and_report(X, y, groups)

    input_shape_dict = Config.get_input_shape_dict(scenario, model_type_arg)
    print(f"Input shapes: {input_shape_dict}")

    study_name = f"{scenario}_{model_type_arg}" if model_type_arg else scenario

    study = _load_optuna_study(output_dir=base_out, study_name=study_name)

    best_params = study.best_params
    if "best_params" in best_params:
        best_params = best_params["best_params"]
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg

    _print_best_params(model_type, study.best_value, best_params)
    _save_study_results(base_out, scenario, model_type, study, best_params,
                        X_trainval, y_trainval, X_test, y_test, groups_trainval)

def run_final_training(args):
    """LOGO over the 12 trainval subjects, evaluated on the fixed 3 test subjects.

    Each LOGO fold trains on 11 subjects and uses the left-out 1 as early-stopping
    validation.  The held-out 3 test subjects were never seen during HP search and
    are never used for any training decision here.

    num_models = number of complete LOGO repetitions (default 1 = 12 folds).
    """
    scenario = args.scenario
    model_type_arg = args.model
    num_reps = args.num_models   # number of complete LOGO repetitions
    epochs = args.epochs

    base_out = Config.get_output_dir(model_type_arg, scenario)
    os.makedirs(base_out, exist_ok=True)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    hp_file = os.path.join(base_out, "best_hyperparameters.json")
    if os.path.exists(hp_file):
        best_params = load_hyperparameters(base_out)
        if "best_params" in best_params:
            best_params = best_params["best_params"]
        model_type = best_params.get("model_type", model_type_arg)
        print(f"Usando hiperparâmetros do Optuna: {hp_file}")
    else:
        if not model_type_arg:
            raise ValueError(
                f"Nenhum resultado de busca encontrado em '{base_out}'. "
                "Use --nn para especificar o modelo ao treinar com parâmetros padrão."
            )
        model_type = model_type_arg
        best_params = Config.DEFAULT_PARAMS[model_type]
        print(f"[AVISO] Nenhum resultado de busca encontrado. "
              f"Usando parâmetros padrão para {model_type}: {best_params}")

    # ── Data ─────────────────────────────────────────────────────────────────
    test_data_file = os.path.join(base_out, "test_data.npz")
    if os.path.exists(test_data_file):
        data = np.load(test_data_file)
        X_trainval   = data['X_trainval']
        y_trainval   = data['y_trainval']
        X_test       = data['X_test']
        y_test       = data['y_test']
        groups_trainval = data['groups_trainval']
        print(f"Dados carregados de: {test_data_file}")
    else:
        print("[AVISO] test_data.npz não encontrado. Carregando e dividindo dados brutos...")
        X = np.load(Config.get_data_file(scenario))
        y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
        groups = np.load(Config.get_groups_file(scenario))
        X_trainval, X_test, y_trainval, y_test, groups_trainval = _split_and_report(X, y, groups)
        np.savez(test_data_file, X_trainval=X_trainval, y_trainval=y_trainval,
                 X_test=X_test, y_test=y_test, groups_trainval=groups_trainval)
        print(f"Divisão salva em {test_data_file}")

    unique_subjects = np.unique(groups_trainval)
    print(f"Sujeitos de treino (LOGO): {sorted(unique_subjects.tolist())} ({len(unique_subjects)} total)")
    print(f"Sujeitos de teste (fixos): {sorted(np.unique(y_test).tolist() if False else [])}  ({X_test.shape[0]} amostras)")
    print(f"Repetições LOGO: {num_reps}")

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups_trainval)
    threshold = best_params.get("decision_threshold", 0.5)

    # ── Classical models (RF / SVM / XGBoost) ──────────────────────────────
    if model_type in Config.CLASSICAL_MODELS:
        X_test_flat = X_test.reshape(len(X_test), -1)
        for rep in range(1, num_reps + 1):
            Config.set_seed(Config.FINAL_TRAINING['seed_offset'] + rep)
            print(f"\n=== Repetição LOGO {rep}/{num_reps} ===")
            for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X_trainval, y_trainval, groups_trainval)):
                left_out = groups_trainval[val_idx[0]]
                print(f"  Fold {fold_idx+1}/{n_folds} — sujeito de val: {left_out}")
                fold_dir = os.path.join(base_out, f"rep_{rep}", f"fold_s{left_out}")
                os.makedirs(fold_dir, exist_ok=True)
                fold_label = f"rep{rep}_s{left_out}"
                X_tr = X_trainval[train_idx].reshape(len(train_idx), -1)
                y_tr = y_trainval[train_idx]
                clf = _make_classical_model(model_type, best_params, y_tr)
                clf.fit(X_tr, y_tr)
                save_results_classical(
                    clf=clf, X_test_flat=X_test_flat, y_test=y_test,
                    decision_threshold=threshold, i=fold_label, output_dir=fold_dir,
                )
                print(f"  Fold s{left_out} concluído")
        print(f"\nLOGO concluído! Resultados em: {base_out}")
        return

    # ── Neural networks ───────────────────────────────────────────────────────
    input_shape_dict = Config.get_input_shape_dict(scenario, model_type)
    input_shape = input_shape_dict[model_type]
    batch_size = Config.TRAINING_CONFIG.get('batch_size', 32)

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False,
    )

    for rep in range(1, num_reps + 1):
        Config.set_seed(Config.FINAL_TRAINING['seed_offset'] + rep)
        print(f"\n=== Repetição LOGO {rep}/{num_reps} ===")

        for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X_trainval, y_trainval, groups_trainval)):
            left_out = groups_trainval[val_idx[0]]
            print(f"\n  Fold {fold_idx+1}/{n_folds} — sujeito de val: {left_out}")
            fold_dir = os.path.join(base_out, f"rep_{rep}", f"fold_s{left_out}")
            os.makedirs(fold_dir, exist_ok=True)
            fold_label = f"rep{rep}_s{left_out}"

            X_train = X_trainval[train_idx]
            y_train = y_trainval[train_idx]
            X_val   = X_trainval[val_idx]
            y_val   = y_trainval[val_idx]

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
                val_loader=test_loader,
                y_val_onehot=y_test,
                i=fold_label,
                decision_threshold=threshold,
                output_dir=fold_dir,
                device=Config.DEVICE,
            )
            print(f"  Fold s{left_out} concluído — salvo em {fold_dir}")

    print(f"\nLOGO concluído! Resultados em: {base_out}")


def run_nested_logo(args):
    """Nested LOGO: outer LOGO over all 15 subjects, inner Optuna over 14 per fold.

    For each outer fold:
      - Inner Optuna runs LOGO over the 14 remaining subjects to select HPs.
      - A model is trained on all 14 (using the best inner fold as val for early stopping).
      - Evaluated on the 1 left-out outer subject.

    This is the gold-standard evaluation: zero HP leakage, all 15 subjects contribute
    exactly one test result.
    """
    scenario      = args.scenario
    model_type_arg = args.model
    n_trials      = args.n_trials
    epochs        = args.epochs

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

        # ── Inner Optuna (LOGO over 14 subjects) ──────────────────────────
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
        )

        best_params = study.best_params
        model_type  = best_params["model_type"] if not model_type_arg else model_type_arg
        threshold   = best_params.get("decision_threshold", 0.5)

        with open(os.path.join(fold_dir, "best_hyperparameters.json"), "w") as f:
            json.dump({"outer_subject": int(left_out),
                       "model_type": model_type,
                       "best_value": float(study.best_value),
                       "best_params": best_params}, f, indent=2)

        print(f"  Best HPs (inner): {best_params}")

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
        description="Fall-detect pipeline: search | post_trials | train | nested",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared scenario/nn arguments
    def add_common(p):
        p.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
        p.add_argument("--model", required=False, choices=["CNN1D", "MLP", "LSTM", "RF", "SVM", "XGBoost"])

    # --- search ---
    p_search = subparsers.add_parser("search", help="Run Optuna hyperparameter search")
    add_common(p_search)
    p_search.add_argument("--n_trials", type=int, default=30)
    p_search.add_argument("--timeout", type=int, default=3600)

    # --- post_trials ---
    p_post = subparsers.add_parser("post_trials", help="Analyse a completed Optuna study")
    add_common(p_post)
    p_post.add_argument("--n_trials", type=int, default=30)
    p_post.add_argument("--timeout", type=int, default=3600)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Final training with best hyperparameters")
    add_common(p_train)
    p_train.add_argument("--num_models", type=int, default=1,
                         help="Number of complete LOGO repetitions (default=1 → 15 folds)")
    p_train.add_argument("--epochs", type=int, default=200)

    # --- nested ---
    p_nested = subparsers.add_parser(
        "nested",
        help="Nested LOGO: outer LOGO over 15 subjects, inner Optuna per fold (gold standard)",
    )
    add_common(p_nested)
    p_nested.add_argument("--n_trials", type=int, default=15,
                          help="Inner Optuna trials per outer fold (default=15)")
    p_nested.add_argument("--epochs", type=int, default=200)

    return parser


def main():
    Config.setup_device()
    Config.set_seed()

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "search":     run_hyperparameter_search,
        "post_trials": run_post_trials,
        "train":      run_final_training,
        "nested":     run_nested_logo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
