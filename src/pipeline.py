"""
Unified pipeline: hyperparameter search, post-trials analysis, and final training.

Usage:
    python pipeline.py search      -scenario <s> [--nn <m>] [--n_trials N] [--timeout T]
    python pipeline.py post_trials -scenario <s> [--nn <m>] [--n_trials N] [--timeout T]
    python pipeline.py train       -scenario <s> [--nn <m>] [--num_models N] [--epochs E]
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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
                        X_trainval, y_trainval, X_test, y_test):
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
             X_test=X_test, y_test=y_test)
    print(f"Dados de treino/validação e teste salvos em: {test_data_file}")
    print("\nBusca de hiperparâmetros concluída!")
    print("Próximo passo: executar treinamento final com os melhores parâmetros")

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

    print(f"\nIniciando Otimização com Optuna...")
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
                        X_trainval, y_trainval, X_test, y_test)

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

    X_trainval, X_test, y_trainval, y_test, _ = _split_and_report(X, y, groups)

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
                        X_trainval, y_trainval, X_test, y_test)

def run_final_training(args):
    """Train N models with the best hyperparameters found by Optuna.

    If no prior search artefacts exist (best_hyperparameters.json / test_data.npz)
    the function falls back to Config.DEFAULT_PARAMS for the requested model type
    and re-splits the raw dataset, so training can run without a search step.
    """
    scenario = args.scenario
    model_type_arg = args.model
    num_models = args.num_models
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
                "Nenhum resultado de busca encontrado em '{base_out}'. "
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
        X_trainval, y_trainval = data['X_trainval'], data['y_trainval']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        print("[AVISO] test_data.npz não encontrado. Carregando e dividindo dados brutos...")
        X = np.load(Config.get_data_file(scenario))
        y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
        groups = np.load(Config.get_groups_file(scenario))
        X_trainval, X_test, y_trainval, y_test, _ = _split_and_report(X, y, groups)
        np.savez(test_data_file, X_trainval=X_trainval, y_trainval=y_trainval,
                 X_test=X_test, y_test=y_test)
        print(f"Divisão salva em {test_data_file}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.2,
        random_state=42,
        shuffle=False,
    )

    # ── Classical models (RF / SVM / XGBoost) ──────────────────────────────
    if model_type in Config.CLASSICAL_MODELS:
        X_trainval_flat = X_trainval.reshape(len(X_trainval), -1)
        X_test_flat     = X_test.reshape(len(X_test), -1)
        for i in range(1, num_models + 1):
            print(f"\nTreinando modelo final {i}/{num_models}...")
            Config.set_seed(Config.FINAL_TRAINING['seed_offset'] + i)
            clf = _make_classical_model(model_type, best_params, y_trainval)
            clf.fit(X_trainval_flat, y_trainval)
            model_dir = os.path.join(base_out, f"model_{i}")
            os.makedirs(model_dir, exist_ok=True)
            save_results_classical(
                clf=clf,
                X_test_flat=X_test_flat,
                y_test=y_test,
                decision_threshold=best_params.get("decision_threshold", 0.5),
                i=i,
                output_dir=model_dir,
            )
            print(f"Modelo {i} treinado e salvo em {model_dir}")
        print(f"\nTreinamento final concluído! Resultados salvos em: {base_out}")
        return

    # ── Neural networks ───────────────────────────────────────────────────────────
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

    for i in range(1, num_models + 1):
        print(f"\nTreinando modelo final {i}/{num_models}...")

        model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
        model.to(Config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
        # Apply same class weights used during hyperparameter search
        y_train_flat = y_train
        class_counts = np.bincount(y_train_flat, minlength=Config.NUM_LABELS)
        class_weights = len(y_train_flat) / (Config.NUM_LABELS * class_counts.astype(float))
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

        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, val_loader, optimizer, criterion, Config.DEVICE,
            epochs=epochs, early_stopping=False, patience=0, scaler=None,
        )

        model_dir = os.path.join(base_out, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_dir, f"model_{i}.pt"))

        plot_loss_curve(train_losses, val_losses, model_dir, f"{i}")

        np.save(os.path.join(model_dir, f"train_losses_model_{i}.npy"), np.array(train_losses))
        np.save(os.path.join(model_dir, f"val_losses_model_{i}.npy"), np.array(val_losses))

        df_losses = pd.DataFrame({
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        })
        df_losses.to_csv(os.path.join(model_dir, f"losses_model_{i}.csv"), index=False)

        save_results(
            model=model,
            val_loader=test_loader,
            y_val_onehot=y_test,
            i=i,
            decision_threshold=best_params.get("decision_threshold", 0.5),
            output_dir=model_dir,
            device=Config.DEVICE,
        )

        print(f"Modelo {i} treinado e salvo em {model_dir}")

    print(f"\nTreinamento final concluído! Resultados salvos em: {base_out}")

def build_parser():
    parser = argparse.ArgumentParser(
        description="Fall-detect pipeline: search | post_trials | train",
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
    p_train.add_argument("--num_models", type=int, default=30)
    p_train.add_argument("--epochs", type=int, default=200)

    return parser


def main():
    Config.setup_device()
    Config.set_seed()

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "search": run_hyperparameter_search,
        "post_trials": run_post_trials,
        "train": run_final_training,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
