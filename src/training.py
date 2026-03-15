"""Training routines and model builders."""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

import optuna

from neural_networks import CNN1DNet, MLPNet, LSTMNet
from config import Config
from test import save_results, save_results_classical, plot_loss_curve


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=25,
    early_stopping=False,
    patience=5,
    scaler=None,
    trial=None,
    step_offset=0,
    scheduler=None,
):
    """Train with optional early stopping, mixed precision and Optuna pruning."""
    model.to(device, non_blocking=True)
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    avg_train_losses, avg_val_losses = [], []

    for epoch in range(epochs):
        print(f"\n[{epoch}/{epochs}]")
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_train_losses.append(avg_train_loss)

        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        out = model(xb)
                        loss = criterion(out, yb)
                else:
                    out = model(xb)
                    loss = criterion(out, yb)
                val_losses.append(loss.item())
                y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        avg_val_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if trial is not None:
            trial.report(avg_val_loss, step_offset + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    torch.cuda.empty_cache()

    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        val_losses, y_pred, y_true = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())
        avg_val_losses[-1] = np.mean(val_losses)

    return y_pred, y_true, avg_val_losses, avg_train_losses


def create_model(model_type, best_params, input_shape, num_labels):
    """Create a neural model from the selected hyperparameters."""
    if model_type == "CNN1D":
        return CNN1DNet(
            input_shape=input_shape,
            filter_size=best_params["filter_size"],
            kernel_size=best_params["kernel_size"],
            num_layers=best_params["num_layers"],
            num_dense_layers=best_params["num_dense_layers"],
            dense_neurons=best_params["dense_neurons"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels,
        )
    if model_type == "MLP":
        return MLPNet(
            input_dim=input_shape,
            num_layers=best_params["num_layers"],
            dense_neurons=best_params["dense_neurons"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels,
        )
    if model_type == "LSTM":
        return LSTMNet(
            input_dim=input_shape[1],
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels,
        )
    raise ValueError(f"Tipo de modelo nao suportado: {model_type}")


def _make_classical_model(model_type, params, y_train):
    """Instantiate a classical model from a parameter dict."""
    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 10)),
            min_samples_split=int(params.get("min_samples_split", 5)),
            class_weight="balanced",
            random_state=Config.SEED,
            n_jobs=-1,
        )
    if model_type == "SVM":
        return CalibratedClassifierCV(
            LinearSVC(
                C=float(params.get("C", 1.0)),
                class_weight="balanced",
                dual="auto",
                max_iter=2000,
                random_state=Config.SEED,
            ),
            cv=3,
            method="sigmoid",
        )
    if model_type == "XGBoost":
        scale_pos_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=Config.SEED,
            n_jobs=-1,
        )
    if model_type == "CatBoost":
        if CatBoostClassifier is None:
            raise ImportError("CatBoost nao esta instalado. Instale com: pip install catboost")
        class_weights = [
            1.0,
            max(float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0), 1.0),
        ]
        return CatBoostClassifier(
            iterations=int(params.get("n_estimators", 200)),
            depth=int(params.get("depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            class_weights=class_weights,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=Config.SEED,
            verbose=False,
        )
    raise ValueError(f"Unknown classical model type: {model_type}")


SCENARIO_CHOICES = list(Config.SCENARIOS.keys())


def drop_mag_channels(X):
    """Drop the engineered magnitude channels (mag_acc and mag_gyr) from X.

    Channel layout per 8-channel sensor block:
      0: mag_acc  1: acc_x  2: acc_y  3: acc_z
      4: mag_gyr  5: gyr_x  6: gyr_y  7: gyr_z

    Drops indices 0 and 4 for each sensor block, so:
      8-ch  -> 6-ch  (drop [0,4])
      16-ch -> 12-ch (drop [0,4,8,12])
      24-ch -> 18-ch (drop [0,4,8,12,16,20])
    """
    C = X.shape[2]
    n_sensors = C // 8
    mag_cols = {s * 8 + offset for s in range(n_sensors) for offset in (0, 4)}
    keep_cols = [c for c in range(C) if c not in mag_cols]
    return X[:, :, keep_cols]


def keep_only_mag_channels(X):
    """Keep only the engineered magnitude channels (mag_acc and mag_gyr), drop raw axes.

    Channel layout per 8-channel sensor block:
      0: mag_acc  1: acc_x  2: acc_y  3: acc_z
      4: mag_gyr  5: gyr_x  6: gyr_y  7: gyr_z

    Keeps indices 0 and 4 for each sensor block, so:
      8-ch  -> 2-ch  (keep [0,4])
      16-ch -> 4-ch  (keep [0,4,8,12])
      24-ch -> 6-ch  (keep [0,4,8,12,16,20])
    """
    C = X.shape[2]
    n_sensors = C // 8
    keep_cols = [s * 8 + offset for s in range(n_sensors) for offset in (0, 4)]
    return X[:, :, keep_cols]


def _input_shape_from_data(X, model_type):
    """Derive the correct input_shape for a model from the actual data array."""
    _, T, C = X.shape
    if model_type == "MLP":
        return T * C
    return (T, C)  # CNN1D and LSTM


def run_final_training(args):
    """Outer LOGO over all subjects using Config.DEFAULT_PARAMS, no HP search."""
    scenario = args.scenario
    model_type_arg = args.model
    epochs = args.epochs
    loss_type = getattr(args, "loss", "weighted")
    inner_val_groups = max(int(getattr(args, "inner_val_groups", 3)), 1)
    scale = getattr(args, "scale", False)
    no_mag = getattr(args, "no_mag", False)
    only_mag = getattr(args, "only_mag", False)

    if not model_type_arg:
        raise ValueError("--model e obrigatorio para o modo train.")

    model_type = model_type_arg
    best_params = Config.DEFAULT_PARAMS[model_type]
    print(f"Usando parametros padrao para {model_type}: {best_params}")
    print(f"Loss: {loss_type} class weights")
    print(f"Inner validation groups per outer fold: {inner_val_groups}")

    scenario_out = scenario if loss_type == "weighted" else scenario + "_NW"
    if model_type not in Config.CLASSICAL_MODELS:
        scenario_out = f"{scenario_out}_IVG{inner_val_groups}"
    if scale:
        scenario_out = f"{scenario_out}_SC"
    if no_mag:
        scenario_out = f"{scenario_out}_NM"
    if only_mag:
        scenario_out = f"{scenario_out}_OM"
    base_out = Config.get_output_dir(model_type_arg, scenario_out)
    os.makedirs(base_out, exist_ok=True)

    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))

    if no_mag:
        X = drop_mag_channels(X)
        print(f"Dropped magnitude channels — new X shape: {X.shape}")
    if only_mag:
        X = keep_only_mag_channels(X)
        print(f"Kept only magnitude channels — new X shape: {X.shape}")

    unique_subjects = np.unique(groups)
    print(f"Sujeitos (LOGO): {sorted(unique_subjects.tolist())} ({len(unique_subjects)} total)")

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    threshold = best_params.get("decision_threshold", 0.5)

    if model_type in Config.CLASSICAL_MODELS:
        Config.set_seed(Config.FINAL_TRAINING["seed_offset"])
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            left_out = groups[test_idx[0]]
            fold_dir = os.path.join(base_out, f"fold_s{left_out}")
            model_fold_dir = os.path.join(Config.get_models_dir(model_type_arg, scenario_out), f"fold_s{left_out}")
            fold_label = f"s{left_out}"
            done_marker = os.path.join(fold_dir, f"metrics_model_{fold_label}.csv")
            if os.path.exists(done_marker):
                print(f"  Fold s{left_out} ja concluido - pulando.")
                continue
            print(f"  Fold {fold_idx + 1}/{n_folds} - sujeito de teste: {left_out}")
            os.makedirs(fold_dir, exist_ok=True)
            X_tr = X[train_idx].reshape(len(train_idx), -1)
            y_tr = y[train_idx]
            X_te = X[test_idx].reshape(len(test_idx), -1)
            y_te = y[test_idx]
            if scale:
                ss = StandardScaler()
                X_tr = ss.fit_transform(X_tr)
                X_te = ss.transform(X_te)
            clf = _make_classical_model(model_type, best_params, y_tr)
            clf.fit(X_tr, y_tr)
            save_results_classical(
                clf=clf,
                X_test_flat=X_te,
                y_test=y_te,
                decision_threshold=threshold,
                i=fold_label,
                output_dir=fold_dir,
                model_output_dir=model_fold_dir,
            )
            print(f"  Fold s{left_out} concluido")
        print(f"\nLOGO concluido! Resultados em: {base_out}")
        return

    input_shape_dict = Config.get_input_shape_dict(scenario, model_type)
    input_shape = input_shape_dict[model_type]
    if no_mag:
        input_shape = _input_shape_from_data(X, model_type)
    if only_mag:
        input_shape = _input_shape_from_data(X, model_type)
    batch_size = Config.TRAINING_CONFIG.get("batch_size", 32)

    Config.set_seed(Config.FINAL_TRAINING["seed_offset"])
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        left_out = groups[test_idx[0]]
        fold_dir = os.path.join(base_out, f"fold_s{left_out}")
        model_fold_dir = os.path.join(Config.get_models_dir(model_type_arg, scenario_out), f"fold_s{left_out}")
        fold_label = f"s{left_out}"
        done_marker = os.path.join(fold_dir, f"metrics_model_{fold_label}.csv")
        if os.path.exists(done_marker):
            print(f"\n  Fold s{left_out} ja concluido - pulando.")
            continue
        print(f"\n  Fold {fold_idx + 1}/{n_folds} - sujeito de teste: {left_out}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train_all = X[train_idx]
        y_train_all = y[train_idx]
        groups_train = groups[train_idx]

        inner_subjects = np.unique(groups_train)
        n_val_groups = min(inner_val_groups, len(inner_subjects) - 1)
        if n_val_groups <= 0:
            raise ValueError(
                "Inner validation requires at least 2 training groups in each outer fold."
            )
        start_idx = fold_idx % len(inner_subjects)
        val_subjects = [
            inner_subjects[(start_idx + k) % len(inner_subjects)] for k in range(n_val_groups)
        ]
        val_mask = np.isin(groups_train, val_subjects)
        print(f"    Validation subjects: {sorted(np.array(val_subjects).tolist())}")
        X_train = X_train_all[~val_mask]
        y_train = y_train_all[~val_mask]
        X_es = X_train_all[val_mask]
        y_es = y_train_all[val_mask]

        X_test = X[test_idx]
        y_test = y[test_idx]

        if scale:
            N_tr, T, C = X_train.shape
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train.reshape(-1, C)).reshape(N_tr, T, C)
            X_es   = ss.transform(X_es.reshape(-1, C)).reshape(X_es.shape[0], T, C)
            X_test = ss.transform(X_test.reshape(-1, C)).reshape(X_test.shape[0], T, C)

        model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
        model.to(Config.DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        if loss_type == "weighted":
            class_counts = np.bincount(y_train, minlength=Config.NUM_LABELS)
            class_weights = len(y_train) / (Config.NUM_LABELS * class_counts.astype(float))
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        es_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_es, dtype=torch.float32),
                torch.tensor(y_es, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        _, _, val_losses, train_losses = train(
            model,
            train_loader,
            es_loader,
            optimizer,
            criterion,
            Config.DEVICE,
            epochs=epochs,
            early_stopping=True,
            patience=Config.TRAINING_CONFIG["patience"],
            scaler=None,
            scheduler=scheduler,
        )

        plot_loss_curve(train_losses, val_losses, fold_dir, fold_label)

        save_results(
            model=model,
            val_loader=test_loader,
            y_val_onehot=y_test,
            i=fold_label,
            decision_threshold=threshold,
            output_dir=fold_dir,
            device=Config.DEVICE,
            model_output_dir=model_fold_dir,
        )
        print(f"  Fold s{left_out} concluido - salvo em {fold_dir}")

    print(f"\nLOGO concluido! Resultados em: {base_out}")


def build_parser():
    parser = argparse.ArgumentParser(description="Final training CLI")
    parser.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
    parser.add_argument(
        "--model",
        required=True,
        choices=["CNN1D", "MLP", "LSTM", "RF", "SVM", "XGBoost", "CatBoost"],
    )
    parser.add_argument("--epochs", type=int, default=Config.TRAINING_CONFIG["epochs"])
    parser.add_argument(
        "--loss",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Loss weighting: 'weighted' uses inverse-frequency class weights (default); "
             "'unweighted' uses plain CrossEntropyLoss. Unweighted results are saved to "
             "<scenario>_NW directories.",
    )
    parser.add_argument(
        "--inner-val-groups",
        type=int,
        default=3,
        help="Number of training subjects held out for inner validation in each outer LOGO fold "
             "(group-wise, default=3).",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Fit a StandardScaler on the training split of each LOGO fold and apply it to "
             "validation and test. Scaled runs are saved to <scenario>_SC directories.",
    )
    parser.add_argument(
        "--no-mag",
        dest="no_mag",
        action="store_true",
        default=False,
        help="Drop the engineered magnitude channels (mag_acc, mag_gyr) from every sensor block "
             "before training. Results are saved to <scenario>_NM directories.",
    )
    parser.add_argument(
        "--only-mag",
        dest="only_mag",
        action="store_true",
        default=False,
        help="Keep only the engineered magnitude channels (mag_acc, mag_gyr), dropping raw axes. "
             "Results are saved to <scenario>_OM directories.",
    )
    return parser


def main():
    Config.setup_device()
    Config.set_seed()
    args = build_parser().parse_args()
    run_final_training(args)


if __name__ == "__main__":
    main()
