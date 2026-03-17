"""Testing, reporting and result persistence helpers."""

import argparse
import json
import os
from collections import OrderedDict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from config import Config

def _cross_sensor_output_dirs(train_scenario, test_scenario, model_type, loss_type="weighted", scale=False, no_mag=False, only_mag=False):
    train_tag = train_scenario if loss_type == "weighted" else f"{train_scenario}_NW"
    if scale:
        train_tag += "_SC"
    if no_mag:
        train_tag += "_NM"
    if only_mag:
        train_tag += "_OM"
    scenario_name = f"cross_sensor_{train_tag}_to_{test_scenario}"
    base_out = Config.get_output_dir(model_type, scenario_name)
    model_out = Config.get_models_dir(model_type, scenario_name)
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(model_out, exist_ok=True)
    return base_out, model_out


def _build_cross_sensor_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=getattr(Config, "TORCH_GENERATOR", None),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def run_cross_sensor_eval(
    train_scenario,
    model_type,
    loss_type="weighted",
    epochs=None,
    scale=False,
    no_mag=False,
    only_mag=False,
):
    """Treina no sensor escolhido e testa automaticamente em todos os outros sensores definidos em Config.SCENARIOS (exceto o de treino)."""
    from training import (
        _input_shape_from_data,
        _make_classical_model,
        create_model,
        train,
        drop_mag_channels,
        keep_only_mag_channels,
    )

    Config.setup_device()
    Config.set_seed()

    print(f"\n[Cross-Sensor Eval] Train: {train_scenario} | Model: {model_type} | Loss: {loss_type}")
    X_full = np.load(Config.get_data_file(train_scenario))
    y_full = np.load(Config.get_labels_file(train_scenario)).astype(np.int64)

    # Shuffle before splitting
    rng = np.random.default_rng(Config.SEED)
    indices = np.arange(len(X_full))
    rng.shuffle(indices)
    X_full = X_full[indices]
    y_full = y_full[indices]

    n_total = len(X_full)
    n_trainval = int(0.8 * n_total)
    n_test = n_total - n_trainval

    X_trainval = X_full[:n_trainval]
    y_trainval = y_full[:n_trainval]
    X_test = X_full[n_trainval:]
    y_test = y_full[n_trainval:]

    # Split trainval into train and val (80/20 of trainval)
    n_train = int(0.8 * n_trainval)
    n_val = n_trainval - n_train
    X_train = X_trainval[:n_train]
    y_train = y_trainval[:n_train]
    X_val = X_trainval[n_train:]
    y_val = y_trainval[n_train:]
    best_params = dict(Config.DEFAULT_PARAMS[model_type])
    threshold = best_params.get("decision_threshold")
    batch_size = Config.TRAINING_CONFIG.get("batch_size")
    epochs = epochs if epochs is not None else Config.TRAINING_CONFIG.get("epochs")

    if no_mag:
        X_train = drop_mag_channels(X_train)
        X_val = drop_mag_channels(X_val)
        X_test = drop_mag_channels(X_test)
    if only_mag:
        X_train = keep_only_mag_channels(X_train)
        X_val = keep_only_mag_channels(X_val)
        X_test = keep_only_mag_channels(X_test)
    if scale:
        n_tr, t_steps, n_ch = X_train.shape
        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(X_train.reshape(-1, n_ch)).reshape(n_tr, t_steps, n_ch)
        X_val = feature_scaler.transform(X_val.reshape(-1, n_ch)).reshape(X_val.shape[0], t_steps, n_ch)
        X_test = feature_scaler.transform(X_test.reshape(-1, n_ch)).reshape(X_test.shape[0], t_steps, n_ch)

    # Treinamento
    if model_type in Config.CLASSICAL_MODELS:
        X_train_flat = X_train.reshape(len(X_train), -1)
        clf = _make_classical_model(model_type, best_params, y_train)
        clf.fit(X_train_flat, y_train)
        trained_model = clf
        # Save classical model ONCE after training
        train_base_out, model_save_dir = _cross_sensor_output_dirs(
            train_scenario,
            train_scenario,
            model_type,
            loss_type,
            scale=scale,
            no_mag=no_mag,
            only_mag=only_mag,
        )
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"model_{train_scenario}.pkl")
        joblib.dump(trained_model, model_save_path)
    else:
        input_shape = _input_shape_from_data(X_train, model_type)
        model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)
        model.to(Config.DEVICE)

        if loss_type == "weighted":
            class_counts = np.bincount(y_train, minlength=Config.NUM_LABELS)
            class_counts = np.maximum(class_counts, 1)
            class_weights = len(y_train) / (Config.NUM_LABELS * class_counts.astype(float))
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32, device=Config.DEVICE)
            )
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=Config.DEVICE.type == "cuda")
        train_loader, val_loader, _ = _build_cross_sensor_loaders(
            X_train, y_train, X_val, y_val, X_val, y_val, batch_size
        )

        _, _, val_losses, train_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=Config.DEVICE,
            epochs=epochs,
            early_stopping=True,
            patience=Config.TRAINING_CONFIG.get("patience"),
            scaler=scaler,
        )
        # Save loss curve using only the training sensor name (no _toALL suffix)
        train_base_out, model_save_dir = _cross_sensor_output_dirs(
            train_scenario,
            train_scenario,
            model_type,
            loss_type,
            scale=scale,
            no_mag=no_mag,
            only_mag=only_mag,
        )
        plot_loss_curve(train_losses, val_losses, train_base_out, f"{train_scenario}")
        trained_model = model
        # Save the model ONCE after training
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"model_{train_scenario}.pt")
        torch.save(model.state_dict(), model_save_path)


    # Teste em todos os sensores (exceto o de treino)
    # Only allow: left_T -> chest_T and chest_T -> left_T
    allowed_pairs = {
        "left_T": ["chest_T", "right_T"],
        "right_T": ["chest_T", "left_T"],
        "chest_T": ["left_T", "right_T"],
        # "chest_left_T": ["chest_right_T"],
        # "chest_right_T": ["chest_left_T"],
        #"chest_left_right_T": [],
    }
    for test_scenario in Config.SCENARIOS:
        if test_scenario == train_scenario:
            continue
        if train_scenario in allowed_pairs and test_scenario not in allowed_pairs[train_scenario]:
            continue
        print(f"Testando modelo treinado em {train_scenario} no sensor {test_scenario}")
        X_test_full = np.load(Config.get_data_file(test_scenario))
        y_test_full = np.load(Config.get_labels_file(test_scenario)).astype(np.int64)

        # Split de teste igual para todos: 20% finais do conjunto
        n_total_test = len(X_test_full)
        n_test = int(0.2 * n_total_test)
        if n_test == 0:
            n_test = 1
        X_te = X_test_full[-n_test:]
        y_te = y_test_full[-n_test:]
        if no_mag:
            X_te = drop_mag_channels(X_te)
        if only_mag:
            X_te = keep_only_mag_channels(X_te)
        if scale:
            n_te, t_steps, n_ch = X_te.shape
            feature_scaler = StandardScaler()
            X_te = feature_scaler.fit_transform(X_te.reshape(-1, n_ch)).reshape(n_te, t_steps, n_ch)

        base_out, model_out = _cross_sensor_output_dirs(
            train_scenario,
            test_scenario,
            model_type,
            loss_type,
            scale=scale,
            no_mag=no_mag,
            only_mag=only_mag,
        )
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(model_out, exist_ok=True)

        if model_type in Config.CLASSICAL_MODELS:
            X_te_flat = X_te.reshape(len(X_te), -1)
            save_results_classical(
                clf=trained_model,
                X_test_flat=X_te_flat,
                y_test=y_te,
                decision_threshold=threshold,
                i=f"{train_scenario}_to_{test_scenario}",
                output_dir=base_out,
                model_output_dir=model_out,
                save_model=False,  # Do not save model again
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_te, dtype=torch.float32),
                    torch.tensor(y_te, dtype=torch.long),
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            save_results(
                model=trained_model,
                val_loader=test_loader,
                y_val_onehot=None,
                i=f"{train_scenario}_to_{test_scenario}",
                decision_threshold=threshold,
                output_dir=base_out,
                device=Config.DEVICE,
                model_output_dir=model_out,
                save_model=False,  # Do not save model again
            )

    print(f"Strict sensor generalization results saved for training sensor: {train_scenario}")

def save_results(
    model,
    val_loader,
    y_val_onehot,
    i,
    decision_threshold,
    output_dir,
    device,
    model_output_dir=None,
    save_model=True,
):
    """Persist model checkpoint and full evaluation artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    if model_output_dir is None:
        model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    if save_model:
        model_path = os.path.join(model_output_dir, f"model_{i}.pt")
        torch.save(model.state_dict(), model_path)

    model.eval()
    y_probs = []
    y_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(yb.numpy())


    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    # Filter out samples with NaN predictions
    nan_mask = ~np.isnan(y_probs).any(axis=1)
    n_skipped = np.sum(~nan_mask)
    if n_skipped > 0:
        print(f"[WARNING] {n_skipped} samples with NaN predictions skipped for {i}.")
    y_probs = y_probs[nan_mask]
    y_true = y_true[nan_mask]

    if len(y_probs) == 0:
        print(f"[WARNING] All predictions are NaN for {i}. Skipping metrics and plots.")
        return

    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir)
    plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)

    metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
    record_metrics(metrics, tp, tn, fp, fn, i, output_dir)


def save_results_classical(
    clf,
    X_test_flat,
    y_test,
    decision_threshold,
    i,
    output_dir,
    model_output_dir=None,
    save_model=True,
):
    """Persist and evaluate classical sklearn/XGBoost/CatBoost models."""
    os.makedirs(output_dir, exist_ok=True)
    if model_output_dir is None:
        model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    if save_model:
        joblib.dump(clf, os.path.join(model_output_dir, f"model_{i}.pkl"))

    y_probs = clf.predict_proba(X_test_flat)
    y_true = y_test

    # Filter out samples with NaN predictions
    nan_mask = ~np.isnan(y_probs).any(axis=1)
    n_skipped = np.sum(~nan_mask)
    if n_skipped > 0:
        print(f"[WARNING] {n_skipped} samples with NaN predictions skipped for {i}.")
    y_probs = y_probs[nan_mask]
    y_true = y_true[nan_mask]

    if len(y_probs) == 0:
        print(f"[WARNING] All predictions are NaN for {i}. Skipping metrics and plots.")
        return

    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # Always use the same output structure as neural networks
    save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir)
    plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)

    metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
    record_metrics(metrics, tp, tn, fp, fn, i, output_dir)


def save_metrics_csv(tp, fp, tn, fn, y_true, y_pred, output_dir):
    """Save compact binary classification metrics as CSV."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=Config.METRICS_CONFIG["fall_class"], zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    path = os.path.join(output_dir, "metrics.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("prec,rec,f1,acc,tp,fp,tn,fn\n")
        f.write(
            f"{precision:.6f},{recall:.6f},{f1:.6f},{acc:.6f},"
            f"{int(tp)},{int(fp)},{int(tn)},{int(fn)}\n"
        )


def plot_roc_curve(y_score, y_true, output_dir, i):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Modelo {i}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_model_{i}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curve(train_losses, val_losses, output_dir, model_idx):
    """Save train/validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title(f"Training and Validation Loss - Modelo {model_idx}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_model_{model_idx}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def calculate_metrics(tp, tn, fp, fn, y_true, y_pred):
    """Calculate evaluation metrics from confusion matrix counts."""
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    f1 = f1_score(y_true, y_pred, pos_label=Config.METRICS_CONFIG["fall_class"], zero_division=0)

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "f1": f1,
    }


def record_metrics(metrics, tp, tn, fp, fn, i, output_dir):
    """Per-model metrics CSV output is intentionally disabled."""
    _ = (metrics, tp, tn, fp, fn, i, output_dir)


def load_model_state(model, path, device="cpu"):
    """Load a state dict and strip DataParallel prefix if needed."""
    state_dict = torch.load(path, map_location=device)

    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


def load_hyperparameters(output_dir):
    """Load best hyperparameters json from output directory."""
    results_file = os.path.join(output_dir, "best_hyperparameters.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Arquivo de hiperparametros nao encontrado: {results_file}")
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def load_test_data(output_dir):
    """Load test arrays from saved npz file."""
    test_data_file = os.path.join(output_dir, "test_data.npz")
    if not os.path.exists(test_data_file):
        raise FileNotFoundError(f"Arquivo de dados de teste nao encontrado: {test_data_file}")
    data = np.load(test_data_file)
    return data["X_test"], data["y_test"]


def plot_learning_curve(
    create_model_fn,
    X_full,
    y_full,
    groups_full,
    X_test,
    y_test,
    input_shape,
    num_labels,
    best_params,
    device,
    output_dir,
    fractions=None,
    epochs=None,
    seed=None,
    loss_type="weighted",
):
    """Generate group-aware learning curves by varying the number of train groups."""
    if fractions is None:
        fractions = Config.LEARNING_CURVE_CONFIG["fractions"]
    if epochs is None:
        epochs = Config.LEARNING_CURVE_CONFIG["epochs"]
    if seed is None:
        seed = Config.SEED

    rng = np.random.RandomState(seed)
    results = []

    print(f"\n{'=' * 50}")
    print("INICIANDO GERACAO DA LEARNING CURVE")
    print(f"{'=' * 50}")

    unique_groups = np.unique(groups_full)
    if len(unique_groups) < 2:
        raise ValueError("Sao necessarios pelo menos 2 grupos para learning curve baseada em grupos.")

    for frac in fractions:
        n_groups = int(round(len(unique_groups) * frac))
        n_groups = max(2, min(n_groups, len(unique_groups)))
        selected_groups = rng.choice(unique_groups, size=n_groups, replace=False)
        subset_mask = np.isin(groups_full, selected_groups)

        X_subset = X_full[subset_mask]
        y_subset = y_full[subset_mask]
        groups_subset = groups_full[subset_mask]

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + int(frac * 1000))
        tr_idx, vl_idx = next(splitter.split(X_subset, y_subset, groups_subset))
        X_tr, X_vl = X_subset[tr_idx], X_subset[vl_idx]
        y_tr, y_vl = y_subset[tr_idx], y_subset[vl_idx]

        print(
            f"\nTreinando com {len(np.unique(groups_subset))} grupos "
            f"({int(frac * 100)}% dos grupos, train={len(X_tr)}, val={len(X_vl)})"
        )

        # Always use config defaults for all hyperparameters
        model_type = best_params.get("model_type", "CNN1D")
        config_params = Config.DEFAULT_PARAMS[model_type].copy()
        config_params.update(best_params)  # allow best_params to override if present
        model = create_model_fn(config_params, input_shape, num_labels)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config_params["learning_rate"])

        if loss_type == "weighted":
            class_counts = np.bincount(y_tr, minlength=num_labels)
            class_counts = np.maximum(class_counts, 1)
            class_weights = len(y_tr) / (num_labels * class_counts.astype(float))
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        from training import train

        batch_size = Config.TRAINING_CONFIG.get("batch_size", 32)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_vl, dtype=torch.float32),
                torch.tensor(y_vl, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        _, _, val_losses, train_losses = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs=epochs,
            early_stopping=False,
            patience=Config.TRAINING_CONFIG.get("patience"),
            scaler=None,
        )

        model.eval()
        y_preds = []
        y_true_final = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb)
                y_preds.append(torch.argmax(preds, dim=1).cpu().numpy())
                y_true_final.append(yb.numpy())

        y_preds = np.concatenate(y_preds)
        y_true_final = np.concatenate(y_true_final)
        f1 = f1_score(y_true_final, y_preds, average="macro")
        acc = accuracy_score(y_true_final, y_preds)
        train_loss_mean = float(np.mean(train_losses))
        val_loss_mean = float(np.mean(val_losses))

        results.append(
            {
                "Fraction": frac,
                "Num_Groups": int(len(np.unique(groups_subset))),
                "f1": f1,
                "Accuracy": acc,
                "Train_Loss": train_loss_mean,
                "Val_Loss": val_loss_mean,
                "Loss_Type": loss_type,
            }
        )
        print(
            f"F1: {f1:.4f} | Acc: {acc:.4f} | "
            f"Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "learning_curve_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metricas da curva de aprendizado salvas em: {csv_path}")

    plt.figure(figsize=(10, 7))
    xvals = df["Num_Groups"] if "Num_Groups" in df.columns else (df["Fraction"] * 100)
    plt.plot(xvals, df["f1"], marker="o", label="F1-score")
    plt.plot(xvals, df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(xvals, df["Train_Loss"], marker="o", label="Train Loss")
    plt.plot(xvals, df["Val_Loss"], marker="o", label="Val Loss")
    plt.xlabel("Numero de Grupos de Treino")
    plt.ylabel("Valor da Metrica")
    plt.title("Curva de Aprendizado (por grupos)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    lc_plot_path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(lc_plot_path, dpi=300)
    plt.close()
    print(f"Curva de aprendizado salva em: {lc_plot_path}")
