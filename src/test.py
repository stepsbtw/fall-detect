"""Testing, reporting and result persistence helpers."""

import os
import json
from collections import OrderedDict

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

from config import Config


def save_results(
    model,
    val_loader,
    y_val_onehot,
    i,
    decision_threshold,
    output_dir,
    device,
    model_output_dir=None,
):
    """Persist model checkpoint and full evaluation artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    if model_output_dir is None:
        model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)

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
    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    save_confusion_matrix_txt(cm, output_dir, i)

    save_classification_report(y_pred, y_true, output_dir, i)
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
):
    """Persist and evaluate classical sklearn/XGBoost/CatBoost models."""
    os.makedirs(output_dir, exist_ok=True)
    if model_output_dir is None:
        model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(model_output_dir, f"model_{i}.pkl"))

    y_probs = clf.predict_proba(X_test_flat)
    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int)
    y_true = y_test

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    save_confusion_matrix_txt(cm, output_dir, i)

    save_classification_report(y_pred, y_true, output_dir, i)
    plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)

    metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
    record_metrics(metrics, tp, tn, fp, fn, i, output_dir)


def save_confusion_matrix_txt(cm, output_dir, i):
    """Save confusion matrix as plain text."""
    labels = ["Nao Queda", "Queda"]
    path = os.path.join(output_dir, f"confusion_matrix_model_{i}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Matriz de Confusao - Modelo {i}\n")
        f.write("=" * 32 + "\n\n")
        f.write("Formato: linhas=Real, colunas=Predito\n\n")
        f.write(f"{'':>12}{labels[0]:>12}{labels[1]:>12}\n")
        f.write(f"{labels[0]:>12}{int(cm[0, 0]):>12}{int(cm[0, 1]):>12}\n")
        f.write(f"{labels[1]:>12}{int(cm[1, 0]):>12}{int(cm[1, 1]):>12}\n")


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
    plt.savefig(os.path.join(output_dir, f"loss_curve_model_{model_idx}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def calculate_metrics(tp, tn, fp, fn, y_true, y_pred):
    """Calculate evaluation metrics from confusion matrix counts."""
    mcc = matthews_corrcoef(y_true, y_pred)
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    f1 = f1_score(y_true, y_pred, pos_label=Config.METRICS_CONFIG["fall_class"], zero_division=0)

    return {
        "MCC": mcc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "F1": f1,
    }


def record_metrics(metrics, tp, tn, fp, fn, i, output_dir):
    """Per-model metrics CSV output is intentionally disabled."""
    _ = (metrics, tp, tn, fp, fn, i, output_dir)


def save_classification_report(y_pred, y_true, output_dir, i):
    """Persist sklearn classification report as text."""
    classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(os.path.join(output_dir, f"classification_report_model_{i}.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))


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

        model = create_model_fn(best_params, input_shape, num_labels)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

        class_counts = np.bincount(y_tr, minlength=num_labels)
        class_counts = np.maximum(class_counts, 1)
        class_weights = len(y_tr) / (num_labels * class_counts.astype(float))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion_weighted = torch.nn.CrossEntropyLoss(weight=weight_tensor)

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
            criterion_weighted,
            device,
            epochs=epochs,
            early_stopping=False,
            patience=5,
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
        mcc = matthews_corrcoef(y_true_final, y_preds)
        f1 = f1_score(y_true_final, y_preds, average="macro")
        acc = accuracy_score(y_true_final, y_preds)
        weighted_train_loss_mean = float(np.mean(train_losses))
        weighted_val_loss_mean = float(np.mean(val_losses))

        results.append(
            {
                "Fraction": frac,
                "Num_Groups": int(len(np.unique(groups_subset))),
                "MCC": mcc,
                "F1": f1,
                "Accuracy": acc,
                "Weighted_Train_Loss": weighted_train_loss_mean,
                "Weighted_Val_Loss": weighted_val_loss_mean,
                "Train_Loss": weighted_train_loss_mean,
                "Val_Loss": weighted_val_loss_mean,
            }
        )
        print(
            f"MCC: {mcc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | "
            f"W-Train Loss: {weighted_train_loss_mean:.4f} | W-Val Loss: {weighted_val_loss_mean:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "learning_curve_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metricas da curva de aprendizado salvas em: {csv_path}")

    plt.figure(figsize=(10, 7))
    xvals = df["Num_Groups"] if "Num_Groups" in df.columns else (df["Fraction"] * 100)
    plt.plot(xvals, df["MCC"], marker="o", label="MCC")
    plt.plot(xvals, df["F1"], marker="o", label="F1-score")
    plt.plot(xvals, df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(xvals, df["Weighted_Train_Loss"], marker="o", label="Weighted Train Loss")
    plt.plot(xvals, df["Weighted_Val_Loss"], marker="o", label="Weighted Val Loss")
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
