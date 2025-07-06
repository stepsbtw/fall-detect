import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F
import torch
import optuna
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import os
import itertools
import csv
from testing.neural_networks import CNN1DNet, MLPNet, LSTMNet
import optuna.visualization as vis
import pandas as pd

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=25, early_stopping=False, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    if early_stopping and best_model_state:
        model.load_state_dict(best_model_state)

    return y_pred, y_true, val_loss


def objective(trial, input_shape_dict, X_train, y_train, X_val, y_val, output_dir, num_labels, device):
    
    model_type = trial.suggest_categorical("model_type", ["CNN1D", "MLP", "LSTM"])

    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    decision_threshold = trial.suggest_float("decision_threshold", 0.5, 0.9)

    # Model-specific hyperparameters
    if model_type == "CNN1D":
        filter_size = trial.suggest_int("filter_size", 8, 128)
        kernel_size = trial.suggest_int("kernel_size", 2, 6)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_dense = trial.suggest_int("num_dense_layers", 1, 3)
        dense_units = trial.suggest_int("dense_units", 64, 256)
        model = CNN1DNet(input_shape_dict["CNN1D"], filter_size, kernel_size, num_layers, num_dense, dense_units, dropout, num_labels)

    elif model_type == "MLP":
        num_layers = trial.suggest_int("num_layers", 1, 5)
        dense_units = trial.suggest_int("dense_units", 64, 2048)
        model = MLPNet(input_dim=input_shape_dict["MLP"], num_layers=num_layers, dense_neurons=dense_units, dropout=dropout, number_of_labels=num_labels)

    elif model_type == "LSTM":
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        model = LSTMNet(input_dim=input_shape_dict["LSTM"][1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, number_of_labels=num_labels)


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

    best_mcc = -1
    for epoch in range(25):
        y_pred, y_true, _ = train(model, train_loader, val_loader, optimizer, criterion, device, epochs=1)

        # Metric
        if num_labels == 2:
            y_probs = []
            model.eval()
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                    y_probs.extend(probs)
            y_pred_thresh = (np.array(y_probs) >= decision_threshold).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred_thresh)
        else:
            mcc = matthews_corrcoef(y_true, y_pred)

        trial.report(mcc, step=epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch} with MCC={mcc:.4f}")
            raise optuna.exceptions.TrialPruned()

        best_mcc = max(best_mcc, mcc)

    return best_mcc


def run_optuna(input_shape_dict, X_train, y_train, X_val, y_val, output_dir, num_labels, device):

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    study.optimize(lambda trial: objective(
        trial, input_shape_dict, X_train, y_train, X_val, y_val, output_dir, num_labels, device
    ), n_trials=20)

    # Logging
    print("Melhor MCC:", study.best_value)
    print("Melhores hiperparâmetros:", study.best_params)

    # Save results to CSV
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    # Save param importance plot
    fig = vis.plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, "param_importance.png"))

    return study


def save_results(model, val_loader, y_val_onehot, number_of_labels, i, decision_threshold, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Salvar modelo
    model_path = os.path.join(output_dir, f"model_{i}.pt")
    torch.save(model.state_dict(), model_path)

    # 2. Inferência
    model.eval()
    y_probs = []
    y_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(yb.numpy())

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    y_pred = (y_probs[:, 1] >= decision_threshold).astype(int) if number_of_labels == 2 else np.argmax(y_probs, axis=1)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if number_of_labels == 2 else (0, 0, 0, 0)
    plot_confusion_matrix(cm, number_of_labels, output_dir, i)

    # 4. Classification report
    save_classification_report(y_pred, y_true, number_of_labels, output_dir, i)

    # 5. ROC curve (apenas binário)
    if number_of_labels == 2:
        plot_roc_curve(y_probs[:, 1], y_true, output_dir, i)

    # 6. Métricas
    if number_of_labels == 2:
        metrics = calculate_metrics(tp, tn, fp, fn, y_true, y_pred)
        record_metrics(metrics, tp, tn, fp, fn, i, output_dir)

def plot_confusion_matrix(cm, number_of_labels, output_dir, i):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(number_of_labels)
    plt.xticks(ticks)
    plt.yticks(ticks)

    thresh = cm.max() / 2.
    for r, c in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(c, r, format(cm[r, c], 'd'),
                 ha="center", color="white" if cm[r, c] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_model_{i}.png"))
    plt.close()

def save_classification_report(y_pred, y_true, number_of_labels, output_dir, i):
    report = classification_report(y_true, y_pred, target_names=[str(x) for x in range(number_of_labels)])
    with open(os.path.join(output_dir, f"classification_report_model_{i}.txt"), "w") as f:
        f.write(report)

def plot_roc_curve(y_score, y_true, output_dir, i):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"roc_curve_model_{i}.png"))
    plt.close()

def calculate_metrics(tp, tn, fp, fn, y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    return {
        "MCC": mcc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy
    }

def record_metrics(metrics, tp, tn, fp, fn, i, output_dir):
    path = os.path.join(output_dir, f"metrics_model_{i}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Model", "MCC", "Sensitivity", "Specificity", "Precision", "Accuracy", "tp", "tn", "fp", "fn"
        ])
        writer.writeheader()
        writer.writerow({
            "Model": i,
            "MCC": metrics["MCC"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Precision": metrics["Precision"],
            "Accuracy": metrics["Accuracy"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        })
