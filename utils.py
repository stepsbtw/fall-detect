import torch
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F
import optuna
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import os
import itertools
import csv
from neural_networks import CNN1DNet, MLPNet, LSTMNet
import optuna.visualization as vis
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import json

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=25, early_stopping=False, patience=5):
    model.to(device, non_blocking=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    avg_train_losses, avg_val_losses = [], []

    for epoch in range(epochs):
        print(f"\n[{epoch}/{epochs}]")
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_train_losses.append(avg_train_loss)

        # Validação
        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        avg_val_losses.append(avg_val_loss)

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
        # Reavaliar após restaurar melhores pesos
        val_losses, y_pred, y_true = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                y_true.extend(yb.cpu().numpy())
        avg_val_losses[-1] = np.mean(val_losses)  # atualiza último val loss

    return y_pred, y_true, avg_val_losses, avg_train_losses

def objective(trial, input_shape_dict, X_trainval, y_trainval, output_dir, num_labels, device, restrict_model_type=None):
    print(f"\nIniciando Trial #{trial.number}\n")

    model_type = restrict_model_type if restrict_model_type else trial.suggest_categorical("model_type", ["CNN1D", "MLP", "LSTM"])
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    decision_threshold = trial.suggest_float('decision_threshold', 0.5, 0.9, step=0.1)

    mcc_scores = []
    all_train_losses = []
    all_val_losses = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval.argmax(axis=1) if len(y_trainval.shape) > 1 else y_trainval)):
        print(f"\n Fold {fold_idx + 1}/{skf.get_n_splits()} ({model_type})")

        torch.manual_seed(42 + fold_idx)
        torch.cuda.manual_seed_all(42 + fold_idx)
        np.random.seed(42 + fold_idx)
        random.seed(42 + fold_idx)

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        batch_size = 32

        if model_type == "CNN1D":
            filter_size = trial.suggest_int("filter_size", 16, 128, log=True)
            kernel_size = trial.suggest_int("kernel_size", 3, 7)
            num_layers = trial.suggest_int("num_layers", 2, 4)
            num_dense = trial.suggest_int("num_dense_layers", 1, 2)
            dense_neurons = trial.suggest_int("dense_neurons", 64, 512, log=True)

            # Prune se convolução reduz demais
            max_seq_len = input_shape_dict["CNN1D"][0]
            reduced_seq_len = max_seq_len // (2 ** num_layers)
            if reduced_seq_len <= kernel_size:
                raise optuna.exceptions.TrialPruned()
    
            model = CNN1DNet(input_shape_dict["CNN1D"], filter_size, kernel_size, num_layers, num_dense, dense_neurons, dropout, num_labels)

        elif model_type == "MLP":
            num_layers = trial.suggest_int("num_layers", 1, 4)
    
            max_dense = min(1024, max(64, input_shape_dict["MLP"] // 4))  # garante que max_dense >= 64
            dense_neurons = trial.suggest_int("dense_neurons", 64, max_dense, log=True)
            model = MLPNet(input_dim=input_shape_dict["MLP"], num_layers=num_layers, dense_neurons=dense_neurons, dropout=dropout, number_of_labels=num_labels)

        elif model_type == "LSTM":
            hidden_dim = trial.suggest_int("hidden_dim", 64, 256, log=True)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            model = LSTMNet(input_dim=input_shape_dict["LSTM"][1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, number_of_labels=num_labels)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train, dtype=torch.long)
            ),
            batch_size=batch_size, shuffle=True, pin_memory=True, #num_workers=6
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val, dtype=torch.long)
            ),
            batch_size=batch_size, pin_memory=True, #num_workers=6
        )

        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, val_loader, optimizer, criterion, device,
            epochs=25, early_stopping=True, patience=5
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        fold_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(fold_dir, exist_ok=True)

        plot_loss_curve(train_losses, val_losses, fold_dir, f"{trial.number}fold{fold_idx + 1}")

        save_results(
            model=model,
            val_loader=val_loader,
            y_val_onehot=y_val,
            number_of_labels=num_labels,
            i=f"{trial.number}fold{fold_idx + 1}",
            decision_threshold=decision_threshold,
            output_dir=fold_dir,
            device=device
        )

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

        mcc_scores.append(mcc)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        del model
        del optimizer
        torch.cuda.empty_cache()

    mean_mcc = np.mean(mcc_scores)
    print(f"Trial {trial.number} - Média MCC: {mean_mcc:.4f}")

    trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    summary = {
        "trial_number": trial.number,
        "model_type": model_type,
        "params": {
            "dropout": dropout,
            "learning_rate": learning_rate,
            "decision_threshold": decision_threshold
        },
        "mean_mcc": float(mean_mcc),
        "mcc_scores": mcc_scores
    }

    if model_type == "CNN1D":
        summary["params"].update({
            "filter_size": filter_size,
            "kernel_size": kernel_size,
            "num_layers": num_layers,
            "num_dense_layers": num_dense,
            "dense_neurons": dense_neurons
        })
    elif model_type == "MLP":
        summary["params"].update({
            "num_layers": num_layers,
            "dense_neurons": dense_neurons
        })
    elif model_type == "LSTM":
        summary["params"].update({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers
        })

    with open(os.path.join(trial_dir, "trial_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return mean_mcc

def run_optuna(input_shape_dict, X_trainval, y_trainval, output_dir, num_labels, device, study_name, restrict_model_type=None):

    db_path = os.path.join(output_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"

    # Tenta carregar o estudo se já existe, senão cria novo
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Estudo existente carregado de: {db_path}")
    except KeyError:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_url,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True
        )
        print(f"Novo estudo criado e salvo em: {db_path}")

    # Rodar otimização com Cross-Validation
    study.optimize(lambda trial: objective(
        trial,
        input_shape_dict,
        X_trainval,
        y_trainval,
        output_dir,
        num_labels,
        device,
        restrict_model_type
    ), n_trials=20, n_jobs=1) # n_trials = 30 original

    print("Melhor MCC:", study.best_value)
    print("Melhores hiperparâmetros:", study.best_params)

    # Salvar resultados
    os.makedirs(output_dir, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)

    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    # Importância dos hiperparâmetros
    try:
        fig = vis.plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, "param_importance.png"))
    except Exception as e:
        print(f"Could not save importance plot: {e}")

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
            xb = xb.to(device, non_blocking=True)
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
    else:
        # Para multiclasse, ainda salva MCC e F1
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        with open(os.path.join(output_dir, f"metrics_model_{i}.txt"), "w") as f:
            f.write(f"MCC: {mcc:.4f}\n")
            f.write(f"F1-score (macro): {f1:.4f}\n")

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

def plot_loss_curve(train_losses, val_losses, output_dir, model_idx):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Model {model_idx}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_model_{model_idx}.png"))
    plt.close()