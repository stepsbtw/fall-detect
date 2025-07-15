"""
- Treinamento e validação
- Otimização com Optuna
- Salvamento e análise
- Visualizações
- Métricas
- Shap Values
"""

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
import seaborn as sns
from config import Config

# =============================================================================
# TREINAMENTO E VALIDAÇÃO
# =============================================================================

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=25, early_stopping=False, patience=5, scaler=None, trial=None):
    """
    Treina modelo com early stopping, mixed precision e pruning opcional
    """
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
            
            # Mixed precision training se scaler disponível
            if scaler is not None:
                with torch.amp.autocast('cuda'):
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

        # Validação
        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                # Mixed precision inference se scaler disponível
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
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

        # Pruning intermediário (se trial fornecido)
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early stopping
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

    # Restaurar melhor modelo se early stopping foi usado
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

# =============================================================================
# OTIMIZAÇÃO COM OPTUNA
# =============================================================================

def objective(trial, input_shape_dict, X_trainval, y_trainval, output_dir, num_labels, device, restrict_model_type=None):
    """
    Função objetivo para otimização com Optuna
    """
    print(f"\nIniciando Trial #{trial.number}\n")

    # Sugerir hiperparâmetros
    model_type = restrict_model_type if restrict_model_type else trial.suggest_categorical("model_type", ["CNN1D", "MLP", "LSTM"])
    dropout = trial.suggest_float('dropout', Config.METRICS_CONFIG['dropout_range'][0], Config.METRICS_CONFIG['dropout_range'][1], step=Config.METRICS_CONFIG['dropout_step'])
    learning_rate = trial.suggest_float('learning_rate', Config.OPTIMIZER_CONFIG['lr_range'][0], Config.OPTIMIZER_CONFIG['lr_range'][1], log=Config.OPTIMIZER_CONFIG['lr_log'])
    decision_threshold = trial.suggest_float('decision_threshold', Config.METRICS_CONFIG['decision_threshold_range'][0], Config.METRICS_CONFIG['decision_threshold_range'][1], step=Config.METRICS_CONFIG['decision_threshold_step'])

    mcc_scores = []
    all_train_losses = []
    all_val_losses = []

    # Cross-validation usando configurações do Config
    skf = StratifiedKFold(n_splits=Config.CV_CONFIG['n_splits'], shuffle=Config.CV_CONFIG['shuffle'], random_state=Config.CV_CONFIG['random_state'])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval.argmax(axis=1) if len(y_trainval.shape) > 1 else y_trainval)):
        print(f"\n Fold {fold_idx + 1}/{skf.get_n_splits()} ({model_type})")

        # Definir seeds para reprodutibilidade
        Config.set_seed(Config.SEED + fold_idx)

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        batch_size = Config.TRAINING_CONFIG['batch_size']

        # Criar modelo baseado no tipo
        if model_type == "CNN1D":
            cnn_config = Config.MODEL_CONFIGS['CNN1D']
            filter_size = trial.suggest_int("filter_size", cnn_config['filter_size_range'][0], cnn_config['filter_size_range'][1], log=True)
            kernel_size = trial.suggest_int("kernel_size", cnn_config['kernel_size_range'][0], cnn_config['kernel_size_range'][1])
            num_layers = trial.suggest_int("num_layers", cnn_config['num_layers_range'][0], cnn_config['num_layers_range'][1])
            num_dense = trial.suggest_int("num_dense_layers", cnn_config['num_dense_layers_range'][0], cnn_config['num_dense_layers_range'][1])
            dense_neurons = trial.suggest_int("dense_neurons", cnn_config['dense_neurons_range'][0], cnn_config['dense_neurons_range'][1], log=True)

            # Prune se convolução reduz demais
            max_seq_len = input_shape_dict["CNN1D"][0]
            reduced_seq_len = max_seq_len // (2 ** num_layers)
            if reduced_seq_len <= kernel_size:
                raise optuna.exceptions.TrialPruned()
    
            model = CNN1DNet(input_shape_dict["CNN1D"], filter_size, kernel_size, num_layers, num_dense, dense_neurons, dropout, num_labels)

        elif model_type == "MLP":
            mlp_config = Config.MODEL_CONFIGS['MLP']
            num_layers = trial.suggest_int("num_layers", mlp_config['num_layers_range'][0], mlp_config['num_layers_range'][1])
    
            max_dense = min(mlp_config['dense_neurons_range'][1], max(mlp_config['dense_neurons_range'][0], input_shape_dict["MLP"] // 4))
            dense_neurons = trial.suggest_int("dense_neurons", mlp_config['dense_neurons_range'][0], max_dense, log=True)
            model = MLPNet(input_dim=input_shape_dict["MLP"], num_layers=num_layers, dense_neurons=dense_neurons, dropout=dropout, number_of_labels=num_labels)

        elif model_type == "LSTM":
            lstm_config = Config.MODEL_CONFIGS['LSTM']
            hidden_dim = trial.suggest_int("hidden_dim", lstm_config['hidden_dim_range'][0], lstm_config['hidden_dim_range'][1], log=True)
            num_layers = trial.suggest_int("num_layers", lstm_config['num_layers_range'][0], lstm_config['num_layers_range'][1])
            model = LSTMNet(input_dim=input_shape_dict["LSTM"][1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, number_of_labels=num_labels)

        model.to(device)
        
        # Usar múltiplas GPUs se disponível
        if torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
            model = torch.nn.DataParallel(model)
            # Ajustar batch size para múltiplas GPUs
            batch_size = batch_size * torch.cuda.device_count()
            print(f"Batch size ajustado para {batch_size} (batch_size * num_gpus)")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # Preparar data loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train, dtype=torch.long)
            ),
            batch_size=batch_size, 
            shuffle=Config.TRAINING_CONFIG['shuffle'], 
            pin_memory=Config.TRAINING_CONFIG['pin_memory'], 
            num_workers=Config.TRAINING_CONFIG['num_workers']
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val, dtype=torch.long)
            ),
            batch_size=batch_size, 
            pin_memory=Config.TRAINING_CONFIG['pin_memory'], 
            num_workers=Config.TRAINING_CONFIG['num_workers']
        )

        # Treinar modelo com pruning intermediário
        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, val_loader, optimizer, criterion, device,
            epochs=Config.TRAINING_CONFIG['epochs'], 
            early_stopping=Config.TRAINING_CONFIG['early_stopping'], 
            patience=Config.TRAINING_CONFIG['patience'], 
            trial=trial
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Salvar resultados do fold
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

        # Calcular MCC
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

        # Pruning check após cada fold
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        del model
        del optimizer
        torch.cuda.empty_cache()

    mean_mcc = np.mean(mcc_scores)
    print(f"Trial {trial.number} - Média MCC: {mean_mcc:.4f}")

    # Salvar resumo do trial
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
    """
    Executa otimização com Optuna
    """
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
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            ),
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
    ), n_trials=Config.OPTUNA_CONFIG['n_trials'], n_jobs=Config.OPTUNA_CONFIG['n_jobs'])

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

# =============================================================================
# SALVAMENTO E ANÁLISE DE RESULTADOS
# =============================================================================

def save_results(model, val_loader, y_val_onehot, number_of_labels, i, decision_threshold, output_dir, device):
    """
    Salva resultados completos do modelo
    """
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

# =============================================================================
# VISUALIZAÇÕES
# =============================================================================

def plot_confusion_matrix(cm, number_of_labels, output_dir, i):
    """
    Plota matriz de confusão
    """
    plt.figure(figsize=(8, 6))
    if number_of_labels == 2:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Não Queda', 'Queda'], 
                    yticklabels=['Não Queda', 'Queda'])
        plt.title(f'Matriz de Confusão - Modelo {i}')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predito')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - Modelo {i}')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predito')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_model_{i}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_score, y_true, output_dir, i):
    """
    Plota curva ROC
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Modelo {i}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_model_{i}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curve(train_losses, val_losses, output_dir, model_idx):
    """
    Plota curva de loss
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Modelo {model_idx}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_curve_model_{model_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# MÉTRICAS E AVALIAÇÃO
# =============================================================================

def calculate_metrics(tp, tn, fp, fn, y_true, y_pred):
    """
    Calcula métricas de avaliação
    """
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
    """
    Salva métricas em CSV
    """
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

def save_classification_report(y_pred, y_true, number_of_labels, output_dir, i):
    """
    Salva relatório de classificação
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(os.path.join(output_dir, f'classification_report_model_{i}.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred))


def plot_learning_curve(create_model_fn, X_full, y_full, X_test, y_test, input_shape, num_labels, best_params, device, output_dir, fractions=None, epochs=10, seed=42):
    """
    Gera a curva de aprendizado variando a fração do dataset de treino.
    Agora salva MCC, F1-score, Accuracy, Loss de treino e validação para cada fração.
    """
    from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
    import pandas as pd
    if fractions is None:
        fractions = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rng = np.random.RandomState(seed)
    results = []

    print(f"\n{'='*50}")
    print("INICIANDO GERAÇÃO DA LEARNING CURVE")
    print(f"{'='*50}")

    for frac in fractions:
        size = int(len(X_full) * frac)
        idx = rng.choice(len(X_full), size, replace=False)
        X_subset = X_full[idx]
        y_subset = y_full[idx]

        print(f"\nTreinando com {size} amostras ({int(frac*100)}% do dataset)")

        # Criar modelo
        model = create_model_fn(best_params, input_shape, num_labels)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
        criterion = torch.nn.CrossEntropyLoss()

        batch_size =  Config.TRAINING_CONFIG.get('batch_size', 32)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_subset, dtype=torch.float32), torch.tensor(y_subset, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False
        )

        # Treinamento curto
        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, test_loader, optimizer, criterion, device,
            epochs=epochs,
            early_stopping=False,
            patience=5,
            scaler=None
        )

        # Avaliação final
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
        train_loss_mean = np.mean(train_losses)
        val_loss_mean = np.mean(val_losses)

        print(f"MCC: {mcc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f}")
        results.append({
            "Fraction": frac,
            "MCC": mcc,
            "F1": f1,
            "Accuracy": acc,
            "Train_Loss": train_loss_mean,
            "Val_Loss": val_loss_mean
        })

    # Salvar CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "learning_curve_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Métricas da curva de aprendizado salvas em: {csv_path}")

    # Plotar curvas
    plt.figure(figsize=(10, 7))
    plt.plot(df["Fraction"]*100, df["MCC"], marker='o', label="MCC")
    plt.plot(df["Fraction"]*100, df["F1"], marker='o', label="F1-score")
    plt.plot(df["Fraction"]*100, df["Accuracy"], marker='o', label="Accuracy")
    plt.plot(df["Fraction"]*100, df["Train_Loss"], marker='o', label="Train Loss")
    plt.plot(df["Fraction"]*100, df["Val_Loss"], marker='o', label="Val Loss")
    plt.xlabel("Porcentagem de Dados de Treino (%)")
    plt.ylabel("Valor da Métrica")
    plt.title("Curva de Aprendizado - MCC, F1, Accuracy, Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    lc_plot_path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(lc_plot_path, dpi=300)
    plt.close()
    print(f"Curva de aprendizado salva em: {lc_plot_path}")



from collections import OrderedDict

def load_model_state(model, path, device='cpu'):
    """
    Carrega um modelo PyTorch de forma robusta, removendo 'module.' se necessário.
    """
    state_dict = torch.load(path, map_location=device)

    # Se tiver o prefixo "module.", remove
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model

def load_hyperparameters(output_dir):
    """Carrega os melhores hiperparâmetros encontrados"""
    import os, json
    results_file = os.path.join(output_dir, "best_hyperparameters.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Arquivo de hiperparâmetros não encontrado: {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def load_test_data(output_dir):
    """Carrega os dados de teste salvos"""
    import os, numpy as np
    test_data_file = os.path.join(output_dir, "test_data.npz")
    if not os.path.exists(test_data_file):
        raise FileNotFoundError(f"Arquivo de dados de teste não encontrado: {test_data_file}")
    data = np.load(test_data_file)
    return data['X_test'], data['y_test']

def create_model(model_type, best_params, input_shape, num_labels):
    """Cria o modelo com os melhores hiperparâmetros"""
    from neural_networks import CNN1DNet, MLPNet, LSTMNet
    if model_type == "CNN1D":
        model = CNN1DNet(
            input_shape=input_shape,
            filter_size=best_params["filter_size"],
            kernel_size=best_params["kernel_size"],
            num_layers=best_params["num_layers"],
            num_dense_layers=best_params["num_dense_layers"],
            dense_neurons=best_params["dense_neurons"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    elif model_type == "MLP":
        model = MLPNet(
            input_dim=input_shape,
            num_layers=best_params["num_layers"],
            dense_neurons=best_params["dense_neurons"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    elif model_type == "LSTM":
        model = LSTMNet(
            input_dim=input_shape[1],
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    return model
