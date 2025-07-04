import torch
import numpy as np
from neural_architectures import CustomMLP, CNN1D, CustomLSTM
import optuna
import os

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    BinaryF1Score, BinaryMatthewsCorrCoef, BinaryAccuracy, BinaryAUROC
)

from utils_output import save_loss_curve


def fit(epochs, lr, model, train_dl, val_dl, criterion, opt_func=torch.optim.Adam,
        patience=5, checkpoint_path=None, trial=None, save_dir="optuna_results"):

    optimizer = opt_func(model.parameters(), lr)
    avg_train_losses = []
    avg_valid_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None


    device = next(model.parameters()).device

    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/trial_{trial.number}" if trial else "runs/default")

    # Torchmetrics
    f1_metric = BinaryF1Score().to(device, non_blocking=True)
    mcc_metric = BinaryMatthewsCorrCoef().to(device, non_blocking=True)
    acc_metric = BinaryAccuracy().to(device, non_blocking=True)
    auc_metric = BinaryAUROC().to(device, non_blocking=True)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for data, target in train_dl:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data.float()).squeeze()
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        all_outputs, all_targets = [], []

        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data.float()).squeeze()
                loss = criterion(output, target.float())
                valid_losses.append(loss.item())
                all_outputs.append(output)
                all_targets.append(target)

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # Metrics
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        probs = torch.sigmoid(all_outputs)

        threshold = getattr(model, 'decision_threshold', 0.5)
        preds = (probs >= threshold).int()

        f1 = f1_metric(preds, all_targets.int()).item()
        mcc = mcc_metric(preds, all_targets.int()).item()
        acc = acc_metric(preds, all_targets.int()).item()

        try:
            auc = auc_metric(probs, all_targets.int()).item()
        except ValueError:
            auc = float('nan')

        # Logs
        print(f"Epoch {epoch+1:>2}/{epochs} - Train: {train_loss:.4f} | Val: {valid_loss:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | AUC: {auc:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)
        writer.add_scalar("Metrics/MCC", mcc, epoch)
        writer.add_scalar("Metrics/Accuracy", acc, epoch)
        writer.add_scalar("Metrics/AUC", auc, epoch)

        # Early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
            else:
                best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Optuna pruning
        if trial:
            trial.report(valid_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    filename = f"loss_curve_trial_{trial.number if trial else 'default'}.png"
    os.makedirs(save_dir, exist_ok=True)
    save_loss_curve(avg_train_losses, avg_valid_losses, image_dir=save_dir, filename=filename)

    # Load best model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Carregando melhor modelo do checkpoint.")
        model.load_state_dict(torch.load(checkpoint_path))
    elif best_model_state:
        print("Carregando melhor modelo salvo em memória.")
        model.load_state_dict(best_model_state)

    writer.close()

    return model, avg_train_losses, avg_valid_losses



def build_model(args, input_shape, num_labels):
    if args.neural_network_type == "MLP":
        return CustomMLP(input_shape, num_labels=num_labels)

    elif args.neural_network_type == "CNN1D":
        return CNN1D(
            input_shape=input_shape,
            n_conv_layers=args.n_conv,
            first_conv_layer_size=25,
            num_dense_layers=args.n_dense,
            first_dense_layer_size=6000,
            num_labels=num_labels
        )

    elif args.neural_network_type == "LSTM":
        return CustomLSTM(
            input_shape=input_shape,
            hidden_size=args.hidden_size,
            num_lstm_layers=args.num_lstm_layers,
            num_dense_layers=args.num_dense_layers_lstm,
            first_dense_size=args.first_dense_size_lstm,
            num_labels=num_labels,
            bidirectional=args.bidirectional
        )

    raise ValueError(f"Tipo de rede desconhecida: {args.neural_network_type}")

def build_model_optuna(model_type, trial, input_shape, num_labels):
    if model_type == "MLP":
        num_layers = trial.suggest_int('num_layers', 1, 5)
        dense_neurons = trial.suggest_int('dense_neurons', 20, 4000, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01])
        decision_threshold = trial.suggest_float('decision_threshold', 0.5, 0.9,step=0.1)
        model = CustomMLP(input_shape, num_layers=num_layers, dense_neurons=dense_neurons, dropout=dropout, learning_rate=learning_rate, decision_threshold=decision_threshold, num_labels=num_labels)

    elif model_type == "CNN1D":
        filter_size = trial.suggest_int('filter_size', 8, 600, log=True)
        kernel_size = trial.suggest_int('kernel_size', 2, 6)
        num_layers = trial.suggest_int('num_layers', 2, 4)
        num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
        dense_neurons = trial.suggest_int('dense_neurons', 60, 320, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01])
        decision_threshold = trial.suggest_float('decision_threshold', 0.5, 0.9,step=0.1)
        model = CNN1D(input_shape, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, decision_threshold, num_labels=num_labels)

    elif model_type == "LSTM":
        hidden_size = trial.suggest_int("hidden_size", 64, 256)
        lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
        dense_layers = trial.suggest_int("num_dense_layers", 1, 3)
        dense_size = trial.suggest_int("first_dense_size", 128, 512)
        bidir = trial.suggest_categorical("bidirectional", [True, False])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01])
        decision_threshold = trial.suggest_float("decision_threshold", 0.5, 0.9, step=0.1)
        model = CustomLSTM(input_shape, hidden_size, lstm_layers, dense_layers, dense_size, num_labels, bidir, dropout, learning_rate, decision_threshold)

    return model

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False