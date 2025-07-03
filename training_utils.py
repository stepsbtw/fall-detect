# training_utils.py
import os
import torch
import numpy as np
from neural_architectures import CustomMLP, CNN1D, CustomLSTM
import torch.nn as nn
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fit(epochs, lr, model, train_dl, val_dl, criterion, opt_func=torch.optim.Adam,
        patience=5, checkpoint_path=None, trial=None):
    optimizer = opt_func(model.parameters(), lr)
    avg_train_losses = []
    avg_valid_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for data, target in train_dl:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output.view(-1), target.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device), target.to(device)
                output = model(data.float())
                loss = criterion(output.view(-1), target.float())
                valid_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(f"[{epoch+1:>{len(str(epochs))}}/{epochs}] train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f}")

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

        if trial:
            trial.report(valid_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Carregando melhor modelo do checkpoint.")
        model.load_state_dict(torch.load(checkpoint_path))
    elif best_model_state:
        print("Carregando melhor modelo salvo em memória.")
        model.load_state_dict(best_model_state)

    return model, avg_train_losses, avg_valid_losses


def save_model(model, filepath):
    torch.save(model, filepath)


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
