# model_utils.py
from neural_architectures import CustomMLP, CNN1D, CustomLSTM
from training_imports import collect_datasets_from_input, generate_batches
import torch

def build_model_from_trial(trial, input_shape, num_labels):
    model_type = trial.suggest_categorical("model_type", ["MLP", "CNN1D", "LSTM"])

    if model_type == "MLP":
        asc = trial.suggest_int("asc", 1, 3)
        desc = trial.suggest_int("desc", 1, 3)
        model = CustomMLP(input_shape, num_asc_layers=asc, num_desc_layers=desc, num_labels=num_labels)

    elif model_type == "CNN1D":
        n_conv = trial.suggest_int("n_conv", 1, 4)
        conv_size = trial.suggest_int("conv_size", 16, 64)
        n_dense = trial.suggest_int("n_dense", 1, 3)
        dense_size = trial.suggest_int("dense_size", 128, 512)
        model = CNN1D(input_shape, n_conv, conv_size, n_dense, dense_size, num_labels=num_labels)

    elif model_type == "LSTM":
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
        dense_layers = trial.suggest_int("num_dense_layers", 1, 3)
        dense_size = trial.suggest_int("first_dense_size", 64, 512)
        bidir = trial.suggest_categorical("bidirectional", [True, False])
        model = CustomLSTM(input_shape, hidden_size, lstm_layers, dense_layers, dense_size, num_labels, bidir)

    return model, model_type


def prepare_datasets(position, label_type, scenario, label_dir, data_dir):
    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test = collect_datasets_from_input(
        position, label_type, scenario, label_dir, data_dir)

    if num_labels > 1:
        y_train = (y_train == 1).float()
        y_val = (y_val == 1).float()
        y_test = (y_test == 1).float()
        num_labels = 1

    train_dl, val_dl, test_dl = generate_batches(X_train, y_train, X_val, y_val, X_test, y_test)
    return input_shape, num_labels, train_dl, val_dl, test_dl
