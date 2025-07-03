# Criado por Rodrigo Parracho - https://github.com/RodrigoKasama
# Adaptado por Caio Passos - https://github.com/stepsbtw

import os
import torch
import torch.nn as nn
import numpy as np
import time
from training_imports import *
from NeuralArchitectures import CustomMLP, CNN1D
import json

def fit(epochs, lr, model, train_dl, val_dl, criterion, opt_func=torch.optim.Adam):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        for data, target in train_dl:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data.float())
                loss = criterion(output.squeeze(), target.float())
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        epoch_len = len(str(epochs))

        print(f"[{epoch+1:>{epoch_len}}/{epochs:>{epoch_len}}] " +
              f"train_loss: {train_loss:.4f} " +
              f"valid_loss: {valid_loss:.4f}")

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        train_losses = []
        valid_losses = []

    return model, avg_train_losses, avg_valid_losses


def save_model(model, filepath):
    torch.save(model, filepath)


def show_datasets_info(X_train, y_train, X_val, y_val, X_test, y_test):
    info = ""

    def format_distribution(y_data: torch.Tensor):
        negative_class = torch.sum(y_data == 0).item()
        negative_percentage = negative_class * 100 / len(y_data)
        positive_class = torch.sum(y_data == 1).item()
        positive_percentage = positive_class * 100 / len(y_data)
        return f"{positive_class}({int(positive_percentage)}%)-{negative_class}({int(negative_percentage)}%)"

    info += "-" * 90 + "\n"
    info += "Datasets | Labels\n"
    info += "-" * 90 + "\n"
    info += f"Treinamento: {X_train.shape} | {y_train.shape} | {format_distribution(y_train)}\n"
    info += f"Validação: {X_val.shape} | {y_val.shape} | {format_distribution(y_val)}\n"
    info += f"Teste: {X_test.shape} | {y_test.shape} | {format_distribution(y_test)}"
    return info


def export_result(scenario, neural_network_type, position, test_report):
    filename = f"results/{scenario}_{neural_network_type}_{position}.json"
    with open(filename, "w") as f:
        json.dump(test_report, f, indent=4)


if __name__ == "__main__":
    export = False
    timestamp = str(int(time.time()))
    current_directory = os.path.dirname(__file__)

    position, label_type, scenario, neural_network_type, n_conv_layers, num_dense_layers, epochs, learning_rate, export = parse_input()

    neural_network_results_dir = create_result_dir(
        current_directory, neural_network_type, position)

    data_dir = os.path.join(current_directory, "labels_and_data", "data", position)
    label_dir = os.path.join(current_directory, "labels_and_data", "labels", position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test = collect_datasets_from_input(
        position, label_type, scenario, label_dir, data_dir)

    print(show_datasets_info(X_train, y_train, X_val, y_val, X_test, y_test))

    train_dl, val_dl, test_dl = generate_batches(
        X_train, y_train, X_val, y_val, X_test, y_test)

    model = None
    if neural_network_type == "MLP":
        model = CustomMLP(input_shape)
    else:
        first_conv_layer_size = 25
        first_dense_layer_size = 6000
        model = CNN1D(
            input_shape=input_shape,
            n_conv_layers=n_conv_layers,
            first_conv_layer_size=first_conv_layer_size,
            num_dense_layers=num_dense_layers,
            first_dense_layer_size=first_dense_layer_size,
            num_labels=num_labels
        )

    if model is None:
        raise Exception("Por algum motivo o modelo não pode ser construido")

    # Suporte a múltiplas GPUs
    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)

    print("-" * 90)
    print(model)
    print("-" * 90)

    loss_fn = nn.BCEWithLogitsLoss()
    model, train_loss, valid_loss = fit(epochs, learning_rate, model, train_dl, val_dl, loss_fn)
    print("-" * 90)

    category = "bin" if num_labels == 2 else "multi"
    filename = f"{timestamp}_{neural_network_type}_{category}_{str(learning_rate)}_{position}_{scenario}"

    if export:
        save_loss_curve(train_loss, valid_loss, neural_network_results_dir, f"{filename}.png")
        print(f"Gráfico de Perda gerado com sucesso.(Verifique o diretório {neural_network_results_dir})")
        print("-" * 90)

    test_report, dict_test_report, matri_conf = get_class_report(model, test_dl)
    print("Relatório de classificação no dataset de treino:")
    print(test_report)

    if export:
        print("-" * 90)
        export_result(scenario, neural_network_type, position, dict_test_report)
        print("Resultados exportado com sucesso.")
        save_model(model, os.path.join("models", f"{filename}.model"))
        print("Modelo salvo com sucesso.")
