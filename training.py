# Criado por Rodrigo Parracho - https://github.com/RodrigoKasama

import os
import torch
import torch.nn as nn
import numpy as np
import time
from training_imports import *
from NeuralArchitectures import CustomMLP, CNN1D


# Hiperparams definidos de forma arbitrária -> Impactam diretamente na potência e no tempo de treinamento


def fit(epochs, lr, model, train_dl, val_dl, criterion, opt_func=torch.optim.Adam):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_dl:

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())

            # calculate the loss
            loss = criterion(output.squeeze(), target.float())

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, target in val_dl:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data.float())
                # calculate the loss
                loss = criterion(output.squeeze(), target.float())
                # record validation loss
                valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch

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
        print(f"Size: {len(y_data)}")
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
    pass


if __name__ == "__main__":

    debug = True
    timestamp = str(int(time.time()))
    current_directory = os.path.dirname(__file__)

    position, label_type, scenario, neural_network_type, n_conv_layers, num_dense_layers = parse_input()

    neural_network_results_dir = create_result_dir(
        current_directory, neural_network_type, position)

    # Path dos datasets e targets. position -> labels_and_data/data/{position}/filename.npy
    data_dir = os.path.join(
        current_directory, "labels_and_data", "data", position)
    label_dir = os.path.join(
        current_directory, "labels_and_data", "labels", position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test, = collect_datasets_from_input(
        position, label_type, scenario, label_dir, data_dir)

    print(show_datasets_info(X_train, y_train, X_val, y_val, X_test, y_test))

    # TODO: Aqui poderia ser feito um porcesso de sintetização de dados de queda

    # Formação dos batches a partir dos datasets
    train_dl, val_dl, test_dl = generate_batches(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # Definição da Rede Neural
    model = None
    # Taxa de Aprendizagem
    learning_rate = 1e-3

    if neural_network_type == "MLP":
        model = CustomMLP(input_shape)
    else:

        first_conv_layer_size = 25
        first_dense_layer_size = 6000
        model = CNN1D(
            # Comprimento das janela
            input_shape=input_shape,
            # "n_conv_layers" Sessões de convolução que duplica o nº de canais a partir da 2ª camada
            n_conv_layers=n_conv_layers,
            first_conv_layer_size=first_conv_layer_size,
            # "n_conv_layers" Sessões de camadas densas que reduzem em 30% o nº de canais a partir da 2ª camada
            num_dense_layers=num_dense_layers,
            first_dense_layer_size=first_dense_layer_size,
            num_labels=num_labels  # A depender de uma classificação binária ou multiclasse
        )

    if model is None:
        raise Exception("Por algum motivo o modelo não pode ser construido")

    print("-" * 90)
    print(model)
    print("-" * 90)

    # Nº de Epochs
    epochs = 10
    # Função de Custo
    loss_fn = nn.BCEWithLogitsLoss()

    # Treinamento própriamente dito
    model, train_loss, valid_loss = fit(
        epochs, learning_rate, model, train_dl, val_dl, loss_fn)
    print("-" * 90)

    # TODO: Implementar rotina de armazenar de: gráfico de loss, métricas no conjunto de teste e modelo treinado.

    # Plotagem do gráfico de perda
    category = "bin" if num_labels == 2 else "multi"

    filename = f"{timestamp}_{neural_network_type}_{category}_{str(learning_rate)}_{position}_{scenario}"

    save_loss_curve(train_loss, valid_loss,
                    neural_network_results_dir, f"{filename}.png")

    print(
        f"Gráfico de Perda gerado com sucesso.(Verifique o diretório {neural_network_results_dir})")

    test_report = get_class_report(model, test_dl)
    print("Relatório de classificação no dataset de treino:")
    print(test_report)

    save_model(model, os.path.join("models", f"{filename}.model"))
