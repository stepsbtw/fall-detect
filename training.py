# Criado por Rodrigo Parracho - https://github.com/RodrigoKasama

import os
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from training_imports import *
from NeuralArchitectures import CustomMLP, CNN1D

BATCH_SIZE = 32


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


def generate_batches(X_train, y_train, X_val, y_val):
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_dl, val_dl


if __name__ == "__main__":

    debug = True

    current_directory = os.path.dirname(__file__)

    # Nº de Epochs
    epochs = 10
    # Taxa de Aprendizagem
    learning_rate = 0.01

    # Função de Custo - BCELoss()??
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    # Hiperparams definidos de forma arbitrária -> Impactam diretamente na potência e no tempo de treinamento
    first_conv_layer_size = 25
    first_dense_layer_size = 6000

    position, label_type, scenario, neural_network_type, n_conv_layers, num_dense_layers = parse_input()

    output_dir = os.path.join(current_directory, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    neural_network_results_dir = os.path.join(output_dir, neural_network_type)
    if neural_network_type == "CNN1D":
        neural_network_results_dir = os.path.join(
            neural_network_results_dir, position)

    if not os.path.exists(neural_network_results_dir):
        os.makedirs(neural_network_results_dir)

    # Diretórios dos datasets e tragets. position -> labels_and_data/data/{position}/filename.npy
    data_dir = os.path.join(
        current_directory, "labels_and_data", "data", position)
    label_dir = os.path.join(
        current_directory, "labels_and_data", "labels", position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test, = collect_datasets_from_input(
        position, label_type, scenario, neural_network_type, label_dir, data_dir)

    # --------------------------------------------------------------------------------------------------------------------
    # Coleta de uma pequena amostra do dataset para treinamento parcial
    i_train = -1
    i_val = -1
    X_train, y_train = X_train[:i_train], y_train[:i_train]
    X_val, y_val = X_val[:i_val], y_val[:i_val]
    # --------------------------------------------------------------------------------------------------------------------

    print("Datasets | Labels")
    print(f"Treinamento: {X_train.shape} | {y_train.shape}")
    print(f"Validação: {X_val.shape} | {y_val.shape}")
    print(f"Teste: {X_test.shape} | {y_test.shape}")

    # Formação dos batches a partir dos datasets
    train_dl, val_dl = generate_batches(X_train, y_train, X_val, y_val)

    # Definição da Rede Neural
    # model = CNN1D(
    #     # Comprimento das features
    #     # Representa 5s de registros de movimentos (450 para os punhos, 1020 para o peito)
    #     input_shape=input_shape,

    #     # "n_conv_layers" Sessões de convolução que duplica o nº de canais a partir da 2ª camada
    #     n_conv_layers=n_conv_layers,
    #     first_conv_layer_size=first_conv_layer_size,

    #     # "n_conv_layers" Sessões de camadas densas que reduzem em 30% o nº de canais a partir da 2ª camada
    #     num_dense_layers=num_dense_layers,
    #     first_dense_layer_size=first_dense_layer_size,
    #     num_labels=num_labels  # A depender de uma classificação binária ou multiclasse
    # )

    model = CustomMLP(input_shape)

    # DEBUG
    if debug:
        print("-"*90)
        print(model)
        print("-"*90)

    # exit()
    # Treinamento própriamente dito
    model, train_loss, valid_loss = fit(epochs, learning_rate, model,
                                        train_dl, val_dl, loss_fn)

    # Plotagem do gráfico de perda
    category = "bin" if num_labels == 2 else "multi"

    plot_filename = f"{neural_network_type}_{category}_{str(learning_rate)}_{position}_{scenario}.png"

    plot_loss_curve(train_loss, valid_loss,
                    neural_network_results_dir, plot_filename)

    print(f"Gráfico de Perda gerado com sucesso. (Verifique o diretório {neural_network_results_dir})")
    
    
    
