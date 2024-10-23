# Criado por Rodrigo Parracho - https://github.com/RodrigoKasama

from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torch.nn as nn
import numpy as np
from training_imports import *

BATCH_SIZE = 32


class CNN1D(nn.Module):
    def __init__(self, input_shape, n_conv_layers, first_conv_layer_size,  num_dense_layers, first_dense_layer_size,  num_labels):
        super(CNN1D, self).__init__()

        # filter_size = 50
        # kernel_size = 5
        # num_layers = 3
        # num_dense_layers = 2
        # dense_neurons = 100

        # learning_rate = 0.0001
        # decision_threshold = 0.5

        self.conv_layer = nn.ModuleList()
        # Hiperparametros definidos de forma fixa
        self.kernel_size = 3
        self.dropout_rate = 0.3

        self.max_pool = 2
        last_layer_channels = 0
        dense_neurons = first_dense_layer_size
        dense_layer_droprate = 4

        # Para cada seq de (Conv1d + ReLU + MaxPool1d + Dropout)
        for i in range(n_conv_layers):

            # PARA CONV1D: Se pading = 0 e stride = 1 |-> [batch, j, k] -> [batch, j*2, k - kernel + 1]
            if i == 0:
                self.conv_layer.append(
                    nn.Conv1d(input_shape[1], first_conv_layer_size, self.kernel_size))
                last_layer_channels = first_conv_layer_size
            else:
                # past_layer_out = self.get_feature_size(i-1, n_channels_init)
                self.conv_layer.append(
                    nn.Conv1d(last_layer_channels, last_layer_channels*2, self.kernel_size))
                last_layer_channels *= 2
            # Relu não altera as dimensoes do tensor - Função de Ativação
            self.conv_layer.append(nn.ReLU())

            # PARA MAXPOOL: Divide a metade |-> [batch, j, k] -> [batch, j, k/2]
            self.conv_layer.append(nn.MaxPool1d(self.max_pool))
            # Dropout não altera as dimensoes do tensor
            self.conv_layer.append(nn.Dropout(self.dropout_rate))

        # Camada Flatten
        self.flatten = nn.Flatten()

        # Simula n sequencias de (Conv1d(kenrnel_size) + MaxPool1D(max_pool)), baseado num numero inicial de passos e retorna o numero de features após essas operações
        last_layer_features = self.get_feature_size(
            n_conv_layers, input_shape[0])

        # Calcular com quantos neuronios a 1ª camada densa deve ter -> nº de canais * nº de features da última camada
        self.first_dense_input = last_layer_channels * last_layer_features

        self.fc_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.fc_layers.append(
                    nn.Linear(self.first_dense_input, dense_neurons))
            else:
                self.fc_layers.append(
                    nn.Linear(dense_neurons, dense_neurons//dense_layer_droprate))
                dense_neurons //= dense_layer_droprate
            self.fc_layers.append(nn.ReLU())

        # Output Layer
        self.output_layer = nn.Linear(dense_neurons, num_labels)

    def get_feature_size(self, k, init_val):
        def feature_sequence(i, a0):
            if i == 0:
                return a0
            else:
                return (feature_sequence(i-1, a0) - self.kernel_size + 1) // self.max_pool
        return feature_sequence(k, init_val)

    def forward(self, x):
        # print("Input:", x.shape)
        # print()
        for layer in self.conv_layer:
            x = layer(x)
            # if layer._get_name() in ("Conv1d", "MaxPool1d"):
            #     print(layer._get_name(), x.shape)
            #     if layer._get_name() in ("MaxPool1d"): print()

        x = self.flatten(x)  # x = x.view(x.size(0), -1)
        # print("Flatten:", x.shape)
        # print()

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            # if fc_layer._get_name() in ("Linear"):
            #     print(fc_layer._get_name(), x.shape)

        # print()
        x = self.output_layer(x)
        # print("Output:", x.shape)
        x = torch.softmax(x, dim=1)
        # x = torch.argmax(x, dim=1)
        # print("Argmax:", x.shape)
        return x


def fit(epochs, lr, model, train_dl, val_dl, criterion, opt_func=torch.optim.SGD):

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
            loss = criterion(output, target)

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
        for data, target in val_dl:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f"[{epoch+1:>{epoch_len}}/{epochs:>{epoch_len}}] " +
                     f"train_loss: {train_loss:.5f} " +
                     f"valid_loss: {valid_loss:.5f}")

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    return model, avg_train_losses, avg_valid_losses


if __name__ == "__main__":

    debug = True

    current_directory = os.path.dirname(__file__)
    
    # Nº de Epochs
    epochs = 50
    # Taxa de Aprendizagem
    learning_rate = 0.0001
    
    # Função de Custo - BCELoss()??
    loss_fn = nn.CrossEntropyLoss()
    
    # Hiperparams definidos de forma arbitrária -> Impactam diretamente na potência e no tempo de treinamento
    first_conv_layer_size=25
    first_dense_layer_size=6000

    position, label_type, scenario, neural_network_type, n_conv_layers, num_dense_layers = parse_input()
    
    output_dir = os.path.join(current_directory, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    neural_network_results_dir = os.path.join(output_dir, neural_network_type)
    if neural_network_type == "CNN1D":
        neural_network_results_dir = os.path.join(neural_network_results_dir, position)
        
    if not os.path.exists(neural_network_results_dir):
        os.makedirs(neural_network_results_dir)

    # Diretórios dos datasets e tragets. position -> labels_and_data/data/{position}/filename.npy
    data_dir = os.path.join(current_directory, "labels_and_data", "data", position)
    label_dir = os.path.join(current_directory, "labels_and_data", "labels", position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test, = collect_datasets_from_input(
        position, label_type, scenario, neural_network_type, label_dir, data_dir)
    
    
    # --------------------------------------------------------------------------------------------------------------------
    # Coleta de uma pequena amostra do dataset para treinamento parcial
    # X_train, y_train = X_train[0:100], y_train[0:100]
    # X_val, y_val = X_val[0:150], y_val[0:150]
    # --------------------------------------------------------------------------------------------------------------------
    
    
    print("Datasets | Labels")
    print(f"Treinamento: {X_train.shape} ({X_train.dtype}) | {y_train.shape} ({y_train.dtype})")
    print(f"Validação: {X_val.shape} ({X_val.dtype}) | {y_val.shape} ({y_val.dtype})")
    print(f"Teste: {X_test.shape} ({X_test.dtype}) | {y_test.shape} ({y_test.dtype})")

	# Formação dos batches a partir dos datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

	# Definição da Rede Neural - Parametrizar
    model = CNN1D(
        
        # Comprimento das features
        # Representa 5s de registros de movimentos (450 para os punhos, 1020 para o peito)
        input_shape=input_shape,
        
        # "n_conv_layers" Sessões de convolução que duplica o nº de canais a partir da 2ª camada
        n_conv_layers=n_conv_layers,
        first_conv_layer_size=first_conv_layer_size,
        
        # "n_conv_layers" Sessões de camadas densas que reduzem em 30% o nº de canais a partir da 2ª camada
        num_dense_layers=num_dense_layers,
        first_dense_layer_size=first_dense_layer_size,
        num_labels=num_labels  # A depender de uma classificação binária ou multiclasse
    )
    
	# DEBUG
    if debug:
        print("-"*90)
        print(model)
        print("-"*90)
    
	# Treinamento própriamente dito
    model, train_loss, valid_loss = fit(epochs, learning_rate, model, train_dl, val_dl, loss_fn)
    
    # Plotagem do gráfico de perda
    category = "binary" if num_labels == 2 else "multiclass"
    plot_loss_curve(train_loss, valid_loss, neural_network_results_dir, f"CNN1D_{category}_{position}_{scenario}.png")
    print(f"Gráfico de Perda gerado com sucesso. (Verifique o diretório {neural_network_results_dir})")