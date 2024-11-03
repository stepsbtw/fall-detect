import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, input_shape, n_conv_layers, first_conv_layer_size,  num_dense_layers, first_dense_layer_size,  num_labels=2):
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
        self.output_layer = nn.Linear(dense_neurons, 1)

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
        # x = torch.softmax(x, dim=1)
        # x = torch.argmax(x, dim=1)
        # print("Argmax:", x.shape)
        return x


class CustomMLP(nn.Module):
    def __init__(self, input_shape, num_asc_layers=1, num_desc_layers=3, num_labels=1):
        super(CustomMLP, self).__init__()
        
        if num_labels > 1:
            raise NotImplemented("Essa feature ainda não foi desenvolvida de forma estável")
        
        # Taxa de crescimento entre camadas
        self.asc_factor = 2
        # Taxa de decrescimento entre camadas
        self.desc_factor = 3
        
        self.layers = nn.ModuleList()
        self.dropout_factor = 0.6
        # Recebe como entrada input_shape tuple(N_observ, N_vari)
        # Caso Multivariado (N_vari > 1) - Add um flatten
        if input_shape[1] > 1: self.layers.append(nn.Flatten())

        input_shape = input_shape[0] * input_shape[1]

        for i in range(num_asc_layers):
            self.layers.append(nn.Linear(in_features=input_shape, out_features=input_shape * self.asc_factor))
            input_shape *= self.asc_factor
            self.layers.append(nn.ReLU()) 
            self.layers.append(nn.Dropout(self.dropout_factor))

        for i in range(num_desc_layers):
            self.layers.append(nn.Linear(in_features=input_shape, out_features=input_shape // self.desc_factor))
            input_shape //= self.desc_factor
            self.layers.append(nn.ReLU()) 
            self.layers.append(nn.Dropout(self.dropout_factor))
            
        self.layers.pop(-1)
            # Classificação binária -> Saida em um único neurônio
        self.output_layer = nn.Linear(in_features=input_shape, out_features=num_labels)

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.output_layer(x)

        return x
