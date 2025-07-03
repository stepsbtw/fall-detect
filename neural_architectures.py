import torch
import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size=64, num_lstm_layers=2, num_dense_layers=2, first_dense_size=100, num_labels=1, bidirectional=False):

        super(CustomLSTM, self).__init__()

        if num_labels > 1:
            raise NotImplementedError("Multiclasse ainda não foi implementado.")

        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.input_size = input_shape[1]
            
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(0.3)

        # Camadas densas após a LSTM
        self.fc_layers = nn.ModuleList()
        dense_input_size = hidden_size  # último hidden state será usado como entrada

        for i in range(num_dense_layers):
            self.fc_layers.append(nn.Linear(dense_input_size, first_dense_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.3))
            dense_input_size = first_dense_size
            first_dense_size = first_dense_size // 2  # decresce pela metade

        self.output_layer = nn.Linear(dense_input_size, num_labels)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Corrige shape: (batch, features, seq_len) → (batch, seq_len, features)
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)

        # Ensure x is in shape (batch, seq_len, input_size)
        # If x is (batch, input_size, seq_len), we transpose
        if x.shape[2] != self.input_size:
            x = x.permute(0, 2, 1)  # from (batch, input_size, seq_len) to (batch, seq_len, input_size)
            if x.shape[2] != self.input_size:
                raise ValueError(f"Input shape mismatch after permute: expected input_size={self.input_size}, got {x.shape[2]}")

        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pega o último hidden state da última camada
        if self.lstm.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]


        x = self.dropout(last_hidden)

        for layer in self.fc_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x


class CNN1D(nn.Module):
    def __init__(self, input_shape, n_conv_layers, kernel_size, num_dense_layers, first_dense_layer_size, num_labels=2):
        super(CNN1D, self).__init__()

        self.kernel_size = kernel_size
        self.dropout_rate = 0.3
        self.max_pool = 2

        self.conv_layers = nn.ModuleList()
        last_layer_channels = 0
        dense_neurons = first_dense_layer_size
        dense_layer_droprate = 4

        for i in range(n_conv_layers):
            in_channels = input_shape[1] if i == 0 else last_layer_channels
            out_channels = first_dense_layer_size if i == 0 else last_layer_channels * 2

            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(self.max_pool))
            self.conv_layers.append(nn.Dropout(self.dropout_rate))

            last_layer_channels = out_channels

        self.flatten = nn.Flatten()

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[1], input_shape[0])  # [batch, channels, seq_len]
            x = dummy_input
            for layer in self.conv_layers:
                x = layer(x)
            self.first_dense_input = x.view(1, -1).shape[1]

        self.fc_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.first_dense_input, dense_neurons))
            else:
                self.fc_layers.append(nn.Linear(dense_neurons, dense_neurons // dense_layer_droprate))
                dense_neurons //= dense_layer_droprate
            self.fc_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(dense_neurons, num_labels)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = self.output_layer(x)
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