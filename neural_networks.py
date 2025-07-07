import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DNet(nn.Module):
    def __init__(self, input_shape, filter_size, kernel_size, num_layers,
                 num_dense_layers, dense_neurons, dropout, number_of_labels):
        super(CNN1DNet, self).__init__()
        
        # input_shape esperado: (seq_len, num_features)
        self.seq_length = input_shape[0]
        self.input_channels = input_shape[1]  # agora sim: num_features → vira canal para Conv1d
        
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = self.input_channels
        out_channels = filter_size
        for _ in range(num_layers):
            self.convs.append(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    padding=kernel_size // 2
                )
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            self.dropouts.append(nn.Dropout(dropout))
            in_channels = out_channels
            out_channels *= 2

        # Flatten
        conv_output_size = self._get_conv_output_size()

        self.fcs = nn.ModuleList()
        in_features = conv_output_size
        for _ in range(num_dense_layers):
            self.fcs.append(nn.Linear(in_features, dense_neurons))
            self.dropouts.append(nn.Dropout(dropout))
            in_features = dense_neurons

        self.output = nn.Linear(in_features, number_of_labels)

    def _get_conv_output_size(self):
        # Simula entrada: (1, num_features, seq_length)
        x = torch.zeros(1, self.input_channels, self.seq_length)
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x))
            x = pool(x)
            if x.shape[-1] <= 0:
                raise ValueError(
                    f"Camada conv produziu saída com comprimento <= 0. "
                    f"Ajuste kernel_size ou num_layers. Shape: {x.shape}"
                )
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # x: (batch_size, 1, seq_length, num_features)
        x = x.squeeze(1)       # (batch_size, seq_length, num_features)
        x = x.permute(0, 2, 1) # (batch_size, num_features, seq_length)

        for conv, pool, drop in zip(self.convs, self.pools, self.dropouts[:len(self.convs)]):
            x = pool(F.relu(conv(x)))
            x = drop(x)

        x = x.view(x.size(0), -1)

        for fc, drop in zip(self.fcs, self.dropouts[len(self.convs):]):
            x = F.relu(fc(x))
            x = drop(x)

        return self.output(x)



class MLPNet(nn.Module):
    def __init__(self, input_dim, num_layers, dense_neurons, dropout, number_of_labels):
        super(MLPNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_features, dense_neurons))
            self.dropouts.append(nn.Dropout(dropout))
            in_features = dense_neurons

        self.output = nn.Linear(in_features, number_of_labels)

    def forward(self, x):
        # Se a entrada for 4D: (batch, 1, seq_len, num_features)
        if x.dim() == 4:
            x = x.squeeze(1)  # (batch, seq_len, num_features)
            x = x.view(x.size(0), -1)  # flatten total para (batch, seq_len * num_features)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)  # flatten genérico

        for layer, drop in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = drop(x)

        return self.output(x)


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, number_of_labels):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, number_of_labels)

    def forward(self, x):
        # x: (batch_size, 1, seq_len, input_dim)
        if x.dim() == 4:
            x = x.squeeze(1)  # (batch_size, seq_len, input_dim)

        out, _ = self.lstm(x)     # (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]       # Último passo de tempo
        out = self.dropout(out)
        out = self.fc(out)
        return out