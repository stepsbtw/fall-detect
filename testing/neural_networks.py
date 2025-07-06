import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DNet(nn.Module):
    def __init__(self, input_shape, filter_size, kernel_size, num_layers,
                 num_dense_layers, dense_neurons, dropout, number_of_labels):
        super(CNN1DNet, self).__init__()
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.input_channels = input_shape[0]
        self.seq_length = input_shape[1]

        in_channels = 1
        out_channels = filter_size
        for i in range(num_layers):
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size))
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
        x = torch.zeros(1, 1, self.seq_length)
        for conv, pool in zip(self.convs, self.pools):
            x = pool(F.relu(conv(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
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
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim, number_of_labels)

    def forward(self, x):
        # Assuming input shape (batch, sequence_length, input_dim)
        out, _ = self.lstm(x)
        # Take last time step output
        out = out[:, -1, :]
        return self.fc(out)
