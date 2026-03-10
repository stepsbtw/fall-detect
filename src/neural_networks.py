import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DNet(nn.Module):
    def __init__(self, input_shape, filter_size, kernel_size, num_layers,
                 num_dense_layers, dense_neurons, dropout, number_of_labels):
        super().__init__()
        self.seq_length = input_shape[0]
        self.input_channels = input_shape[1]
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        in_ch = self.input_channels
        for _ in range(num_layers):
            self.convs.append(nn.Conv1d(in_ch, filter_size, kernel_size, padding=kernel_size // 2))
            in_ch, filter_size = filter_size, filter_size * 2

        self.fcs = nn.ModuleList()
        in_feat = self._get_conv_output_size()
        for _ in range(num_dense_layers):
            self.fcs.append(nn.Linear(in_feat, dense_neurons))
            in_feat = dense_neurons

        self.output = nn.Linear(in_feat, number_of_labels)

    def _get_conv_output_size(self):
        x = torch.zeros(1, self.input_channels, self.seq_length)
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        for conv in self.convs:
            x = self.drop(self.pool(F.relu(conv(x))))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.drop(F.relu(fc(x)))
        return self.output(x)



class MLPNet(nn.Module):
    def __init__(self, input_dim, num_layers, dense_neurons, dropout, number_of_labels):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        in_features = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_features, dense_neurons))
            in_features = dense_neurons

        self.output = nn.Linear(in_features, number_of_labels)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = self.drop(F.relu(layer(x)))
        return self.output(x)


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, number_of_labels):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, number_of_labels)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        out, _ = self.lstm(x)
        return self.output(self.drop(out[:, -1, :]))