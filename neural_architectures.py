import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, input_shape, num_layers=3, dense_neurons=512, dropout=0.3, learning_rate=0.001, decision_threshold=0.5, num_labels=1):
        super(CustomMLP, self).__init__()
        
        if num_labels > 1:
            raise NotImplementedError("Multiclass ainda não foi implementado.")

        self.decision_threshold = decision_threshold

        self.layers = nn.ModuleList()
        self.input_dim = input_shape[0] * input_shape[1]
        self.layers.append(nn.Flatten())

        current_dim = self.input_dim

        for _ in range(num_layers):
            self.layers.append(nn.Linear(current_dim, dense_neurons))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            current_dim = dense_neurons

        self.output_layer = nn.Linear(current_dim, num_labels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class CNN1D(nn.Module):
    def __init__(self, input_shape, filter_size=64, kernel_size=3, num_layers=2, num_dense_layers=2, dense_neurons=128, dropout=0.3, learning_rate=0.001, decision_threshold=0.5, num_labels=1):
        super(CNN1D, self).__init__()

        self.decision_threshold = decision_threshold
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[1]

        for i in range(num_layers):
            out_channels = filter_size * (2 ** i)
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            self.conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.flatten = nn.Flatten()

        # Compute output size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[1], input_shape[0])
            x = dummy_input
            for layer in self.conv_layers:
                x = layer(x)
            flattened_dim = x.view(1, -1).shape[1]

        self.fc_layers = nn.ModuleList()
        current_dim = flattened_dim
        for _ in range(num_dense_layers):
            self.fc_layers.append(nn.Linear(current_dim, dense_neurons))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            current_dim = dense_neurons

        self.output_layer = nn.Linear(current_dim, num_labels)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class CustomLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size=128, num_lstm_layers=2, num_dense_layers=2, first_dense_size=256, num_labels=1, bidirectional=False, dropout=0.3, learning_rate=0.001, decision_threshold=0.5):
        super(CustomLSTM, self).__init__()

        if num_labels > 1:
            raise NotImplementedError("Multiclasse ainda não foi implementado.")

        self.decision_threshold = decision_threshold
        self.input_size = input_shape[1]

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        # Set input size for dense layers based on bidirectionality
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc_layers = nn.ModuleList()
        current_dim = first_dense_size
        self.fc_layers.append(nn.Linear(lstm_output_size, current_dim))
        self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Dropout(dropout))

        for _ in range(1, num_dense_layers):
            next_dim = current_dim // 2
            self.fc_layers.append(nn.Linear(current_dim, next_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        self.output_layer = nn.Linear(current_dim, num_labels)

    def forward(self, x):
        # Ensure (batch, seq_len, features)
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)

        if x.shape[2] != self.input_size:
            x = x.permute(0, 2, 1)
            if x.shape[2] != self.input_size:
                raise ValueError(f"Input shape mismatch: expected {self.input_size}, got {x.shape[2]}")

        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]

        x = self.dropout(h)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
