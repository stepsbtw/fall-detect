import torch
import torch.nn as nn

import src.config as config


class CNN1D(nn.Module):
    def __init__(self, input_shape, width, kernel_size, depth, head_depth, dropout, number_of_labels):
        super().__init__()
        self.seq_length = input_shape[0]
        self.input_channels = input_shape[1]
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_ch = self.input_channels
        for _ in range(depth):
            self.convs.append(nn.Conv1d(in_channels=in_ch, out_channels=width, kernel_size=kernel_size, padding=kernel_size // 2))
            self.bns.append(nn.BatchNorm1d(width))
            in_ch = width

        x = torch.zeros(1, self.input_channels, self.seq_length)
        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(nn.functional.relu(bn(conv(x))))
        in_feat = x.view(1, -1).shape[1]

        self.fcs = nn.ModuleList()
        for _ in range(head_depth):
            self.fcs.append(nn.Linear(in_feat, width))
            in_feat = width

        self.output = nn.Linear(in_feat, number_of_labels)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(self.pool(nn.functional.relu(bn(conv(x)))))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.drop(nn.functional.relu(fc(x)))
        return self.output(x)


class MLP(nn.Module):
    def __init__(self, input_dim, depth, width, dropout, number_of_labels):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_features = input_dim
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, width))
            self.norms.append(nn.LayerNorm(width))
            in_features = width

        self.output = nn.Linear(in_features, number_of_labels)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        for layer, norm in zip(self.layers, self.norms):
            x = self.drop(nn.functional.relu(norm(layer(x))))
        return self.output(x)


class LSTM(nn.Module):
    def __init__(self, input_dim, width, depth, head_depth, dropout, number_of_labels):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=width,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=True,
        )

        feat_dim = width * 2
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(dropout)

        self.fcs = nn.ModuleList()
        in_dim = feat_dim
        for _ in range(head_depth):
            self.fcs.append(nn.Linear(in_dim, width))
            in_dim = width

        self.output = nn.Linear(in_dim, number_of_labels)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)

        out, _ = self.lstm(x)

        feat_mean = out.mean(dim=1)
        feat_max, _ = out.max(dim=1)
        x = 0.5 * (feat_mean + feat_max)

        x = self.norm(x)
        x = self.drop(x)

        for fc in self.fcs:
            x = self.drop(nn.functional.relu(fc(x)))

        return self.output(x)


class GRU(nn.Module):
    def __init__(self, input_dim, width, depth, head_depth, dropout, number_of_labels):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=width,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=True,
        )

        feat_dim = width * 2
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(dropout)

        self.fcs = nn.ModuleList()
        in_dim = feat_dim
        for _ in range(head_depth):
            self.fcs.append(nn.Linear(in_dim, width))
            in_dim = width

        self.output = nn.Linear(in_dim, number_of_labels)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)

        out, _ = self.gru(x)

        feat_mean = out.mean(dim=1)
        feat_max, _ = out.max(dim=1)
        x = 0.5 * (feat_mean + feat_max)

        x = self.norm(x)
        x = self.drop(x)

        for fc in self.fcs:
            x = self.drop(nn.functional.relu(fc(x)))

        return self.output(x)


class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


def create_model(model_type, input_shape, number_of_labels=1):
    if isinstance(input_shape, tuple):
        flat_input_dim = 1
        for d in input_shape:
            flat_input_dim *= d
    else:
        flat_input_dim = input_shape

    if model_type == "CNN1D":
        return CNN1D(
            input_shape=input_shape,
            width=config.WIDTH,
            kernel_size=5,
            depth=config.DEPTH,
            head_depth=config.HEAD_DEPTH,
            dropout=config.DROPOUT,
            number_of_labels=number_of_labels,
        )

    if model_type == "MLP":
        return MLP(
            input_dim=flat_input_dim,
            depth=config.DEPTH,
            width=config.WIDTH,
            dropout=config.DROPOUT,
            number_of_labels=number_of_labels,
        )

    if model_type == "LSTM":
        return LSTM(
            input_dim=input_shape[1],
            width=config.WIDTH,
            depth=config.DEPTH,
            head_depth=config.HEAD_DEPTH,
            dropout=config.DROPOUT,
            number_of_labels=number_of_labels,
        )

    if model_type == "GRU":
        return GRU(
            input_dim=input_shape[1],
            width=config.WIDTH,
            depth=config.DEPTH,
            head_depth=config.HEAD_DEPTH,
            dropout=config.DROPOUT,
            number_of_labels=number_of_labels,
        )

    if model_type == "LinearModel":
        return LogReg(input_dim=flat_input_dim)

    raise ValueError(f"Unknown neural model: {model_type}")
