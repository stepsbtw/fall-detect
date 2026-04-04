import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import DEFAULT_PARAMS, SEED

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


class CNN1DNet(nn.Module):
    def __init__(
        self,
        input_shape,
        width,
        kernel_size,
        depth,
        head_depth,
        dropout,
        number_of_labels,
    ):
        super().__init__()
        self.seq_length = input_shape[0]
        self.input_channels = input_shape[1]
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_ch = self.input_channels
        for _ in range(depth):
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=width,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.bns.append(nn.BatchNorm1d(width))
            in_ch = width

        x = torch.zeros(1, self.input_channels, self.seq_length)
        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(F.relu(bn(conv(x))))
        in_feat = x.view(1, -1).shape[1]

        self.fcs = nn.ModuleList()
        for _ in range(head_depth):
            self.fcs.append(nn.Linear(in_feat, width))
            in_feat = width

        self.output = nn.Linear(in_feat, number_of_labels)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(self.pool(F.relu(bn(conv(x)))))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.drop(F.relu(fc(x)))
        return self.output(x)


class MLPNet(nn.Module):
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
            x = self.drop(F.relu(norm(layer(x))))
        return self.output(x)


class LSTMNet(nn.Module):
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
            x = self.drop(F.relu(fc(x)))

        return self.output(x)


class GRUNet(nn.Module):
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
            x = self.drop(F.relu(fc(x)))

        return self.output(x)


class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


def create_model(model_type, params, input_shape, num_labels):
    if isinstance(input_shape, tuple):
        flat_input_dim = 1
        for d in input_shape:
            flat_input_dim *= d
    else:
        flat_input_dim = input_shape

    if model_type == 'CNN1D':
        return CNN1DNet(
            input_shape=input_shape,
            width=params['width'],
            kernel_size=params['kernel_size'],
            depth=params['depth'],
            head_depth=params.get('head_depth', 1),
            dropout=params['dropout'],
            number_of_labels=num_labels,
        )

    if model_type == 'MLP':
        return MLPNet(
            input_dim=flat_input_dim,
            depth=params['depth'],
            width=params['width'],
            dropout=params['dropout'],
            number_of_labels=num_labels,
        )

    if model_type == 'LSTM':
        return LSTMNet(
            input_dim=input_shape[1],
            width=params['width'],
            depth=params['depth'],
            head_depth=params.get('head_depth', 1),
            dropout=params['dropout'],
            number_of_labels=num_labels,
        )

    if model_type == 'GRU':
        return GRUNet(
            input_dim=input_shape[1],
            width=params['width'],
            depth=params['depth'],
            head_depth=params.get('head_depth', 1),
            dropout=params['dropout'],
            number_of_labels=num_labels,
        )

    if model_type == 'LogReg':
        return LogReg(
            input_dim=flat_input_dim,
            output_dim=num_labels,
        )

    raise ValueError(f'Unsupported neural model: {model_type}')


def make_classical_model(model_type, params, y_train):
    if model_type == 'RF':
        d = DEFAULT_PARAMS['RF']
        return RandomForestClassifier(
            n_estimators=int(params.get('n_estimators', d['n_estimators'])),
            max_depth=int(params.get('max_depth', d['max_depth'])),
            min_samples_split=int(params.get('min_samples_split', d['min_samples_split'])),
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        )

    if model_type == 'SVM':
        from sklearn.svm import LinearSVC

        d = DEFAULT_PARAMS['SVM']
        return CalibratedClassifierCV(
            LinearSVC(
                C=float(params.get('C', d['C'])),
                class_weight='balanced',
                dual='auto',
                max_iter=2000,
                random_state=SEED,
            ),
            cv=3,
            method='sigmoid',
        )

    if model_type == 'XGBoost':
        d = DEFAULT_PARAMS['XGBoost']
        scale_pos_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
        return XGBClassifier(
            n_estimators=int(params.get('n_estimators', d['n_estimators'])),
            max_depth=int(params.get('max_depth', d['max_depth'])),
            learning_rate=float(params.get('learning_rate', d['learning_rate'])),
            subsample=float(params.get('subsample', d['subsample'])),
            colsample_bytree=float(params.get('colsample_bytree', d['colsample_bytree'])),
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=SEED,
            n_jobs=-1,
        )

    if model_type == 'CatBoost':
        if CatBoostClassifier is None:
            raise ImportError('CatBoost is not installed. Install with: pip install catboost')

        d = DEFAULT_PARAMS['CatBoost']
        class_weights = [
            1.0,
            max(float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0), 1.0),
        ]
        return CatBoostClassifier(
            iterations=int(params.get('n_estimators', d['n_estimators'])),
            depth=int(params.get('depth', d['depth'])),
            learning_rate=float(params.get('learning_rate', d['learning_rate'])),
            l2_leaf_reg=float(params.get('l2_leaf_reg', d['l2_leaf_reg'])),
            class_weights=class_weights,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=SEED,
            verbose=False,
        )

    if model_type == 'LogisticRegression':
        d = DEFAULT_PARAMS['LogisticRegression']
        return LogisticRegression(
            C=float(params.get('C', d['C'])),
            class_weight='balanced',
            max_iter=1000,
            random_state=SEED,
            solver='lbfgs',
        )

    raise ValueError(f'Unknown classical model: {model_type}')
