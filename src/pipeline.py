import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BLOCK_SIZE,
    CANONICAL_SENSORS,
    CLASSICAL_MODELS,
    DATA_PATH,
    DEFAULT_PARAMS,
    DEVICE,
    MODEL_CAPACITY,
    MODELS_ROOT,
    NUM_LABELS,
    OUTPUT_ROOT,
    SCENARIOS,
    TRAINING_CONFIG,
    setup_runtime,
)
from models import create_model, make_classical_model


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal experiment runner')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cross_sensor', action='store_true')
    parser.add_argument('--fused_missing', action='store_true')
    parser.add_argument('--model', choices=list(DEFAULT_PARAMS.keys()), required=True)
    parser.add_argument('--scenario', choices=list(SCENARIOS.keys()), required=True)
    parser.add_argument('--test_scenario', choices=list(SCENARIOS.keys()))
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'])
    parser.add_argument('--loss', choices=['weighted', 'unweighted'], default='weighted')
    parser.add_argument('--inner_val_groups', type=int, default=3)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--no_mag', action='store_true')
    parser.add_argument('--only_mag', action='store_true')
    parser.add_argument('--sensor_dropout', action='store_true')
    parser.add_argument('--sensor_dropout_p', type=float, default=0.5)
    parser.add_argument('--sensor_dropout_max_off', type=int, default=1)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--tune_threshold', action='store_true')
    parser.add_argument('--threshold_metric', choices=['f1', 'youden'], default='f1')
    parser.add_argument('--width', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--head_depth', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--kernel_size', type=int)
    parser.add_argument('--save_arrays', action='store_true', default=False)
    return parser.parse_args()

def logits_to_binary_probs(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return np.empty((0, 2), dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.shape[1] == 1:
        prob_pos = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return np.column_stack([1.0 - prob_pos, prob_pos])
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_shift = np.exp(shift)
    return exp_shift / np.clip(np.sum(exp_shift, axis=1, keepdims=True), 1e-12, None)



def sensors_from_scenario(scenario):
    base = scenario
    if base.endswith('_NW'):
        base = base[:-3]
    if '_IVG' in base:
        base = base.split('_IVG', 1)[0]
    for suffix in ('_SC', '_NM', '_OM'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    if '_SDP' in base:
        base = base.split('_SDP', 1)[0]
    if base.endswith('_T') or base.endswith('_F'):
        base = base[:-2]
    sensors = [p for p in base.split('_') if p in CANONICAL_SENSORS]
    if not sensors:
        raise ValueError(f"Could not infer sensors from scenario '{scenario}'")
    return sensors


def scenario_name(args):
    name = args.scenario if args.loss == 'weighted' else f'{args.scenario}_NW'
    if args.model not in CLASSICAL_MODELS:
        name = f'{name}_IVG{max(int(args.inner_val_groups), 1)}'
    if args.scale:
        name += '_SC'
    if args.no_mag:
        name += '_NM'
    if args.only_mag:
        name += '_OM'
    if args.sensor_dropout:
        name += f"_SDP{str(args.sensor_dropout_p).replace('.', 'p')}_M{int(args.sensor_dropout_max_off)}"
    if args.tune_threshold:
        name += f'_TT_{args.threshold_metric}'
    elif args.threshold is not None:
        name += f"_TH_{str(float(args.threshold)).replace('.', 'p')}"
    return name


def resolve_params(args):
    params = dict(DEFAULT_PARAMS[args.model])

    if args.model not in CLASSICAL_MODELS:
        params = {**MODEL_CAPACITY, **params}
        params['dropout'] = float(TRAINING_CONFIG['dropout'])
        params['learning_rate'] = float(TRAINING_CONFIG['learning_rate'])
        params['weight_decay'] = float(TRAINING_CONFIG['weight_decay'])

        for key in (
            'width',
            'depth',
            'head_depth',
            'dropout',
            'learning_rate',
            'weight_decay',
            'kernel_size',
        ):
            value = getattr(args, key, None)
            if value is not None:
                params[key] = value
    else:
        for key in ('kernel_size',):
            value = getattr(args, key, None)
            if value is not None:
                params[key] = value

    params['decision_threshold'] = float(
        args.threshold if args.threshold is not None else TRAINING_CONFIG['decision_threshold']
    )
    return params


def train_experiment(args):
    setup_runtime()
    run_name = scenario_name(args)
    output_dir = os.path.join(OUTPUT_ROOT, args.model, run_name)
    model_dir = os.path.join(MODELS_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'run_config.json'), 'w', encoding='utf-8') as fh:
        json.dump(vars(args), fh, indent=2)
    if os.path.exists(os.path.join(output_dir, 'DONE')) or os.path.exists(os.path.join(output_dir, 'summary_metrics.csv')):
        print(f'Run already complete at: {output_dir} - skipping.')
        return output_dir

    bundle = load_bundle(args.scenario, args)
    logo = LeaveOneGroupOut()
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(bundle['X'], bundle['y'], bundle['groups'])):
        left_out = bundle['groups'][test_idx[0]]
        fold_label = f's{left_out}'
        fold_dir = os.path.join(output_dir, f'fold_{fold_label}')
        fold_model_dir = os.path.join(model_dir, f'fold_{fold_label}')
        row = fit_and_eval_fold(
            args,
            bundle,
            train_idx,
            test_idx,
            fold_idx,
            fold_dir,
            fold_model_dir,
            bundle['scenario'],
            {'missing': [], 'available': sensors_from_scenario(bundle['scenario'])},
        )
        row['fold'] = fold_label
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    open(os.path.join(output_dir, 'DONE'), 'w').close()
    with open(os.path.join(output_dir, 'status.json'), 'w', encoding='utf-8') as fh:
        json.dump({'mode': 'train', 'n_folds': len(rows)}, fh, indent=2)
    return output_dir


def fused_missing_experiment(args):
    if not args.test_scenario:
        raise ValueError('--test_scenario is required for fused-missing mode')

    setup_runtime()

    train_sensors = tuple(sensors_from_scenario(args.scenario))
    test_sensors = tuple(sensors_from_scenario(args.test_scenario))
    train_shape = SCENARIOS[args.scenario][2]

    allowed_targets = [
        name for name, (_, _, shape) in SCENARIOS.items()
        if shape[0] == train_shape[0]
        and tuple(sensors_from_scenario(name)) != train_sensors
        and set(sensors_from_scenario(name)).issubset(set(train_sensors))
    ]
    if args.test_scenario not in allowed_targets:
        raise ValueError(
            f'Invalid fused-missing pair: {args.scenario} -> {args.test_scenario}. '
            f'Allowed test scenarios: {allowed_targets}'
        )

    train_run_name = scenario_name(args)
    trained_output_dir = os.path.join(OUTPUT_ROOT, args.model, train_run_name)
    trained_model_dir = os.path.join(MODELS_ROOT, args.model, train_run_name)

    if not os.path.exists(trained_output_dir):
        raise FileNotFoundError(
            f'Trained run not found: {trained_output_dir}. '
            f'Run --train first for the same configuration.'
        )
    if not os.path.exists(trained_model_dir):
        raise FileNotFoundError(
            f'Trained model dir not found: {trained_model_dir}. '
            f'Run --train first for the same configuration.'
        )

    run_name = f"padded_eval_{train_run_name}_on_{args.test_scenario}"
    output_dir = os.path.join(OUTPUT_ROOT, args.model, run_name)
    model_dir = os.path.join(MODELS_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'run_config.json'), 'w', encoding='utf-8') as fh:
        json.dump(vars(args), fh, indent=2)

    train_bundle = load_bundle(args.scenario, args)
    test_bundle = load_bundle(args.test_scenario, args)

    train_bundle['X'] = expand_to_train_layout(train_bundle['X'], train_sensors, train_sensors)
    test_bundle['X'] = expand_to_train_layout(test_bundle['X'], test_sensors, train_sensors)

    rows = []
    logo = LeaveOneGroupOut()

    for fold_idx, (_, test_idx) in enumerate(logo.split(test_bundle['X'], test_bundle['y'], test_bundle['groups'])):
        left_out = test_bundle['groups'][test_idx[0]]
        fold_label = f's{left_out}'

        fold_dir = os.path.join(output_dir, f'fold_{fold_label}')
        fold_model_dir = os.path.join(model_dir, f'fold_{fold_label}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(fold_model_dir, exist_ok=True)

        trained_fold_output_dir = os.path.join(trained_output_dir, f'fold_{fold_label}')
        trained_fold_model_dir = os.path.join(trained_model_dir, f'fold_{fold_label}')

        fit_idx = np.where(train_bundle['groups'] != left_out)[0]
        sensor_status = {
            'missing': [s for s in train_sensors if s not in test_sensors],
            'available': list(test_sensors),
        }

        row = infer_with_trained_fold(
            args=args,
            train_bundle=train_bundle,
            test_bundle=test_bundle,
            fit_idx=fit_idx,
            test_idx=test_idx,
            fold_idx=fold_idx,
            fold_label=fold_label,
            trained_fold_output_dir=trained_fold_output_dir,
            trained_fold_model_dir=trained_fold_model_dir,
            out_fold_dir=fold_dir,
            scenario_for_outputs=args.test_scenario,
            sensor_status=sensor_status,
        )
        row['fold'] = fold_label
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)

    open(os.path.join(output_dir, 'DONE'), 'w').close()
    with open(os.path.join(output_dir, 'status.json'), 'w', encoding='utf-8') as fh:
        json.dump({'mode': 'fused_missing_inference_only', 'n_folds': len(rows)}, fh, indent=2)

    return output_dir


def cross_sensor_experiment(args):
    if not args.test_scenario:
        raise ValueError('--test_scenario is required for cross-sensor mode in this minimal version')
    setup_runtime()
    source_bundle = load_bundle(args.scenario, args)
    target_bundle = load_bundle(args.test_scenario, args)
    run_name = f'cross_sensor_{scenario_name(args)}_to_{args.test_scenario}'
    output_dir = os.path.join(OUTPUT_ROOT, args.model, run_name)
    model_dir = os.path.join(MODELS_ROOT, args.model, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    rows = []
    logo = LeaveOneGroupOut()
    for fold_idx, (train_idx, test_idx_source) in enumerate(logo.split(source_bundle['X'], source_bundle['y'], source_bundle['groups'])):
        left_out = source_bundle['groups'][test_idx_source[0]]
        aligned = []
        wanted = set(zip(source_bundle['groups'][test_idx_source], source_bundle['window_ids'][test_idx_source]))
        for i, pair in enumerate(zip(target_bundle['groups'], target_bundle['window_ids'])):
            if pair in wanted:
                aligned.append(i)
        if not aligned:
            continue
        fold_label = f's{left_out}'
        fold_dir = os.path.join(output_dir, f'fold_{fold_label}')
        fold_model_dir = os.path.join(model_dir, f'fold_{fold_label}')
        row = fit_and_eval_fold(
            args,
            source_bundle,
            train_idx,
            np.asarray(aligned),
            fold_idx,
            fold_dir,
            fold_model_dir,
            args.test_scenario,
            {
                'missing': [s for s in sensors_from_scenario(args.scenario) if s not in sensors_from_scenario(args.test_scenario)],
                'available': sensors_from_scenario(args.test_scenario),
            },
            test_bundle=target_bundle,
        )
        row['fold'] = fold_label
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    open(os.path.join(output_dir, 'DONE'), 'w').close()
    return output_dir


def augment_sensor_dropout(X, y, sensors, block_size=BLOCK_SIZE, p=0.5, max_off=1, copies=1, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    sensors = list(sensors)
    if X.ndim != 3:
        raise ValueError(f'Expected X with shape (n_samples, timesteps, channels), got {X.shape}')
    if not sensors or copies <= 0 or p <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    X_parts = [X]
    y_parts = [y]
    max_off = min(max(int(max_off), 1), len(sensors))

    for _ in range(int(copies)):
        X_copy = X.copy()
        for i in range(len(X_copy)):
            if rng.random() >= float(p):
                continue
            n_drop = int(rng.integers(1, max_off + 1))
            dropped = rng.choice(sensors, size=n_drop, replace=False)
            for sensor in dropped:
                start = sensors.index(sensor) * block_size
                X_copy[i, :, start:start + block_size] = 0.0
        X_parts.append(X_copy)
        y_parts.append(y)

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def apply_sensor_dropout_torch(xb, n_sensors, block_size=BLOCK_SIZE, p=0.5, max_off=1):
    """
    Apply sensor dropout directly on a torch batch tensor of shape (B, T, C).
    Entire sensor channel blocks are zeroed without leaving torch/device memory.
    """
    if p <= 0 or max_off <= 0 or n_sensors <= 0:
        return xb

    B = xb.shape[0]
    max_off = min(max(int(max_off), 1), int(n_sensors))
    out = xb.clone()

    apply_mask = torch.rand(B, device=xb.device) < float(p)
    active_rows = torch.nonzero(apply_mask, as_tuple=False).flatten()

    if active_rows.numel() == 0:
        return out

    for row in active_rows.tolist():
        n_drop = int(torch.randint(1, max_off + 1, (1,), device=xb.device).item())
        dropped = torch.randperm(int(n_sensors), device=xb.device)[:n_drop]
        for sidx in dropped.tolist():
            start = sidx * block_size
            out[row, :, start:start + block_size] = 0.0

    return out

def get_loader_kwargs(shuffle=False, generator=None):
    pin_memory = bool(TRAINING_CONFIG.get('pin_memory', DEVICE.type == 'cuda'))
    configured = int(TRAINING_CONFIG.get('num_workers', 0))
    if configured > 0:
        num_workers = configured
    else:
        cpu_total = os.cpu_count() or 1
        num_workers = min(4, max(cpu_total - 1, 0)) if cpu_total > 1 else 0

    kwargs = {
        'batch_size': TRAINING_CONFIG['batch_size'],
        'shuffle': shuffle,
        'pin_memory': pin_memory,
        'num_workers': num_workers,
    }
    if generator is not None:
        kwargs['generator'] = generator
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def maybe_autocast():
    return torch.amp.autocast(device_type='cuda', enabled=(DEVICE.type == 'cuda'))


def fit_and_eval_fold(
    args,
    train_bundle,
    train_idx,
    test_idx,
    fold_idx,
    fold_dir,
    fold_model_dir,
    scenario_for_outputs,
    sensor_status,
    test_bundle=None,
):
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(fold_model_dir, exist_ok=True)
    if os.path.exists(os.path.join(fold_dir, 'metrics.csv')):
        return pd.read_csv(os.path.join(fold_dir, 'metrics.csv')).iloc[0].to_dict()

    test_bundle = test_bundle or train_bundle
    X_train_all = train_bundle['X'][train_idx]
    y_train_all = train_bundle['y'][train_idx]
    groups_train = train_bundle['groups'][train_idx]
    inner_subjects = np.unique(groups_train)
    n_val_groups = min(int(args.inner_val_groups), len(inner_subjects) - 1)
    if n_val_groups <= 0:
        raise ValueError('Inner validation requires at least 2 training groups in each outer fold.')

    start_idx = int(fold_idx) % len(inner_subjects)
    val_subjects = [inner_subjects[(start_idx + k) % len(inner_subjects)] for k in range(n_val_groups)]
    val_mask = np.isin(groups_train, val_subjects)

    X_train = X_train_all[~val_mask]
    y_train = y_train_all[~val_mask]
    X_val = X_train_all[val_mask]
    y_val = y_train_all[val_mask]
    X_test = test_bundle['X'][test_idx]
    y_test = test_bundle['y'][test_idx]

    scaler = None
    if args.scale:
        scaler = fit_fold_scaler(X_train)
        X_train = apply_scaler_3d(X_train, scaler)
        X_val = apply_scaler_3d(X_val, scaler)
        X_test = apply_scaler_3d(X_test, scaler)
        joblib.dump(scaler, os.path.join(fold_model_dir, 'scaler.joblib'))

    params = resolve_params(args)
    threshold = float(args.threshold if args.threshold is not None else params.get('decision_threshold', 0.5))
    threshold_score = float('nan')
    fold_label = os.path.basename(fold_dir).replace('fold_', '')

    if args.model in CLASSICAL_MODELS:
        if args.sensor_dropout:
            drop_sensors = sensors_from_scenario(args.scenario)
            X_train, y_train = augment_sensor_dropout(
                X_train,
                y_train,
                sensors=drop_sensors,
                block_size=BLOCK_SIZE,
                p=args.sensor_dropout_p,
                max_off=args.sensor_dropout_max_off,
                copies=1,
                seed=42 + int(fold_idx),
            )
        X_train_fit = X_train.reshape(len(X_train), -1)
        X_val_fit = X_val.reshape(len(X_val), -1)
        X_test_fit = X_test.reshape(len(X_test), -1)
        model = make_classical_model(args.model, params, y_train)
        model.fit(X_train_fit, y_train)
        if args.tune_threshold:
            if hasattr(model, 'predict_proba'):
                val_prob = model.predict_proba(X_val_fit)[:, 1]
            else:
                val_prob = 1.0 / (1.0 + np.exp(-np.asarray(model.decision_function(X_val_fit), dtype=float)))
            threshold, threshold_score = pick_threshold(y_val, val_prob, args.threshold_metric)
        if hasattr(model, 'predict_proba'):
            test_prob_1 = model.predict_proba(X_test_fit)[:, 1]
        else:
            test_prob_1 = 1.0 / (1.0 + np.exp(-np.asarray(model.decision_function(X_test_fit), dtype=float)))
        test_probs = np.column_stack([1.0 - test_prob_1, test_prob_1])
        joblib.dump(model, os.path.join(fold_model_dir, f'{fold_label}.joblib'))
    else:
        if args.model in {'MLP'}:
            input_shape = X_train.shape[1] * X_train.shape[2]
        else:
            input_shape = (X_train.shape[1], X_train.shape[2])

        neural_output_dim = 1
        model = create_model(args.model, params, input_shape, neural_output_dim)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', TRAINING_CONFIG['weight_decay']),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=TRAINING_CONFIG['patience'],
            min_lr=1e-6,
        )
        if args.loss == 'weighted':
            pos_count = max(int((y_train == 1).sum()), 1)
            neg_count = max(int((y_train == 0).sum()), 1)
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        generator = setup_runtime()
        scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            **get_loader_kwargs(shuffle=True, generator=generator),
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            **get_loader_kwargs(shuffle=False),
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            **get_loader_kwargs(shuffle=False),
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        train_losses, val_losses = [], []
        drop_sensors = sensors_from_scenario(args.scenario)

        for epoch in range(int(args.epochs or TRAINING_CONFIG['epochs'])):
            print(f'[{epoch}/{args.epochs}]')
            model.train()
            epoch_train = []

            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                if args.sensor_dropout:
                    xb = apply_sensor_dropout_torch(
                        xb,
                        n_sensors=len(drop_sensors),
                        block_size=BLOCK_SIZE,
                        p=args.sensor_dropout_p,
                        max_off=args.sensor_dropout_max_off,
                    )

                optimizer.zero_grad(set_to_none=True)

                with maybe_autocast():
                    out = model(xb)
                    target = yb.float().unsqueeze(1)
                    loss = criterion(out, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_train.append(float(loss.detach().item()))

            train_losses.append(float(np.mean(epoch_train)) if epoch_train else float('nan'))

            model.eval()
            epoch_val = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    with maybe_autocast():
                        out = model(xb)
                        target = yb.float().unsqueeze(1)
                        loss = criterion(out, target)
                    epoch_val.append(float(loss.detach().item()))

            avg_val_loss = float(np.mean(epoch_val)) if epoch_val else float('nan')
            val_losses.append(avg_val_loss)

            if np.isfinite(avg_val_loss):
                scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= TRAINING_CONFIG['patience']:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        loss_df = pd.DataFrame({
            'epoch': np.arange(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            })
        loss_df.to_csv(os.path.join(fold_dir, 'loss_curve.csv'), index=False)
        save_loss_curve(
            train_losses,
            val_losses,
            os.path.join(fold_dir, 'loss_curve.png'),
        )

        model.eval()
        val_logits = []
        test_logits = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                val_logits.append(model(xb).detach().cpu().numpy())
            for xb, _ in test_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                test_logits.append(model(xb).detach().cpu().numpy())

        val_logits = np.concatenate(val_logits, axis=0) if val_logits else np.empty((0, 1), dtype=float)
        test_logits = np.concatenate(test_logits, axis=0) if test_logits else np.empty((0, 1), dtype=float)

        val_probs = logits_to_binary_probs(val_logits)
        test_probs = logits_to_binary_probs(test_logits)

        if args.tune_threshold:
            threshold, threshold_score = pick_threshold(y_val, val_probs[:, 1], args.threshold_metric)

        torch.save(model.state_dict(), os.path.join(fold_model_dir, f'{fold_label}.pt'))

    metrics = score_and_save_fold_outputs(
        y_test=y_test,
        test_probs=test_probs,
        threshold=threshold,
        fold_dir=fold_dir,
        test_bundle=test_bundle,
        test_idx=test_idx,
        scenario_for_outputs=scenario_for_outputs,
        sensor_status=sensor_status,
        save_arrays=args.save_arrays,
    )

    if args.tune_threshold:
        metrics['threshold_metric'] = args.threshold_metric
        if not np.isnan(threshold_score):
            metrics['threshold_tuning_score'] = float(threshold_score)
        pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)
        pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'done.csv'), index=False)

    return metrics


def fit_fold_scaler(X_train):
    n_tr, t_steps, n_channels = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_channels))
    return scaler


def apply_scaler_3d(X, scaler):
    n, t_steps, n_channels = X.shape
    return scaler.transform(X.reshape(-1, n_channels)).reshape(n, t_steps, n_channels)


def get_fold_threshold(args, trained_fold_output_dir):
    metrics_path = os.path.join(trained_fold_output_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        if 'threshold' in metrics_df.columns and len(metrics_df) > 0:
            thr = metrics_df.iloc[0]['threshold']
            if pd.notna(thr):
                return float(thr)
    params = resolve_params(args)
    return float(args.threshold if args.threshold is not None else params.get('decision_threshold', 0.5))


def load_or_rebuild_fold_scaler(args, train_bundle, fit_idx, fold_idx, trained_fold_model_dir):
    scaler_path = os.path.join(trained_fold_model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)

    X_train_all = train_bundle['X'][fit_idx]
    groups_train = train_bundle['groups'][fit_idx]
    inner_subjects = np.unique(groups_train)
    n_val_groups = min(int(args.inner_val_groups), len(inner_subjects) - 1)
    if n_val_groups <= 0:
        raise ValueError('Inner validation requires at least 2 training groups in each outer fold.')

    start_idx = int(fold_idx) % len(inner_subjects)
    val_subjects = [inner_subjects[(start_idx + k) % len(inner_subjects)] for k in range(n_val_groups)]
    val_mask = np.isin(groups_train, val_subjects)
    X_train = X_train_all[~val_mask]

    return fit_fold_scaler(X_train)


def infer_probs_with_loaded_model(args, X_test, trained_fold_model_dir, fold_label):
    params = resolve_params(args)

    if args.model in CLASSICAL_MODELS:
        model_path = os.path.join(trained_fold_model_dir, f'{fold_label}.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Missing trained classical model: {model_path}')
        model = joblib.load(model_path)
        X_test_fit = X_test.reshape(len(X_test), -1)

        if hasattr(model, 'predict_proba'):
            prob_1 = model.predict_proba(X_test_fit)[:, 1]
        else:
            prob_1 = 1.0 / (1.0 + np.exp(-np.asarray(model.decision_function(X_test_fit), dtype=float)))

        return np.column_stack([1.0 - prob_1, prob_1])

    if args.model in {'MLP'}:
        input_shape = X_test.shape[1] * X_test.shape[2]
    else:
        input_shape = (X_test.shape[1], X_test.shape[2])

    neural_output_dim = 1
    model = create_model(args.model, params, input_shape, neural_output_dim)
    model_path = os.path.join(trained_fold_model_dir, f'{fold_label}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Missing trained neural model: {model_path}')

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.zeros(len(X_test), dtype=torch.long)),
        **get_loader_kwargs(shuffle=False),
    )

    logits = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            with maybe_autocast():
                out = model(xb)
            logits.append(out.detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0) if logits else np.empty((0, 1), dtype=float)
    return logits_to_binary_probs(logits)


def score_and_save_fold_outputs(
    y_test,
    test_probs,
    threshold,
    fold_dir,
    test_bundle,
    test_idx,
    scenario_for_outputs,
    sensor_status,
    save_arrays=False,
):
    good = ~np.isnan(test_probs).any(axis=1)
    y_true = np.asarray(y_test, dtype=int)[good]
    y_probs = np.asarray(test_probs, dtype=float)[good]
    y_pred = (y_probs[:, 1] >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    except Exception:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(y_true, y_probs[:, 1])
    except Exception:
        pr_auc = float('nan')

    metrics = {
        'acc': float(accuracy_score(y_true, y_pred)),
        'prec': float(precision_score(y_true, y_pred, zero_division=0)),
        'rec': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'threshold': float(threshold),
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'done.csv'), index=False)

    data = {
        'y_true': y_true,
        'y_prob_0': y_probs[:, 0],
        'y_prob_1': y_probs[:, 1],
        'y_pred': y_pred,
    }
    if test_bundle.get('groups') is not None:
        data['group_id'] = np.asarray(test_bundle['groups'][test_idx])[good]
    if test_bundle.get('window_ids') is not None:
        data['window_id'] = np.asarray(test_bundle['window_ids'][test_idx], dtype=object)[good]
    data['scenario'] = [scenario_for_outputs] * len(y_true)
    data['missing_sensors'] = [','.join(sensor_status.get('missing', []))] * len(y_true)
    data['available_sensors'] = [','.join(sensor_status.get('available', []))] * len(y_true)
    pd.DataFrame(data).to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False)

    if save_arrays:
        np.save(os.path.join(fold_dir, 'y_true.npy'), y_true)
        np.save(os.path.join(fold_dir, 'y_probs.npy'), y_probs)
        np.save(os.path.join(fold_dir, 'y_pred.npy'), y_pred)

    return metrics


def infer_with_trained_fold(
    args,
    train_bundle,
    test_bundle,
    fit_idx,
    test_idx,
    fold_idx,
    fold_label,
    trained_fold_output_dir,
    trained_fold_model_dir,
    out_fold_dir,
    scenario_for_outputs,
    sensor_status,
):
    if os.path.exists(os.path.join(out_fold_dir, 'metrics.csv')):
        return pd.read_csv(os.path.join(out_fold_dir, 'metrics.csv')).iloc[0].to_dict()

    X_test = test_bundle['X'][test_idx]
    y_test = test_bundle['y'][test_idx]

    if args.scale:
        scaler = load_or_rebuild_fold_scaler(
            args=args,
            train_bundle=train_bundle,
            fit_idx=fit_idx,
            fold_idx=fold_idx,
            trained_fold_model_dir=trained_fold_model_dir,
        )
        X_test = apply_scaler_3d(X_test, scaler)

    threshold = get_fold_threshold(args, trained_fold_output_dir)
    test_probs = infer_probs_with_loaded_model(
        args=args,
        X_test=X_test,
        trained_fold_model_dir=trained_fold_model_dir,
        fold_label=fold_label,
    )

    return score_and_save_fold_outputs(
        y_test=y_test,
        test_probs=test_probs,
        threshold=threshold,
        fold_dir=out_fold_dir,
        test_bundle=test_bundle,
        test_idx=test_idx,
        scenario_for_outputs=scenario_for_outputs,
        sensor_status=sensor_status,
        save_arrays=args.save_arrays,
    )


def load_bundle(scenario, args):
    dir_name, filename, _ = SCENARIOS[scenario]
    X = np.load(os.path.join(DATA_PATH, dir_name, 'data', filename))
    y = np.load(os.path.join(DATA_PATH, dir_name, 'labels', 'labels.npy')).astype(np.int64)
    groups = np.load(os.path.join(DATA_PATH, dir_name, 'labels', 'groups.npy'))
    window_ids_path = os.path.join(DATA_PATH, dir_name, 'labels', 'window_ids.npy')
    window_ids = np.load(window_ids_path, allow_pickle=True) if os.path.exists(window_ids_path) else None
    if args.no_mag:
        channels = X.shape[2]
        n_sensors = channels // 8
        drop_cols = {s * 8 + offset for s in range(n_sensors) for offset in (0, 4)}
        X = X[:, :, [c for c in range(channels) if c not in drop_cols]]
    if args.only_mag:
        channels = X.shape[2]
        n_sensors = channels // 8
        X = X[:, :, [s * 8 + offset for s in range(n_sensors) for offset in (0, 4)]]
    return {'scenario': scenario, 'X': X, 'y': y, 'groups': groups, 'window_ids': window_ids}


def expand_to_train_layout(X, source_sensors, target_sensors):
    out = np.full((X.shape[0], X.shape[1], len(target_sensors) * BLOCK_SIZE), 0.0, dtype=X.dtype)
    for sensor in source_sensors:
        s0 = source_sensors.index(sensor) * BLOCK_SIZE
        t0 = target_sensors.index(sensor) * BLOCK_SIZE
        out[:, :, t0:t0 + BLOCK_SIZE] = X[:, :, s0:s0 + BLOCK_SIZE]
    return out


def pick_threshold(y_true, y_prob_pos, metric='f1'):
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos, dtype=float)
    if metric == 'f1':
        _, _, thresholds = precision_recall_curve(y_true, y_prob_pos)
    elif metric == 'youden':
        _, _, thresholds = roc_curve(y_true, y_prob_pos)
    else:
        raise ValueError(f'Unsupported threshold metric: {metric}')
    candidates = np.asarray(sorted({float(t) for t in np.ravel(thresholds) if np.isfinite(t)}), dtype=float)
    if candidates.size == 0:
        candidates = np.asarray([0.5], dtype=float)
    candidates = np.clip(candidates, 0.0, 1.0)
    best_threshold, best_score = 0.5, float('-inf')
    for thr in candidates:
        y_pred = (y_prob_pos >= float(thr)).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        score = f1_score(y_true, y_pred, zero_division=0) if metric == 'f1' else tpr + tnr - 1.0
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold, best_score = float(thr), float(score)
    return best_threshold, best_score

def save_loss_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
