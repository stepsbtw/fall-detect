"""Unified analysis/output pipeline.

Usage:
    python analysis.py shap           -scenario <s> --model <m> [--background_size N] [--sample_size N]
    python analysis.py learning_curve -scenario <s> [--model <m>] [--epochs N]
    python analysis.py aggregate      -scenario <s> --model <m>
    python analysis.py analyze        [--base_dir <dir>] [--output_dir <dir>]
"""

import argparse
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit

from src.config import Config
from src.train import create_model
from src.test import (
    load_model_state,
    load_hyperparameters,
    load_test_data,
    plot_learning_curve,
)

SCENARIO_CHOICES = list(Config.SCENARIOS.keys())


def _build_human_readable_df(df, decimals=3):
    """Build a compact dataframe for TXT output (e.g., mean +- std columns)."""
    if df.empty:
        return df

    formatted = df.copy()

    mean_cols = [c for c in formatted.columns if c.endswith("_mean")]
    metric_names = []
    for mean_col in mean_cols:
        metric = mean_col[:-5]
        std_col = f"{metric}_std"
        if std_col in formatted.columns:
            metric_names.append(metric)

    if metric_names:
        for metric in metric_names:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            formatted[metric] = formatted.apply(
                lambda row: f"{row[mean_col]:.{decimals}f} +- {row[std_col]:.{decimals}f}",
                axis=1,
            )

        keep_cols = [
            c for c in formatted.columns
            if not (c.endswith("_mean") or c.endswith("_std") or c in metric_names)
        ]
        metric_cols = [m for m in metric_names if m in formatted.columns]
        formatted = formatted[keep_cols + metric_cols]

    numeric_cols = formatted.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        formatted[numeric_cols] = formatted[numeric_cols].round(decimals)

    # Shorter display names for TXT/TEX to keep table width manageable.
    rename_map = {
        "model_type": "Model",
        "scenario": "Data",
        "subject_id": "ID",
        "Accuracy": "Acc",
        "Precision": "Prec",
        "Sensitivity": "Recall",
        "tp": "TP",
        "fp": "FP",
        "tn": "TN",
        "fn": "FN",
    }
    formatted = formatted.rename(columns=rename_map)

    # Reorder columns for readability if present
    col_order = [
        "Model", "Data", "ID", "Acc", "Prec", "Recall", "f1", "TP", "FP", "TN", "FN"
    ]
    existing = [c for c in col_order if c in formatted.columns]
    rest = [c for c in formatted.columns if c not in existing]
    formatted = formatted[existing + rest]

    return formatted


def _save_csv_and_txt(df, csv_path, title, output_root=None):
    """Save dataframe as CSV, TXT, and LaTeX table.

    If output_root is provided, exports are split into:
    - <output_root>/csv/<relative_path>.csv
    - <output_root>/txt/<relative_path>.txt
    - <output_root>/tex/<relative_path>.tex
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    if output_root:
        csv_root = os.path.join(output_root, "csv")
        rel_csv_path = os.path.relpath(csv_path, csv_root)
        rel_base_no_ext = os.path.splitext(rel_csv_path)[0]
        txt_path = os.path.join(output_root, "txt", rel_base_no_ext + ".txt")
        tex_path = os.path.join(output_root, "tex", rel_base_no_ext + ".tex")
    else:
        txt_path = os.path.splitext(csv_path)[0] + ".txt"
        tex_path = os.path.splitext(csv_path)[0] + ".tex"

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)

    lines = [title, "=" * len(title), ""]
    readable_df = _build_human_readable_df(df)

    if readable_df.empty:
        lines.append("No rows available.")
    else:
        lines.append(f"Rows: {len(readable_df)}")
        lines.append(f"Columns: {len(readable_df.columns)}")
        lines.append("")
        lines.append(readable_df.to_string(index=False, max_colwidth=28))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    use_longtable = len(readable_df) > 30
    latex_df = readable_df.copy()
    numeric_cols = latex_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        latex_df[col] = latex_df[col].map(lambda x: f"{x:.3f}")

    latex_table = latex_df.to_latex(
        index=False,
        escape=True,
        caption=title,
        label=None,
        longtable=use_longtable,
    )
    # Improve readability in rendered LaTeX for mean/std summary cells.
    latex_table = latex_table.replace(" +- ", " $\\pm$ ")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)


def run_shap(args):
    """Compute and save SHAP feature importance for the best trained model."""
    import shap

    scenario = args.scenario
    model_type = args.model
    device = Config.DEVICE

    base_out = Config.get_output_dir(model_type, scenario)
    results = load_hyperparameters(base_out)
    best_params = results["best_params"]
    input_shape = Config.get_input_shape_dict(scenario, model_type)[model_type]
    model = create_model(model_type, best_params, input_shape, Config.NUM_LABELS)

    all_metrics_path = os.path.join(base_out, "all_metrics.csv")
    if not os.path.exists(all_metrics_path):
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {all_metrics_path}")
    metrics_df = pd.read_csv(all_metrics_path)
    best_row = metrics_df.loc[metrics_df["f1"].idxmax()]
    best_label = str(best_row.get("Model", "")).strip()
    if best_label.isdigit():
        best_label = f"s{best_label}"
    if not best_label:
        best_label = f"s{int(metrics_df['f1'].idxmax()) + 1}"

    best_model_path = os.path.join(
        Config.get_models_dir(model_type, scenario),
        f"fold_{best_label}",
        f"model_{best_label}.pt",
    )
    if not os.path.exists(best_model_path):
        legacy_path = os.path.join(base_out, f"model_{best_label}", f"model_{best_label}.pt")
        if os.path.exists(legacy_path):
            best_model_path = legacy_path
        else:
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {best_model_path}")
    model = load_model_state(model, best_model_path, device=str(device))
    model.to(device)

    X_test, y_test = load_test_data(base_out)
    feature_names = Config.get_feature_names(scenario)

    background = torch.tensor(X_test[:args.background_size], dtype=torch.float32).to(device)
    sample = torch.tensor(X_test[:args.sample_size], dtype=torch.float32).to(device)

    print(f"Rodando SHAP para {model_type}...")

    if model_type == "LSTM":
        torch.backends.cudnn.enabled = False
        model.train()
    else:
        model.eval()

    explainer = shap.DeepExplainer(model, background)

    if model_type == "LSTM":
        model.train()
    else:
        model.eval()

    shap_values = explainer.shap_values(sample, check_additivity=False)

    if model_type == "LSTM":
        model.eval()
        torch.backends.cudnn.enabled = True

    shap_out = os.path.join("analysis", "shap")
    os.makedirs(shap_out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"{model_type}_{scenario}_{timestamp}"

    np.save(os.path.join(shap_out, f"shap_values_{prefix}.npy"), shap_values)

    def _mean_abs_shap(sv, model_type):
        if model_type == "MLP":
            mean_abs = np.abs(sv).mean(axis=0)
            try:
                mean_abs = mean_abs.reshape(X_test.shape[1], X_test.shape[2])
                return mean_abs.sum(axis=0)
            except Exception:
                return mean_abs
        elif model_type == "CNN1D":
            return np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
        elif model_type == "LSTM":
            return np.abs(sv).mean(axis=(0, 1)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
        return np.abs(sv).mean(axis=0)

    def _save_shap_class(sv, class_suffix, prefix, shap_out, feature_names, model_type, title_suffix):
        feat = np.array(_mean_abs_shap(sv, model_type)).flatten()
        df = pd.DataFrame({
            "feature": feature_names[:len(feat)],
            "mean_abs_shap": feat[:len(feature_names)],
        })
        df.to_csv(os.path.join(shap_out, f"shap_importance{class_suffix}_{prefix}.csv"), index=False)
        plt.figure(figsize=(10, 6))
        plt.bar(df["feature"], df["mean_abs_shap"])
        plt.ylabel("Importância média (|SHAP|)")
        plt.title(f"SHAP Feature Importance - {model_type}{title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_out, f"shap_importance{class_suffix}_{prefix}.png"))
        plt.close()

    if isinstance(shap_values, list):
        for class_idx, sv in enumerate(shap_values):
            _save_shap_class(sv, f"_class{class_idx}", prefix, shap_out,
                             feature_names, model_type, f" - Classe {class_idx}")
    else:
        _save_shap_class(shap_values, "", prefix, shap_out, feature_names, model_type, "")

    print("SHAP concluído!")


def run_learning_curve(args):
    """Generate and save the learning curve for a scenario."""
    scenario = args.scenario
    model_type_arg = args.model
    loss_type = getattr(args, "loss", "weighted")

    scenario_out = scenario if loss_type == "weighted" else scenario + "_NW"
    base_out = Config.get_output_dir(model_type_arg, scenario_out)

    best_params = None
    model_type = model_type_arg

    hp_path = os.path.join(base_out, "best_hyperparameters.json")
    if os.path.exists(hp_path):
        results = load_hyperparameters(base_out)
        if isinstance(results, dict) and "best_params" in results:
            best_params = results["best_params"]
        else:
            best_params = results

        if model_type is None:
            model_type = best_params.get("model_type")

    if model_type is None:
        raise ValueError("--model e obrigatorio quando best_hyperparameters.json nao existe.")

    if model_type in Config.CLASSICAL_MODELS:
        raise ValueError("learning_curve suporta apenas modelos neurais: CNN1D, MLP, LSTM, GRU.")

    if best_params is None:
        best_params = dict(Config.DEFAULT_PARAMS[model_type])

    if "model_type" not in best_params:
        best_params["model_type"] = model_type

    test_data_path = os.path.join(base_out, "test_data.npz")
    X = np.load(Config.get_data_file(scenario))
    y = np.load(Config.get_labels_file(scenario)).astype(np.int64)
    groups = np.load(Config.get_groups_file(scenario))
    window_ids_path = os.path.join(os.path.dirname(Config.get_labels_file(scenario)), "window_ids.npy")
    window_ids = np.load(window_ids_path, allow_pickle=True) if os.path.exists(window_ids_path) else None
    # window_ids = np.load(Config.get_window_ids_file(scenario)) if os.path.exists(Config.get_window_ids_file(scenario)) else None

    needs_group_rebuild = True
    if os.path.exists(test_data_path):
        data = np.load(test_data_path)
        has_groups = "groups_trainval" in data and "groups_test" in data
        if has_groups:
            X_trainval, y_trainval = data["X_trainval"], data["y_trainval"]
            X_test, y_test = data["X_test"], data["y_test"]
            groups_trainval = data["groups_trainval"]
            needs_group_rebuild = False

    if needs_group_rebuild:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_trainval, y_trainval = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        groups_trainval = groups[train_idx]
        groups_test = groups[test_idx]

        os.makedirs(base_out, exist_ok=True)
        np.savez(
            test_data_path,
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            groups_trainval=groups_trainval,
            X_test=X_test,
            y_test=y_test,
            groups_test=groups_test,
        )
        print(f"test_data.npz atualizado com split por grupos em: {test_data_path}")

    input_shape = Config.get_input_shape_dict(scenario, model_type)[model_type]
    plot_learning_curve(
        create_model_fn=lambda bp, shape, nl: create_model(model_type, bp, shape, nl),
        X_full=X_trainval,
        y_full=y_trainval,
        groups_full=groups_trainval,
        X_test=X_test, y_test=y_test,
        input_shape=input_shape,
        num_labels=Config.NUM_LABELS,
        best_params=best_params,
        device=Config.DEVICE,
        output_dir=base_out,
        epochs=args.epochs,
        loss_type=loss_type,
    )


def _aggregate_model_metrics(base_out):
    """Aggregate per-model CSVs into all_metrics.csv and summary_metrics.csv."""
    print(f"\n{'='*50}\nAGREGANDO MÉTRICAS DOS MODELOS FINAIS\n{'='*50}")

    all_metrics = []
    fold_dirs = sorted(glob.glob(os.path.join(base_out, "fold_s*")))
    for fold_dir in fold_dirs:
        subject_id = os.path.basename(fold_dir)[len("fold_"):]  # e.g. "s1"
        metrics_file = os.path.join(fold_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                df.insert(0, 'Model', subject_id)
                all_metrics.append(df)
                print(f"Fold {subject_id}: f1={df['f1'].iloc[0]:.4f}, acc={df['acc'].iloc[0]:.4f}")
            except Exception as e:
                print(f"Erro ao ler métricas do fold {subject_id}: {e}")
        else:
            print(f"Arquivo não encontrado para fold {subject_id}: {metrics_file}")

    if not all_metrics:
        print("Nenhuma métrica encontrada!")
        return False

    combined_df = pd.concat(all_metrics, ignore_index=True)

    expected_columns = ['Model', 'f1', 'acc', 'prec', 'rec', 'tp', 'tn', 'fp', 'fn']
    combined_df = combined_df[[c for c in expected_columns if c in combined_df.columns]]

    all_metrics_path = os.path.join(base_out, "all_metrics.csv")
    combined_df.to_csv(all_metrics_path, index=False)
    print(f"\nMétricas consolidadas salvas em: {all_metrics_path}")
    print(f"Total de modelos processados: {len(combined_df)}")

    numeric_cols = [c for c in combined_df.columns if c not in ['subject_id', 'Model']]
    # Standard metrics table
    agg_metrics = [c for c in numeric_cols if c not in ['tp', 'fp', 'tn', 'fn']]
    summary_stats = combined_df[agg_metrics].describe().loc[['mean', 'std']].copy()
    summary_stats.insert(0, 'Model', ['mean', 'std'])
    summary_stats.to_csv(os.path.join(base_out, "summary_metrics_standard.csv"), index=False)

    # Confusion matrix table
    cm_cols = ['tp', 'fp', 'tn', 'fn']
    cm_sum = {col: combined_df[col].sum() if col in cm_cols else '' for col in combined_df.columns}
    cm_sum['Model'] = 'TOTAL'
    cm_sum['Total_P'] = cm_sum['tp'] + cm_sum['fp'] if cm_sum['tp'] != '' and cm_sum['fp'] != '' else ''
    cm_sum['Total_N'] = cm_sum['tn'] + cm_sum['fn'] if cm_sum['tn'] != '' and cm_sum['fn'] != '' else ''
    cm_sum['Total'] = sum([cm_sum[c] for c in cm_cols if cm_sum[c] != '']) if all(cm_sum[c] != '' for c in cm_cols) else ''
    cm_df = pd.DataFrame([cm_sum])
    cm_df = cm_df.rename(columns={"tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN"})
    cm_df = cm_df[[c for c in ["Model", "TP", "FP", "Total_P", "TN", "FN", "Total_N", "Total"] if c in cm_df.columns]]
    cm_df.to_csv(os.path.join(base_out, "summary_metrics_confusion.csv"), index=False)
    print(f"Estatísticas resumidas salvas em: {os.path.join(base_out, 'summary_metrics_standard.csv')} (standard)")
    print(f"Estatísticas resumidas salvas em: {os.path.join(base_out, 'summary_metrics_confusion.csv')} (confusion matrix)")

    return True


def run_aggregate(args):
    """Aggregate metrics for a trained scenario."""
    # If scenario contains _ (variant), use as output subdir directly; else use Config.get_output_dir
    if "_" in args.scenario:
        base_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", args.model, args.scenario)
    else:
        base_out = Config.get_output_dir(args.model, args.scenario)
    print(f"Diretório de saída: {base_out}")

    if not os.path.exists(base_out):
        print(f"Erro: Diretório não encontrado: {base_out}\nExecute o treinamento final primeiro.")
        return

    success = _aggregate_model_metrics(base_out)
    banner = "AGREGAÇÃO DE MÉTRICAS CONCLUÍDA COM SUCESSO!" if success else "ERRO NA AGREGAÇÃO DE MÉTRICAS!"
    print(f"\n{'='*50}\n{banner}\n{'='*50}")


def _scan_output_dir(base_dir="output"):
    """Walk output directory and collect experiment summaries.

    Supports:
    - output/<model>/<scenario>/all_metrics.csv
    - output/<model>/<scenario>/summary_metrics.csv
    - output/<model>/<scenario>/summary_metrics_*.csv
    - output/<model>/<scenario>/fold_s*/metrics.csv
    """
    results = []

    for root, _, files in os.walk(base_dir):
        parts = [p for p in root.replace("\\", "/").split("/") if p]
        if len(parts) < 3:
            continue

        # Expected experiment root: output/<model>/<scenario>
        if parts[-1] == "analysis" or parts[-2] == "analysis":
            continue

        summary_candidates = sorted(
            [f for f in files if f == "summary_metrics.csv" or f.startswith("summary_metrics_")]
        )
        has_all_metrics = "all_metrics.csv" in files
        fold_metric_files = sorted(glob.glob(os.path.join(root, "fold_s*", "metrics.csv")))
        has_fold_metrics = len(fold_metric_files) > 0

        if not summary_candidates and not has_all_metrics and not has_fold_metrics:
            continue

        model_type = parts[-2]
        scenario = parts[-1]

        prediction_files = sorted(glob.glob(os.path.join(root, "predictions_*.csv")))
        condition_fold_metric_files = sorted(glob.glob(os.path.join(root, "fold_metrics_*.csv")))

        results.append({
            "model_type": model_type,
            "scenario": scenario,
            "experiment_dir": root,
            "all_metrics": os.path.join(root, "all_metrics.csv") if has_all_metrics else None,
            "summary_metrics": os.path.join(root, summary_candidates[0]) if summary_candidates else None,
            "fold_metrics": fold_metric_files,
            "prediction_files": prediction_files,
            "condition_fold_metrics": condition_fold_metric_files,
        })

    return pd.DataFrame(results)


def _ensure_aggregated(df):
    """Create all_metrics.csv automatically when only fold metrics exist."""
    if df.empty:
        return df

    updated_rows = []

    for _, row in df.iterrows():
        row = row.copy()
        exp_dir = row["experiment_dir"]
        all_metrics_path = os.path.join(exp_dir, "all_metrics.csv")

        if not os.path.exists(all_metrics_path):
            fold_metrics = row.get("fold_metrics", [])
            if fold_metrics:
                print(f"Agregando automaticamente: model={row['model_type']} | scenario={row['scenario']}")
                success = _aggregate_model_metrics(exp_dir)
                if success and os.path.exists(all_metrics_path):
                    row["all_metrics"] = all_metrics_path

                    summary_candidates = sorted(
                        glob.glob(os.path.join(exp_dir, "summary_metrics*.csv"))
                    )
                    if summary_candidates:
                        row["summary_metrics"] = summary_candidates[0]

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)


def _compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "acc": accuracy,
        "prec": precision,
        "rec": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }



def _summarize_multisensor_prediction_file(predictions_path, model_type, scenario, condition_name=None):
    pred_df = pd.read_csv(predictions_path)
    if pred_df.empty or "y_true" not in pred_df.columns:
        return None, []

    group_col = None
    for candidate in ["group_id", "subject_id", "Model"]:
        if candidate in pred_df.columns:
            group_col = candidate
            break
    if group_col is None:
        return None, []

    pred_col = None
    for candidate in ["y_pred_stacked", "y_pred_fused", "y_pred"]:
        if candidate in pred_df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        prob_candidates = [c for c in pred_df.columns if c.startswith("y_prob")]
        if prob_candidates:
            pred_col = prob_candidates[0]
            pred_df["_tmp_pred"] = (pred_df[pred_col].to_numpy() >= 0.5).astype(int)
            pred_col = "_tmp_pred"
    if pred_col is None:
        return None, []

    if condition_name is None:
        basename = os.path.splitext(os.path.basename(predictions_path))[0]
        if basename.startswith("predictions_"):
            condition_name = basename[len("predictions_"):]
        else:
            condition_name = scenario

    scenario_label = f"{scenario}_{condition_name}"
    per_group_rows = []
    for group_value, group_df in pred_df.groupby(group_col):
        metrics = _compute_binary_metrics(group_df["y_true"].to_numpy(), group_df[pred_col].to_numpy())
        per_group_rows.append({
            "subject_id": group_value,
            "model_type": model_type,
            "scenario": scenario_label,
            "base_scenario": scenario,
            "condition": condition_name,
            **metrics,
        })

    if not per_group_rows:
        return None, []

    metrics_df = pd.DataFrame(per_group_rows)
    summary = {
        "model_type": model_type,
        "scenario": scenario_label,
        "base_scenario": scenario,
        "condition": condition_name,
    }
    for met in ["f1", "acc", "prec", "rec"]:
        summary[f"{met}_mean"] = metrics_df[met].mean()
        summary[f"{met}_std"] = metrics_df[met].std(ddof=1)
    for met, label in zip(["tp", "fp", "tn", "fn"], ["TP", "FP", "TN", "FN"]):
        summary[label] = metrics_df[met].sum()
    return summary, per_group_rows


def _subject_sort_key(value):
    s = str(value)
    try:
        return (0, int(float(s)))
    except Exception:
        return (1, s)


def _analyze_final_models(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_root = os.path.join(output_dir, "csv")
    os.makedirs(csv_root, exist_ok=True)

    summary_rows = []
    subject_rows = []
    for _, row in df.iterrows():
        prediction_files = row.get("prediction_files", [])
        used_prediction_breakdown = False

        if isinstance(prediction_files, list) and prediction_files:
            for prediction_file in prediction_files:
                condition_name = os.path.splitext(os.path.basename(prediction_file))[0].replace("predictions_", "", 1)
                summary, per_group_rows = _summarize_multisensor_prediction_file(
                    prediction_file,
                    model_type=row["model_type"],
                    scenario=row["scenario"],
                    condition_name=condition_name,
                )
                if summary is not None:
                    summary_rows.append(summary)
                    subject_rows.extend(per_group_rows)
                    used_prediction_breakdown = True

        if used_prediction_breakdown:
            continue

        all_metrics = row.get("all_metrics")
        summary_metrics = row.get("summary_metrics")

        metrics_source = None
        if pd.notna(all_metrics) and isinstance(all_metrics, (str, os.PathLike)) and all_metrics:
            metrics_source = all_metrics
        elif pd.notna(summary_metrics) and isinstance(summary_metrics, (str, os.PathLike)) and summary_metrics:
            metrics_source = summary_metrics

        if metrics_source is None or not os.path.exists(metrics_source):
            continue

        metrics_df = pd.read_csv(metrics_source)

        metrics_df = metrics_df.rename(columns={
            "Accuracy": "acc",
            "Precision": "prec",
            "Recall": "rec",
        })

        metricas_plot = [c for c in ["f1", "acc", "prec", "rec", "tp", "fp", "tn", "fn"] if c in metrics_df.columns]
        if not metricas_plot:
            print(f"Nenhuma métrica reconhecida em {metrics_source}, pulando.")
            continue

        agg_metrics = [m for m in metricas_plot if m not in ["tp", "fp", "tn", "fn"]]
        stats = metrics_df[agg_metrics].describe().loc[["mean", "std"]] if agg_metrics else pd.DataFrame()

        summary = {"model_type": row["model_type"], "scenario": row["scenario"]}
        for met in agg_metrics:
            summary[f"{met}_mean"] = stats.loc["mean", met]
            summary[f"{met}_std"] = stats.loc["std", met]
        for met, label in zip(["tp", "fp", "tn", "fn"], ["TP", "FP", "TN", "FN"]):
            if met in metrics_df.columns:
                summary[label] = metrics_df[met].sum()
        summary_rows.append(summary)

        subject_column = None
        if "subject_id" in metrics_df.columns:
            subject_column = "subject_id"
        elif "Model" in metrics_df.columns:
            subject_column = "Model"

        if subject_column:
            for _, metric_row in metrics_df.iterrows():
                subject_summary = {
                    "subject_id": metric_row[subject_column],
                    "model_type": row["model_type"],
                    "scenario": row["scenario"],
                }
                for met in metricas_plot:
                    subject_summary[met] = metric_row[met]
                subject_rows.append(subject_summary)

    summary_df = pd.DataFrame(summary_rows)
    subject_df = pd.DataFrame(subject_rows)

    summary_csv = os.path.join(csv_root, "summary_final_models.csv")
    _save_csv_and_txt(summary_df, summary_csv, "Final Models Summary", output_root=output_dir)
    print(f"Resumo dos modelos finais salvo em: {summary_csv}")

    if not summary_df.empty and "scenario" in summary_df.columns:
        scenario_dir = os.path.join(csv_root, "summary_final_models_by_scenario")
        os.makedirs(scenario_dir, exist_ok=True)
        for scenario in sorted(summary_df["scenario"].dropna().unique()):
            scenario_df = summary_df[summary_df["scenario"] == scenario].reset_index(drop=True)
            scenario_csv = os.path.join(scenario_dir, f"summary_final_models_{scenario}.csv")
            _save_csv_and_txt(
                scenario_df,
                scenario_csv,
                f"Final Models Summary - Scenario: {scenario}",
                output_root=output_dir,
            )

    if not summary_df.empty and "model_type" in summary_df.columns:
        model_type_dir = os.path.join(csv_root, "summary_final_models_by_model_type")
        os.makedirs(model_type_dir, exist_ok=True)
        for model_type in sorted(summary_df["model_type"].dropna().unique()):
            model_type_df = summary_df[summary_df["model_type"] == model_type].reset_index(drop=True)
            model_type_csv = os.path.join(model_type_dir, f"summary_final_models_{model_type}.csv")
            _save_csv_and_txt(
                model_type_df,
                model_type_csv,
                f"Final Models Summary - Model Type: {model_type}",
                output_root=output_dir,
            )

    subject_summary_csv = os.path.join(csv_root, "summary_final_models_by_subject.csv")
    _save_csv_and_txt(subject_df, subject_summary_csv, "Final Models Summary by Subject", output_root=output_dir)
    print(f"Resumo por subject salvo em: {subject_summary_csv}")

    if not subject_df.empty and "subject_id" in subject_df.columns:
        subject_dir = os.path.join(csv_root, "summary_final_models_by_subject")
        os.makedirs(subject_dir, exist_ok=True)

        def _subject_sort_key(value):
            s = str(value).strip()
            return (0, int(s)) if s.isdigit() else (1, s)

        unique_subject_ids = [sid for sid in subject_df["subject_id"].dropna().unique()]
        for subject_id in sorted(unique_subject_ids, key=_subject_sort_key):
            subject_metrics_df = subject_df[subject_df["subject_id"] == subject_id].reset_index(drop=True)
            subject_label = str(subject_id).strip()
            subject_csv = os.path.join(subject_dir, f"summary_final_models_{subject_label}.csv")
            _save_csv_and_txt(
                subject_metrics_df,
                subject_csv,
                f"Final Models Summary - Subject: {subject_label}",
                output_root=output_dir,
            )

    return summary_df, subject_df

def _analyze_per_model(summary_df, subject_df, base_dir):
    """Write per-model summary bundles inside each model folder."""
    if summary_df.empty or "model_type" not in summary_df.columns:
        return

    for model_type in sorted(summary_df["model_type"].dropna().unique()):
        model_summary_df = summary_df[summary_df["model_type"] == model_type].reset_index(drop=True)
        model_subject_df = (
            subject_df[subject_df["model_type"] == model_type].reset_index(drop=True)
            if not subject_df.empty and "model_type" in subject_df.columns
            else pd.DataFrame()
        )

        model_analysis_root = os.path.join(base_dir, model_type, "analysis")
        model_csv_root = os.path.join(model_analysis_root, "csv")
        os.makedirs(model_csv_root, exist_ok=True)

        model_summary_csv = os.path.join(model_csv_root, f"summary_final_models_{model_type}.csv")
        _save_csv_and_txt(
            model_summary_df,
            model_summary_csv,
            f"Model Summary - {model_type}",
            output_root=model_analysis_root,
        )

        if not model_summary_df.empty and "scenario" in model_summary_df.columns:
            by_scenario_dir = os.path.join(model_csv_root, "summary_by_scenario")
            os.makedirs(by_scenario_dir, exist_ok=True)
            for scenario in sorted(model_summary_df["scenario"].dropna().unique()):
                scenario_df = model_summary_df[model_summary_df["scenario"] == scenario].reset_index(drop=True)
                scenario_csv = os.path.join(by_scenario_dir, f"summary_{model_type}_{scenario}.csv")
                _save_csv_and_txt(
                    scenario_df,
                    scenario_csv,
                    f"Model Summary - {model_type} - Scenario: {scenario}",
                    output_root=model_analysis_root,
                )

        if not model_subject_df.empty:
            by_subject_csv = os.path.join(model_csv_root, f"summary_by_subject_{model_type}.csv")
            _save_csv_and_txt(
                model_subject_df,
                by_subject_csv,
                f"Model Subject Summary - {model_type}",
                output_root=model_analysis_root,
            )

            if "subject_id" in model_subject_df.columns:
                by_subject_dir = os.path.join(model_csv_root, "summary_by_subject")
                os.makedirs(by_subject_dir, exist_ok=True)
                unique_subject_ids = model_subject_df["subject_id"].dropna().unique()
                for subject_id in sorted(unique_subject_ids, key=_subject_sort_key):
                    subject_metrics_df = model_subject_df[model_subject_df["subject_id"] == subject_id].reset_index(drop=True)
                    subject_csv = os.path.join(by_subject_dir, f"summary_{model_type}_{subject_id}.csv")
                    _save_csv_and_txt(
                        subject_metrics_df,
                        subject_csv,
                        f"Model Subject Summary - {model_type} - Subject: {subject_id}",
                        output_root=model_analysis_root,
                    )

        print(f"Pacote de resumo por modelo salvo em: {model_analysis_root}")


def _write_master_analysis_tex(output_dir):
    """Create a minimal master LaTeX file with required packages and \\input statements."""
    tex_root = os.path.join(output_dir, "tex")
    if not os.path.exists(tex_root):
        return

    tex_files = sorted(
        [
            os.path.relpath(os.path.join(root, file_name), tex_root).replace("\\", "/")
            for root, _, files in os.walk(tex_root)
            for file_name in files
            if file_name.endswith(".tex") and file_name != "analysis.tex"
        ]
    )

    # Keep only compact summaries in the master document.
    tex_files = [
        rel_path
        for rel_path in tex_files
        if "by_subject" not in rel_path
    ]

    if not tex_files:
        return

    lines = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{caption}",
        "\\begin{document}",
    ]
    for rel_path in tex_files:
        escaped_rel_path = rel_path.replace("_", "\\_")
        lines.append(f"\\input{{{escaped_rel_path}}}")
    lines.append("\\end{document}")

    analysis_tex_path = os.path.join(tex_root, "analysis.tex")
    with open(analysis_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Master LaTeX de análise salvo em: {analysis_tex_path}")


def run_analyze(args):
    """Run global analysis focused on summary_final_models only."""
    df = _scan_output_dir(args.base_dir)
    print(f"Total de experimentos encontrados: {len(df)}")

    if df.empty:
        print("Nenhum experimento encontrado.")
        return

    df = _ensure_aggregated(df)

    valid_mask = df["all_metrics"].notna() | df["summary_metrics"].notna()
    df = df[valid_mask].reset_index(drop=True)

    print(f"Total de experimentos válidos após agregação: {len(df)}")
    if df.empty:
        print("Nenhum experimento agregado disponível para análise.")
        return

    out = args.output_dir
    summary_df, subject_df = _analyze_final_models(df, out)
    _write_master_analysis_tex(out)

    if not summary_df.empty and not subject_df.empty:
        _analyze_per_model(summary_df, subject_df, args.base_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fall-detect analysis pipeline: shap | learning_curve | aggregate | analyze",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_scenario_nn(p, nn_required=False):
        p.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
        p.add_argument("--model", required=nn_required, choices=list(Config.DEFAULT_PARAMS.keys()))

    # --- shap ---
    p_shap = subparsers.add_parser("shap", help="SHAP feature importance for the best model")
    add_scenario_nn(p_shap, nn_required=True)
    p_shap.add_argument("--background_size", type=int, default=100)
    p_shap.add_argument("--sample_size", type=int, default=200)

    # --- learning_curve ---
    p_lc = subparsers.add_parser("learning_curve", help="Generate learning curve")
    add_scenario_nn(p_lc)
    p_lc.add_argument("--epochs", type=int, default=Config.LEARNING_CURVE_CONFIG["epochs"], help="Épocas por fração")
    p_lc.add_argument(
        "--loss",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Loss weighting for neural learning curves: 'weighted' uses inverse-frequency class weights; 'unweighted' uses plain CrossEntropyLoss.",
    )

    # --- aggregate ---
    p_agg = subparsers.add_parser("aggregate", help="Aggregate per-model metrics")
    p_agg.add_argument("-scenario", required=True,
                       help="Scenario variant name (e.g. chest_T, chest_T_IVG1_SC_NM)")
    p_agg.add_argument("--model", required=True,
                       choices=list(Config.DEFAULT_PARAMS.keys()))

    # --- analyze ---
    p_ana = subparsers.add_parser("analyze", help="Global analysis of all experiments")
    p_ana.add_argument("--base_dir", default="output", help="Root output directory to scan")
    p_ana.add_argument("--output_dir", default="output/analysis", help="Where to write analysis results")

    return parser


def main(args=None):
    Config.setup_device()
    Config.set_seed()

    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    dispatch = {
        "shap": run_shap,
        "learning_curve": run_learning_curve,
        "aggregate": run_aggregate,
        "analyze": run_analyze,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
