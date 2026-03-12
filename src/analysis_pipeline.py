"""
Unified analysis/output pipeline.

Usage:
    python analysis_pipeline.py shap           -scenario <s> --nn <m> [--background_size N] [--sample_size N]
    python analysis_pipeline.py learning_curve -scenario <s> [--nn <m>] [--epochs N]
    python analysis_pipeline.py aggregate      -scenario <s> --nn <m>
    python analysis_pipeline.py analyze        [--base_dir <dir>] [--output_dir <dir>]
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
import seaborn as sns
from datetime import datetime

from config import Config
from utils import (
    load_model_state, create_model, load_hyperparameters,
    load_test_data, plot_learning_curve,
)

SCENARIO_CHOICES = [
    "chest_T", # "chest_F",
    "left_T",  # "left_F",
    "right_T", # "right_F",
    # "chest_left_right_T", # "chest_left_right_F",
    "chest_left_T",  # "chest_left_F",
    "chest_right_T", # "chest_right_F",
]

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
    best_idx = metrics_df["F1"].idxmax() + 1
    best_model_path = os.path.join(base_out, f"model_{best_idx}", f"model_{best_idx}.pt")
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

    shap_out = os.path.join("analise_global", "shap")
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

    base_out = Config.get_output_dir(model_type_arg, scenario)

    results = load_hyperparameters(base_out)
    best_params = results["best_params"]
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg

    data = np.load(os.path.join(base_out, "test_data.npz"))
    X_trainval, y_trainval = data['X_trainval'], data['y_trainval']
    X_test, y_test = data['X_test'], data['y_test']

    input_shape = Config.get_input_shape_dict(scenario, model_type)[model_type]
    plot_learning_curve(
        create_model_fn=lambda bp, shape, nl: create_model(model_type, bp, shape, nl),
        X_full=X_trainval, y_full=y_trainval,
        X_test=X_test, y_test=y_test,
        input_shape=input_shape,
        num_labels=Config.NUM_LABELS,
        best_params=best_params,
        device=Config.DEVICE,
        output_dir=base_out,
        epochs=args.epochs,
    )

def _create_metric_visualizations(df, base_out):
    """Create and save metric visualisation plots."""
    plt.style.use('default')
    sns.set_palette("husl")

    metrics_cols = [c for c in ['F1', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC']
                    if c in df.columns]

    if metrics_cols:
        fig, ax = plt.subplots(figsize=(12, 8))
        df[metrics_cols].boxplot(ax=ax)
        ax.set_title('Distribuição das Métricas dos Modelos Finais', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor da Métrica', fontsize=12)
        ax.set_xlabel('Métrica', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, "metrics_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if 'F1' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['F1'], bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(df['F1'].mean(), color='red', linestyle='--',
                    label=f'Média: {df["F1"].mean():.4f}')
        plt.axvline(df['F1'].median(), color='green', linestyle='--',
                    label=f'Mediana: {df["F1"].median():.4f}')
        plt.xlabel('F1-Score', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição do F1-Score dos Modelos Finais', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, "f1_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if 'F1' in df.columns and 'Accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Accuracy'], df['F1'], alpha=0.7, s=50)
        plt.xlabel('Accuracy', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title('F1-Score vs Accuracy dos Modelos Finais', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, "f1_vs_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if len(metrics_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[metrics_cols].corr(), annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Matriz de Correlação das Métricas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print("\nEstatísticas das métricas:")
    for col in metrics_cols:
        print(f"{col}: média={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"min={df[col].min():.4f}  max={df[col].max():.4f}  mediana={df[col].median():.4f}")


def _aggregate_model_metrics(base_out):
    """Aggregate per-model CSVs into all_metrics.csv and summary_metrics.csv."""
    print(f"\n{'='*50}\nAGREGANDO MÉTRICAS DOS MODELOS FINAIS\n{'='*50}")

    all_metrics = []
    fold_dirs = sorted(glob.glob(os.path.join(base_out, "fold_s*")))
    for fold_dir in fold_dirs:
        subject_id = os.path.basename(fold_dir)[len("fold_"):]  # e.g. "s1"
        metrics_file = os.path.join(fold_dir, f"metrics_model_{subject_id}.csv")
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                all_metrics.append(df)
                print(f"Fold {subject_id}: F1={df['F1'].iloc[0]:.4f}, Acc={df['Accuracy'].iloc[0]:.4f}")
            except Exception as e:
                print(f"Erro ao ler métricas do fold {subject_id}: {e}")
        else:
            print(f"Arquivo não encontrado para fold {subject_id}: {metrics_file}")

    if not all_metrics:
        print("Nenhuma métrica encontrada!")
        return False

    combined_df = pd.concat(all_metrics, ignore_index=True)

    expected_columns = ['Model', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'F1',
                        'tp', 'tn', 'fp', 'fn']
    combined_df = combined_df[[c for c in expected_columns if c in combined_df.columns]]

    all_metrics_path = os.path.join(base_out, "all_metrics.csv")
    combined_df.to_csv(all_metrics_path, index=False)
    print(f"\nMétricas consolidadas salvas em: {all_metrics_path}")
    print(f"Total de modelos processados: {len(combined_df)}")

    numeric_cols = [c for c in combined_df.columns if c != 'Model']
    summary_df = combined_df[numeric_cols].describe().loc[['mean', 'std']].copy()
    summary_df.insert(0, 'Model', ['mean', 'std'])
    summary_df.to_csv(os.path.join(base_out, "summary_metrics.csv"))
    print(f"Estatísticas resumidas salvas em: {os.path.join(base_out, 'summary_metrics.csv')}")

    _create_metric_visualizations(combined_df, base_out)
    return True


def run_aggregate(args):
    """Aggregate metrics for a trained scenario."""
    base_out = Config.get_output_dir(args.model, args.scenario)
    print(f"Diretório de saída: {base_out}")

    if not os.path.exists(base_out):
        print(f"Erro: Diretório não encontrado: {base_out}\nExecute o treinamento final primeiro.")
        return

    success = _aggregate_model_metrics(base_out)
    banner = "AGREGAÇÃO DE MÉTRICAS CONCLUÍDA COM SUCESSO!" if success else "ERRO NA AGREGAÇÃO DE MÉTRICAS!"
    print(f"\n{'='*50}\n{banner}\n{'='*50}")

def _scan_output_dir(base_dir="output"):
    """Walk output directory and collect experiment summaries."""
    results = []
    for root, dirs, files in os.walk(base_dir):
        if "summary_metrics.csv" not in files:
            continue
        parts = root.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue
        nn = parts[-2]
        scenario = parts[-1]
        results.append({
            "model_type": nn,
            "position": "",
            "scenario": scenario,
            "label_type": "",
            "summary_metrics": os.path.join(root, "summary_metrics.csv"),
            "all_metrics": os.path.join(root, "all_metrics.csv") if "all_metrics.csv" in files else None,
            "learning_curve": os.path.join(root, "learning_curve_metrics.csv") if "learning_curve_metrics.csv" in files else None,
            "permutation_importance": os.path.join(root, "permutation_importance.csv") if "permutation_importance.csv" in files else None,
            "optuna_trials": os.path.join(root, "optuna_trials.csv") if "optuna_trials.csv" in files else None,
            "optuna_db": os.path.join(root, "optuna_study.db") if "optuna_study.db" in files else None,
        })
    return pd.DataFrame(results)


def _analyze_final_models(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    boxplot_root = os.path.join(output_dir, "boxplots")
    subfolder_map = {
        "Accuracy": "acc", "F1": "f1", "MCC": "mcc",
        "Precision": "prec", "Sensitivity": "sens", "Specificity": "spec", "all": "all",
    }
    for sub in subfolder_map.values():
        os.makedirs(os.path.join(boxplot_root, sub), exist_ok=True)

    summary_rows = []
    for _, row in df.iterrows():
        if not row["all_metrics"]:
            continue
        metrics_df = pd.read_csv(row["all_metrics"])
        metricas_plot = [c for c in ["F1", "Accuracy", "Precision", "Sensitivity", "Specificity", "MCC"]
                         if c in metrics_df.columns]
        if not metricas_plot:
            print(f"Nenhuma métrica reconhecida em {row['all_metrics']}, pulando.")
            continue

        tag = "_".join(x for x in [row["model_type"], row["position"], row["scenario"], row["label_type"]] if x)

        # All-metrics boxplot
        plt.figure()
        metrics_df[metricas_plot].boxplot()
        plt.title(f"Boxplot das métricas de validação {tag.replace('_', ' ')}")
        plt.ylabel("Valor da métrica")
        plt.xticks(rotation=45)
        plt.tight_layout()
        all_plot_path = os.path.join(boxplot_root, "all", f"boxplot_{tag}.png")
        plt.savefig(all_plot_path)
        plt.close()

        # Per-metric boxplots
        for met in metricas_plot:
            plt.figure()
            metrics_df.boxplot(column=met)
            plt.title(f"Boxplot de {met} de validação\n{tag.replace('_', ' ')}")
            plt.ylabel(f"Valor de {met}")
            plt.tight_layout()
            met_dir = os.path.join(boxplot_root, subfolder_map.get(met, met.lower()))
            os.makedirs(met_dir, exist_ok=True)
            plt.savefig(os.path.join(met_dir, f"boxplot_{tag}_{met}.png"))
            plt.close()

        stats = metrics_df.describe().loc[["mean", "std", "min", "max"]]
        summary = {"model_type": row["model_type"], "position": row["position"],
                   "scenario": row["scenario"], "label_type": row["label_type"],
                   "boxplot_path": all_plot_path}
        for met in metricas_plot:
            summary[f"{met}_mean"] = stats.loc["mean", met]
            summary[f"{met}_std"] = stats.loc["std", met]
        summary_rows.append(summary)

    summary_csv = os.path.join(output_dir, "summary_final_models.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Resumo dos modelos finais salvo em: {summary_csv}")


def _analyze_optuna_trials(df, output_dir):
    converg_dir = os.path.join(output_dir, "optuna", "convergencia")
    os.makedirs(converg_dir, exist_ok=True)
    for _, row in df.iterrows():
        if not row["optuna_trials"]:
            continue
        trials_df = pd.read_csv(row["optuna_trials"])
        if "value" not in trials_df.columns:
            continue
        tag = "_".join(x for x in [row["model_type"], row["position"], row["scenario"], row["label_type"]] if x)
        plt.figure(figsize=(8, 5))
        plt.plot(trials_df["value"].cummax(), marker='o')
        plt.xlabel("Trial")
        plt.ylabel("Melhor MCC médio de validação acumulado")
        plt.title(f"Optuna Convergência - {tag.replace('_', ' ')}")
        plt.grid(True); plt.tight_layout()
        path = os.path.join(converg_dir, f"optuna_convergencia_{tag}.png")
        plt.savefig(path); plt.close()
        print(f"Curva de convergência do Optuna salva em: {path}")


def _analyze_optuna_param_importance(df, output_dir):
    import optuna as _optuna
    paramimp_dir = os.path.join(output_dir, "optuna", "param_importance")
    os.makedirs(paramimp_dir, exist_ok=True)
    for _, row in df.iterrows():
        if not row["optuna_db"]:
            continue
        tag = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"
        try:
            storage = f"sqlite:///{row['optuna_db']}"
            studies = _optuna.study.get_all_study_summaries(storage=storage)
            if not studies:
                continue
            study = _optuna.load_study(study_name=studies[0].study_name, storage=storage)
            fig = _optuna.visualization.plot_param_importances(study)
            base_path = os.path.join(paramimp_dir, f"optuna_param_importance_{tag}")
            fig.write_html(base_path + ".html")
            fig.write_image(base_path + ".png")
            print(f"Importância dos hiperparâmetros salva em: {base_path}.html / .png")
        except Exception as e:
            print(f"[WARN] Não foi possível gerar importância dos hiperparâmetros para {tag}: {e}")


def run_analyze(args):
    """Run global analysis focused on summary, boxplots, and Optuna outputs."""
    df = _scan_output_dir(args.base_dir)
    print(f"Total de experimentos encontrados: {len(df)}")
    if df.empty:
        print("Nenhum experimento encontrado.")
        return
    out = args.output_dir
    _analyze_final_models(df, out)
    _analyze_optuna_trials(df, out)
    _analyze_optuna_param_importance(df, out)

def build_parser():
    parser = argparse.ArgumentParser(
        description="Fall-detect analysis pipeline: shap | learning_curve | aggregate | analyze",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_scenario_nn(p, nn_required=False):
        p.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
        p.add_argument("--model", required=nn_required, choices=["CNN1D", "MLP", "LSTM", "RF", "SVM", "XGBoost", "CatBoost"])

    # --- shap ---
    p_shap = subparsers.add_parser("shap", help="SHAP feature importance for the best model")
    add_scenario_nn(p_shap, nn_required=True)
    p_shap.add_argument("--background_size", type=int, default=100)
    p_shap.add_argument("--sample_size", type=int, default=200)

    # --- learning_curve ---
    p_lc = subparsers.add_parser("learning_curve", help="Generate learning curve")
    add_scenario_nn(p_lc)
    p_lc.add_argument("--epochs", type=int, default=10, help="Épocas por fração")

    # --- aggregate ---
    p_agg = subparsers.add_parser("aggregate", help="Aggregate per-model metrics")
    add_scenario_nn(p_agg, nn_required=True)

    # --- analyze ---
    p_ana = subparsers.add_parser("analyze", help="Global analysis of all experiments")
    p_ana.add_argument("--base_dir", default="output", help="Root output directory to scan")
    p_ana.add_argument("--output_dir", default="analise_global", help="Where to write analysis results")

    return parser


def main():
    Config.setup_device()
    Config.set_seed()

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
