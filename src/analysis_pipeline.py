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
import shutil
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
    best_idx = metrics_df["MCC"].idxmax() + 1
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

    metrics_cols = [c for c in ['MCC', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity']
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

    if 'MCC' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['MCC'], bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(df['MCC'].mean(), color='red', linestyle='--',
                    label=f'Média: {df["MCC"].mean():.4f}')
        plt.axvline(df['MCC'].median(), color='green', linestyle='--',
                    label=f'Mediana: {df["MCC"].median():.4f}')
        plt.xlabel('MCC', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição do MCC dos Modelos Finais', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, "mcc_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if 'MCC' in df.columns and 'Accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Accuracy'], df['MCC'], alpha=0.7, s=50)
        plt.xlabel('Accuracy', fontsize=12)
        plt.ylabel('MCC', fontsize=12)
        plt.title('MCC vs Accuracy dos Modelos Finais', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_out, "mcc_vs_accuracy.png"), dpi=300, bbox_inches='tight')
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
    for i in range(1, 21):
        metrics_file = os.path.join(base_out, f"model_{i}", f"metrics_model_{i}.csv")
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                all_metrics.append(df)
                print(f"Modelo {i}: MCC={df['MCC'].iloc[0]:.4f}, Acc={df['Accuracy'].iloc[0]:.4f}")
            except Exception as e:
                print(f"Erro ao ler métricas do modelo {i}: {e}")
        else:
            print(f"Arquivo não encontrado para modelo {i}: {metrics_file}")

    if not all_metrics:
        print("Nenhuma métrica encontrada!")
        return False

    combined_df = pd.concat(all_metrics, ignore_index=True)
    combined_df['Model'] = combined_df['Model'].astype(float)

    expected_columns = ['Model', 'MCC', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy',
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
        parts = root.split(os.sep)
        if len(parts) < 5:
            continue
        nn, position, scenario, label = parts[-4:]
        results.append({
            "model_type": nn,
            "position": position,
            "scenario": scenario,
            "label_type": label,
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
        if "Precision" in metrics_df.columns and "Sensitivity" in metrics_df.columns:
            p, r = metrics_df["Precision"], metrics_df["Sensitivity"]
            metrics_df["F1"] = (2 * p * r / (p + r)).fillna(0)
        metricas_plot = [c for c in ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
                         if c in metrics_df.columns]
        if not metricas_plot:
            print(f"Nenhuma métrica reconhecida em {row['all_metrics']}, pulando.")
            continue

        tag = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"

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


def _analyze_learning_curves(df, output_dir):
    lc_root = os.path.join(output_dir, "learning_curves")
    for sub in ("metrics", "loss"):
        os.makedirs(os.path.join(lc_root, sub), exist_ok=True)

    for _, row in df.iterrows():
        if not row["learning_curve"]:
            continue
        lc_df = pd.read_csv(row["learning_curve"])
        tag = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"
        title_base = tag.replace('_', ' ')

        plt.figure(figsize=(10, 7))
        for col in ["MCC", "F1", "Accuracy"]:
            if col in lc_df.columns:
                plt.plot(lc_df["Fraction"] * 100, lc_df[col], marker='o', label=col)
        plt.xlabel("Porcentagem de Dados de Treino (%)")
        plt.ylabel("Valor da Métrica")
        plt.title(f"Learning Curve (Métricas) - {title_base}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        path = os.path.join(lc_root, "metrics", f"learning_curve_metrics_{tag}.png")
        plt.savefig(path); plt.close()
        print(f"Learning curve (métricas) salva em: {path}")

        plt.figure(figsize=(10, 7))
        for col in ["Train_Loss", "Val_Loss"]:
            if col in lc_df.columns:
                plt.plot(lc_df["Fraction"] * 100, lc_df[col], marker='o', label=col)
        plt.xlabel("Porcentagem de Dados de Treino (%)")
        plt.ylabel("Loss")
        plt.title(f"Learning Curve (Loss) - {title_base}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        path = os.path.join(lc_root, "loss", f"learning_curve_loss_{tag}.png")
        plt.savefig(path); plt.close()
        print(f"Learning curve (loss) salva em: {path}")


def _analyze_permutation_importance(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        if not row["permutation_importance"]:
            continue
        pi_df = pd.read_csv(row["permutation_importance"])
        tag = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"
        plt.figure(figsize=(10, 6))
        for col, color in zip(["delta_mcc", "delta_f1", "delta_acc"], ["C0", "C1", "C2"]):
            if col in pi_df.columns:
                plt.bar(pi_df["feature"], pi_df[col], alpha=0.7,
                        label=col.replace("delta_", "Δ").upper(), color=color)
        plt.ylabel("Queda na métrica ao embaralhar feature")
        plt.title(f"Permutation Importance - {tag.replace('_', ' ')}")
        plt.legend(); plt.tight_layout()
        path = os.path.join(output_dir, f"permutation_importance_{tag}.png")
        plt.savefig(path); plt.close()
        print(f"Permutation importance salva em: {path}")


def _analyze_optuna_trials(df, output_dir):
    converg_dir = os.path.join(output_dir, "optuna", "convergencia")
    os.makedirs(converg_dir, exist_ok=True)
    for _, row in df.iterrows():
        if not row["optuna_trials"]:
            continue
        trials_df = pd.read_csv(row["optuna_trials"])
        if "value" not in trials_df.columns:
            continue
        tag = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"
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


def _centralize_best_model_outputs(df, output_dir):
    """Copy best-model artefacts into themed subfolders in the global analysis dir."""
    file_patterns = {
        "confusion_matrix": "confusion_matrix_model_{}.png",
        "classification_report": "classification_report_model_{}.txt",
        "roc_curves": "roc_curve_model_{}.png",
        "loss_curves": "loss_curve_model_{}.png",
    }
    for folder in file_patterns:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    for _, row in df.iterrows():
        if not row["all_metrics"] or not os.path.exists(row["all_metrics"]):
            continue
        metrics_df = pd.read_csv(row["all_metrics"])
        if "MCC" not in metrics_df.columns:
            continue
        best_idx = str(metrics_df["MCC"].idxmax() + 1)
        exp_dir = os.path.dirname(row["summary_metrics"])
        tag = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"
        for folder, pattern in file_patterns.items():
            src = os.path.join(exp_dir, f"model_{best_idx}", pattern.format(best_idx))
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, folder, f"{tag}_{pattern.format(best_idx)}"))

    print(f"Arquivos do melhor modelo centralizados em: {output_dir}")


def run_analyze(args):
    """Run all global analysis steps over the output directory."""
    df = _scan_output_dir(args.base_dir)
    print(f"Total de experimentos encontrados: {len(df)}")
    if df.empty:
        print("Nenhum experimento encontrado.")
        return
    out = args.output_dir
    _analyze_final_models(df, out)
    _analyze_learning_curves(df, out)
    _analyze_permutation_importance(df, out)
    _analyze_optuna_trials(df, out)
    _analyze_optuna_param_importance(df, out)
    _centralize_best_model_outputs(df, out)

def build_parser():
    parser = argparse.ArgumentParser(
        description="Fall-detect analysis pipeline: shap | learning_curve | aggregate | analyze",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_scenario_nn(p, nn_required=False):
        p.add_argument("-scenario", required=True, choices=SCENARIO_CHOICES)
        p.add_argument("--model", required=nn_required, choices=["CNN1D", "MLP", "LSTM", "RF", "SVM", "XGBoost"])

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
