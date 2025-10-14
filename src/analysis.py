import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def scan_output_dir(base_dir="output"):
    results = []
    for root, dirs, files in os.walk(base_dir):
        if "summary_metrics.csv" in files:
            # Extrai info do path: output/NN/position/scenario/label/
            parts = root.split(os.sep)
            if len(parts) < 5:
                continue
            nn, position, scenario, label = parts[-4:]
            result = {
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
            }
            results.append(result)
    return pd.DataFrame(results)

def analyze_final_models(df, output_dir="analise_global"):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    # Diretórios organizados
    boxplot_root = os.path.join(output_dir, "boxplots")
    boxplot_subfolders = {
        "Accuracy": "acc",
        "F1": "f1",
        "MCC": "mcc",
        "Precision": "prec",
        "Sensitivity": "sens",
        "Specificity": "spec",
        "all": "all"
    }
    os.makedirs(boxplot_root, exist_ok=True)
    for sub in boxplot_subfolders.values():
        os.makedirs(os.path.join(boxplot_root, sub), exist_ok=True)
    for idx, row in df.iterrows():
        if not row["all_metrics"]:
            continue
        metrics_df = pd.read_csv(row["all_metrics"])
        # Calcular F1-score se possível
        if "Precision" in metrics_df.columns and "Sensitivity" in metrics_df.columns:
            prec = metrics_df["Precision"]
            rec = metrics_df["Sensitivity"]
            f1 = 2 * (prec * rec) / (prec + rec)
            f1 = f1.fillna(0)
            metrics_df["F1"] = f1
        # Seleciona apenas as colunas que existem
        metricas_plot = [col for col in ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1"] if col in metrics_df.columns]
        if not metricas_plot:
            print(f"Nenhuma métrica reconhecida em {row['all_metrics']}, pulando.")
            continue
        # Boxplot geral (all)
        plt.figure()
        metrics_df[metricas_plot].boxplot()
        #metricas_legenda = ", ".join(metricas_plot)
        plt.title(f"Boxplot das métricas de validação {row['model_type']} - {row['position']} - {row['scenario']} - {row['label_type']}")
        plt.ylabel(f"Valor da métrica")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(boxplot_root, "all", f"boxplot_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.png")
        plt.savefig(plot_path)
        plt.close()
        # Boxplots individuais por métrica
        for met in metricas_plot:
            plt.figure()
            metrics_df.boxplot(column=met)
            plt.title(f"Boxplot de {met} de validação\n{row['model_type']} - {row['position']} - {row['scenario']} - {row['label_type']}")
            plt.ylabel(f"Valor de {met}")
            plt.tight_layout()
            met_dir = os.path.join(boxplot_root, boxplot_subfolders.get(met, met.lower()))
            os.makedirs(met_dir, exist_ok=True)
            met_plot_path = os.path.join(met_dir, f"boxplot_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}_{met}.png")
            plt.savefig(met_plot_path)
            plt.close()
        # Resumo
        stats = metrics_df.describe().loc[["mean", "std", "min", "max"]]
        summary = {
            "model_type": row["model_type"],
            "position": row["position"],
            "scenario": row["scenario"],
            "label_type": row["label_type"],
            "boxplot_path": plot_path
        }
        for met in metricas_plot:
            summary[f"{met}_mean"] = stats.loc["mean", met]
            summary[f"{met}_std"] = stats.loc["std", met]
        summary_rows.append(summary)
    # Salva CSV mestre
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "summary_final_models.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Resumo das métricas dos modelos finais salvo em: {summary_csv}")

def analyze_learning_curves(df, output_dir="analise_global"):
    # Diretórios organizados
    lc_root = os.path.join(output_dir, "learning_curves")
    lc_metrics_dir = os.path.join(lc_root, "metrics")
    lc_loss_dir = os.path.join(lc_root, "loss")
    os.makedirs(lc_metrics_dir, exist_ok=True)
    os.makedirs(lc_loss_dir, exist_ok=True)
    for idx, row in df.iterrows():
        if not row["learning_curve"]:
            continue
        lc_df = pd.read_csv(row["learning_curve"])
        # Plot 1: Métricas
        plt.figure(figsize=(10, 7))
        for col in ["MCC", "F1", "Accuracy"]:
            if col in lc_df.columns:
                plt.plot(lc_df["Fraction"]*100, lc_df[col], marker='o', label=col)
        plt.xlabel("Porcentagem de Dados de Treino (%)")
        plt.ylabel("Valor da Métrica")
        plt.title(f"Learning Curve (Métricas) - {row['model_type']} - {row['position']} - {row['scenario']} - {row['label_type']}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_metrics = os.path.join(lc_metrics_dir, f"learning_curve_metrics_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.png")
        plt.savefig(plot_path_metrics)
        plt.close()
        print(f"Learning curve (métricas) salva em: {plot_path_metrics}")
        # Plot 2: Losses
        plt.figure(figsize=(10, 7))
        for col in ["Train_Loss", "Val_Loss"]:
            if col in lc_df.columns:
                plt.plot(lc_df["Fraction"]*100, lc_df[col], marker='o', label=col)
        plt.xlabel("Porcentagem de Dados de Treino (%)")
        plt.ylabel("Loss")
        plt.title(f"Learning Curve (Loss) - {row['model_type']} - {row['position']} - {row['scenario']} - {row['label_type']}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path_loss = os.path.join(lc_loss_dir, f"learning_curve_loss_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.png")
        plt.savefig(plot_path_loss)
        plt.close()
        print(f"Learning curve (loss) salva em: {plot_path_loss}")

def analyze_permutation_importance(df, output_dir="analise_global"):
    for idx, row in df.iterrows():
        if not row["permutation_importance"]:
            continue
        pi_df = pd.read_csv(row["permutation_importance"])
        plt.figure(figsize=(10, 6))
        for col, color in zip(["delta_mcc", "delta_f1", "delta_acc"], ["C0", "C1", "C2"]):
            if col in pi_df.columns:
                plt.bar(pi_df["feature"], pi_df[col], alpha=0.7, label=col.replace("delta_", "Δ").upper(), color=color)
        plt.ylabel("Queda na métrica ao embaralhar feature")
        plt.title(f"Permutation Importance - {row['model_type']} - {row['position']} - {row['scenario']} - {row['label_type']}")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"permutation_importance_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Permutation importance salva em: {plot_path}")

def analyze_optuna_trials(df, output_dir="analise_global"):
    # Diretórios organizados
    optuna_root = os.path.join(output_dir, "optuna")
    converg_dir = os.path.join(optuna_root, "convergencia")
    os.makedirs(converg_dir, exist_ok=True)
    for idx, row in df.iterrows():
        if not row["optuna_trials"]:
            continue
        trials_df = pd.read_csv(row["optuna_trials"])
        if "value" not in trials_df.columns:
            continue
        # Curva de convergência
        plt.figure(figsize=(8, 5))
        plt.plot(trials_df["value"].cummax(), marker='o')
        plt.xlabel("Trial")
        plt.ylabel("Melhor MCC médio de validação acumulado")
        plt.title(f"Optuna Convergência (MCC médio de validação) - {row['model_type']} - {row['position']} - {row['scenario']} - {row['label_type']}")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(converg_dir, f"optuna_convergencia_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Curva de convergência do Optuna salva em: {plot_path}")

def analyze_optuna_param_importance(df, output_dir="analise_global"):
    # Diretórios organizados
    optuna_root = os.path.join(output_dir, "optuna")
    paramimp_dir = os.path.join(optuna_root, "param_importance")
    os.makedirs(paramimp_dir, exist_ok=True)
    for idx, row in df.iterrows():
        if not row["optuna_db"]:
            continue
        try:
            import optuna
            study_name = None
            db_path = row["optuna_db"]
            storage = f"sqlite:///{db_path}"
            studies = optuna.study.get_all_study_summaries(storage=storage)
            if not studies:
                continue
            study_name = studies[0].study_name
            study = optuna.load_study(study_name=study_name, storage=storage)
            fig = optuna.visualization.plot_param_importances(study)
            # Salva como HTML
            html_path = os.path.join(paramimp_dir, f"optuna_param_importance_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.html")
            fig.write_html(html_path)
            # Salva como PNG
            png_path = os.path.join(paramimp_dir, f"optuna_param_importance_{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}.png")
            fig.write_image(png_path)
            print(f"Importância dos hiperparâmetros salva em: {html_path} e {png_path}")
        except Exception as e:
            print(f"[WARN] Não foi possível gerar importância dos hiperparâmetros para {row['model_type']} {row['position']} {row['scenario']} {row['label_type']}: {e}")

def centralize_best_model_outputs(df, output_dir="analise_global"):
    """
    Centraliza arquivos do melhor modelo (maior MCC) de cada experimento em subpastas temáticas na análise global.
    """
    folders = {
        "confusion_matrix": "confusion_matrix_model_{}.png",
        "classification_report": "classification_report_model_{}.txt",
        "roc_curves": "roc_curve_model_{}.png",
        "loss_curves": "loss_curve_model_{}.png"
    }
    for folder in folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    for idx, row in df.iterrows():
        exp_dir = os.path.dirname(row["summary_metrics"])
        all_metrics_path = row["all_metrics"]
        if not all_metrics_path or not os.path.exists(all_metrics_path):
            continue
        metrics_df = pd.read_csv(all_metrics_path)
        if "MCC" not in metrics_df.columns:
            continue
        best_idx = metrics_df["MCC"].idxmax() + 1  # model_1, model_2, ...
        model_str = str(best_idx)
        prefix = f"{row['model_type']}_{row['position']}_{row['scenario']}_{row['label_type']}"
        for folder, pattern in folders.items():
            fname = pattern.format(model_str)
            fpath = os.path.join(exp_dir, f"model_{model_str}", fname)
            if os.path.exists(fpath):
                target = os.path.join(output_dir, folder, f"{prefix}_{fname}")
                shutil.copy2(fpath, target)
    print(f"Arquivos do melhor modelo centralizados em: {output_dir}")

if __name__ == "__main__":
    df = scan_output_dir("output")
    print(df.head())
    print(f"Total de experimentos encontrados: {len(df)}")
    analyze_final_models(df)
    analyze_learning_curves(df)
    analyze_permutation_importance(df)
    analyze_optuna_trials(df)
    analyze_optuna_param_importance(df)
    centralize_best_model_outputs(df)