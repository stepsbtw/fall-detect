import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import csv
import json

def analyze_optuna_study(study_name, study_path="optuna_studies", save_dir="optuna_results", metric="value"):
    os.makedirs(save_dir, exist_ok=True)

    # Carrega o estudo
    storage = f"sqlite:///{os.path.join(study_path, study_name + '.db')}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    # DataFrame com os trials válidos
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]

    # Flatten dos hiperparâmetros
    params_df = pd.json_normalize(trials_df["params"])
    result_df = pd.concat([trials_df[["number", "value"]], params_df], axis=1)
    result_df = result_df.sort_values(by=metric, ascending=False)

    # Exporta para CSV e JSON
    csv_path = os.path.join(save_dir, f"{study_name}_results.csv")
    json_path = os.path.join(save_dir, f"{study_name}_best_params.json")
    result_df.to_csv(csv_path, index=False)

    best_trial = study.best_trial
    with open(json_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)

    print(f"[✓] Resultados salvos em: {csv_path}")
    print(f"[✓] Best trial ({metric}): {best_trial.value:.4f}")

    # Plot: F1 (ou outra métrica) x um dos parâmetros
    if "lr" in result_df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=result_df, x="lr", y=metric, hue="number", palette="viridis", s=100)
        plt.xscale("log")
        plt.title("Desempenho por taxa de aprendizado (lr)")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{study_name}_lr_plot.png"))
        plt.show()

    return result_df

def log_optuna_trial(trial, score, save_dir="optuna_logs"):
    os.makedirs(save_dir, exist_ok=True)
    
    params = trial.params
    params["trial_number"] = trial.number
    params["score"] = score

    # CSV
    csv_file = os.path.join(save_dir, "optuna_trials.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=params.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(params)

    # JSON
    json_file = os.path.join(save_dir, f"trial_{trial.number}.json")
    with open(json_file, "w") as f:
        json.dump(params, f, indent=4)

    print(f"[✓] Trial {trial.number} logado com score={score:.4f}")

