import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import csv
import json

def analyze_optuna_study(scenario, position, metric="value"):
    study_name = f"{scenario}_{position}"
    save_dir = os.path.join("optuna_results", study_name)
    storage_path = f"sqlite:///{os.path.join(save_dir, 'optuna_study.db')}"

    os.makedirs(save_dir, exist_ok=True)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
    except Exception as e:
        print(f"[!] Failed to load study '{study_name}': {e}")
        return pd.DataFrame()

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]

    params_df = pd.json_normalize(trials_df["params"])
    result_df = pd.concat([trials_df[["number", "value"]], params_df], axis=1)
    result_df = result_df.sort_values(by=metric, ascending=False)

    # Save results
    result_df.to_csv(os.path.join(save_dir, f"{study_name}_results.csv"), index=False)
    with open(os.path.join(save_dir, f"{study_name}_best_params.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print(f"[✓] Results saved to: {save_dir}")
    print(f"[✓] Best trial ({metric}): {study.best_trial.value:.4f}")
    return result_df

def analyze_logged_trials(log_csv="optuna_logs/optuna_trials.csv", save_dir="optuna_results", metric="score"):
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(log_csv):
        raise FileNotFoundError(f"Arquivo de log CSV não encontrado: {log_csv}")
    
    # Carrega os trials
    df = pd.read_csv(log_csv)
    if df.empty:
        raise ValueError("O arquivo de log está vazio.")
    
    # Ordena pelo score
    df_sorted = df.sort_values(by=metric, ascending=False)

    # Salva resultados ordenados
    study_name = os.path.splitext(os.path.basename(log_csv))[0]
    csv_path = os.path.join(save_dir, f"{study_name}_sorted.csv")
    json_path = os.path.join(save_dir, f"{study_name}_best_params.json")
    df_sorted.to_csv(csv_path, index=False)

    # Extrai o melhor trial
    best_trial = df_sorted.iloc[0]
    best_params = best_trial.drop(["trial_number", metric]).to_dict()

    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"[✓] Resultados salvos em: {csv_path}")
    print(f"[✓] Best trial ({metric}): {best_trial[metric]:.4f}")

    # Plot: score x parâmetro, se "lr" existir
    if "lr" in df_sorted.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df_sorted, x="lr", y=metric, hue="trial_number", palette="viridis", s=100)
        plt.xscale("log")
        plt.title("Desempenho por taxa de aprendizado (lr)")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{study_name}_lr_plot.png"))
        plt.show()

    return df_sorted

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

