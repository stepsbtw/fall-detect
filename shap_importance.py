import argparse
import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from config import Config
from utils import load_model_state, create_model, load_hyperparameters, load_test_data


def prepare_input_for_model(X, model_type):
    if model_type == "MLP":
        return X.reshape(X.shape[0], -1) if X.ndim > 2 else X
    elif model_type == "CNN1D":
        return X[:, np.newaxis, :, :] if X.ndim == 3 else X
    elif model_type == "LSTM":
        return X if X.ndim == 3 else X
    else:
        raise ValueError("Modelo não suportado")


def main():
    parser = argparse.ArgumentParser(description="SHAP Feature Importance para sensores")
    parser.add_argument("-scenario", required=True)
    parser.add_argument("-position", required=True)
    parser.add_argument("-label_type", required=True)
    parser.add_argument("--nn", required=True)
    parser.add_argument("--background_size", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=200)
    args = parser.parse_args()

    scenario = args.scenario
    position = args.position
    label_type = args.label_type
    model_type = args.nn
    num_labels = Config.get_num_labels(label_type)
    device = Config.DEVICE

    base_out = Config.get_output_dir(model_type, position, scenario, label_type)
    results = load_hyperparameters(base_out)
    best_params = results["best_params"]
    input_shape = Config.get_input_shape_dict(scenario, position, model_type)[model_type]
    model = create_model(model_type, best_params, input_shape, num_labels)

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

    X_shap = prepare_input_for_model(X_test, model_type)

    background = torch.tensor(X_shap[:args.background_size], dtype=torch.float32).to(device)
    sample = torch.tensor(X_shap[:args.sample_size], dtype=torch.float32).to(device)
    sample_np = sample.detach().cpu().numpy()

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
    else:
        model.eval()

    shap_out = os.path.join("analise_global", "shap")
    os.makedirs(shap_out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"{model_type}_{position}_{scenario}_{label_type}_{timestamp}"

    np.save(os.path.join(shap_out, f"shap_values_{prefix}.npy"), shap_values)

    if isinstance(shap_values, list):
        for class_idx, sv in enumerate(shap_values):
            if model_type == "MLP":
                mean_abs = np.abs(sv).mean(axis=0)
                try:
                    mean_abs = mean_abs.reshape(X_test.shape[1], X_test.shape[2])
                    mean_abs_feat = mean_abs.sum(axis=0)
                except:
                    mean_abs_feat = mean_abs
            elif model_type == "CNN1D":
                mean_abs_feat = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
            elif model_type == "LSTM":
                mean_abs_feat = np.abs(sv).mean(axis=(0, 1)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
            else:
                mean_abs_feat = np.abs(sv).mean(axis=0)

            mean_abs_feat = np.array(mean_abs_feat).flatten()
            df = pd.DataFrame({
                "feature": feature_names[:len(mean_abs_feat)],
                "mean_abs_shap": mean_abs_feat[:len(feature_names)]
            })
            csv_path = os.path.join(shap_out, f"shap_importance_class{class_idx}_{prefix}.csv")
            df.to_csv(csv_path, index=False)

            plt.figure(figsize=(10, 6))
            plt.bar(df["feature"], df["mean_abs_shap"])
            plt.ylabel("Importância média (|SHAP|)")
            plt.title(f"SHAP Feature Importance - {model_type} - Classe {class_idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_out, f"shap_importance_class{class_idx}_{prefix}.png"))
            plt.close()
    else:
        mean_abs_feat = np.abs(shap_values).mean(axis=0).flatten()
        df = pd.DataFrame({
            "feature": feature_names[:len(mean_abs_feat)],
            "mean_abs_shap": mean_abs_feat[:len(feature_names)]
        })
        csv_path = os.path.join(shap_out, f"shap_importance_{prefix}.csv")
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 6))
        plt.bar(df["feature"], df["mean_abs_shap"])
        plt.ylabel("Importância média (|SHAP|)")
        plt.title(f"SHAP Feature Importance - {model_type}")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_out, f"shap_importance_{prefix}.png"))
        plt.close()

    print("SHAP concluído!")


if __name__ == "__main__":
    main()