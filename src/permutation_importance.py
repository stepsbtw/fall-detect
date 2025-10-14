import argparse
import os
import numpy as np
import torch
from config import Config
from utils import load_model_state, create_model, load_hyperparameters, load_test_data
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd


def permutation_importance(model, X_test, y_test, device, feature_names):
    # X_test: (N, seq_len, n_features)
    base_preds = predict(model, X_test, device)
    base_mcc = matthews_corrcoef(y_test, base_preds)
    base_f1 = f1_score(y_test, base_preds, average="macro")
    base_acc = accuracy_score(y_test, base_preds)
    importances = []
    n_features = X_test.shape[2]
    for i in range(n_features):
        X_perm = X_test.copy()
        # Embaralhar todos os valores do canal i
        flat = X_perm[:, :, i].reshape(-1)
        np.random.shuffle(flat)
        X_perm[:, :, i] = flat.reshape(X_perm[:, :, i].shape)
        preds = predict(model, X_perm, device)
        mcc = matthews_corrcoef(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        importances.append({
            "feature": feature_names[i],
            "delta_mcc": base_mcc - mcc,
            "delta_f1": base_f1 - f1,
            "delta_acc": base_acc - acc,
            "mcc": mcc,
            "f1": f1,
            "acc": acc
        })
    return importances, base_mcc, base_f1, base_acc

def predict(model, X, device):
    model.eval()
    y_preds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.tensor(X[i:i+64], dtype=torch.float32).to(device)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)  # (batch, 1, seq_len, n_features)
            preds = model(xb)
            y_preds.append(torch.argmax(preds, dim=1).cpu().numpy())
    return np.concatenate(y_preds)

def main():
    parser = argparse.ArgumentParser(description="Permutation Feature Importance para sensores")
    parser.add_argument("-scenario", required=True)
    parser.add_argument("-position", required=True)
    parser.add_argument("-label_type", required=True)
    parser.add_argument("--nn", required=True)
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
    input_shape_dict = Config.get_input_shape_dict(scenario, position, model_type)
    if model_type == "CNN1D":
        input_shape = input_shape_dict["CNN1D"]
    elif model_type == "LSTM":
        input_shape = input_shape_dict["LSTM"]
    else:
        input_shape = input_shape_dict["MLP"]
    model = create_model(model_type, best_params, input_shape, num_labels)

    # Encontrar o melhor modelo pelo maior MCC
    all_metrics_path = os.path.join(base_out, "all_metrics.csv")
    if not os.path.exists(all_metrics_path):
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {all_metrics_path}")
    metrics_df = pd.read_csv(all_metrics_path)
    if "MCC" not in metrics_df.columns:
        raise ValueError("Coluna MCC não encontrada em all_metrics.csv")
    best_idx = metrics_df["MCC"].idxmax() + 1  # model_1, model_2, ...
    best_model_path = os.path.join(base_out, f"model_{best_idx}", f"model_{best_idx}.pt")
    model = load_model_state(model, best_model_path, device=device)
    model.to(device)

    X_test, y_test = load_test_data(base_out)
    feature_names = Config.get_feature_names(scenario)

    # Novo diretório de saída para permutation importance
    perm_out = os.path.join("analise_global", "permutation_importance")
    os.makedirs(perm_out, exist_ok=True)
    prefix = f"{model_type}_{position}_{scenario}_{label_type}"

    print(f"Rodando Permutation Feature Importance para {model_type}...")
    importances, base_mcc, base_f1, base_acc = permutation_importance(model, X_test, y_test, device, feature_names)

    # Salvar CSV
    df = pd.DataFrame(importances)
    csv_path = os.path.join(perm_out, f"permutation_importance_{prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Resultados salvos em: {csv_path}")

    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(df["feature"], df["delta_mcc"], label="ΔMCC")
    plt.bar(df["feature"], df["delta_f1"], alpha=0.7, label="ΔF1-score")
    plt.bar(df["feature"], df["delta_acc"], alpha=0.5, label="ΔAccuracy")
    plt.ylabel("Queda na métrica ao embaralhar feature")
    plt.title(f"Permutation Importance - {model_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(perm_out, f"permutation_importance_{prefix}.png"))
    plt.close()
    print(f"Gráfico salvo em: {os.path.join(perm_out, f'permutation_importance_{prefix}.png')}")
    print(f"Métricas originais: MCC={base_mcc:.4f}, F1={base_f1:.4f}, Acc={base_acc:.4f}")

if __name__ == "__main__":
    main() 