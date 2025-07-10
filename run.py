import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import train, save_results, run_optuna, plot_loss_curve
from neural_networks import CNN1DNet, MLPNet, LSTMNet
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # ----------------------------- #
    #          Dispositivo          #
    # ----------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Número de GPUs:", torch.cuda.device_count())
        
        # Configurar para usar múltiplas GPUs
        if torch.cuda.device_count() > 1:
            print("Configurando para usar múltiplas GPUs...")
            print(f"GPUs disponíveis: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ----------------------------- #
    #   Fixar Seeds (Reprodutível)  #
    # ----------------------------- #
    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed()

    # ----------------------------- #
    #         Argumentos CLI        #
    # ----------------------------- #
    parser = argparse.ArgumentParser(description="Otimização Bayesiana e Treinamento PyTorch")

    parser.add_argument("-scenario", required=True, choices=[
        "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
        "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
        "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
    ])
    parser.add_argument("-position", required=True, choices=["left", "chest", "right"])
    parser.add_argument("-label_type", required=True, choices=["multiple_one", "multiple_two", "binary_one", "binary_two"])
    parser.add_argument("--nn", required=False, choices=["CNN1D", "MLP", "LSTM"])

    args = parser.parse_args()

    # ----------------------------- #
    #         Configurações         #
    # ----------------------------- #
    position = args.position
    label_type = args.label_type
    scenario = args.scenario
    model_type_arg = args.nn
    num_labels = 37 if label_type == "multiple_one" else 26 if label_type == "multiple_two" else 2

    # Diretórios
    root_dir = os.path.dirname(__file__)
    data_path = os.path.join(root_dir, "labels_and_data", "data", position)
    label_path = os.path.join(root_dir, "labels_and_data", "labels", position)
    base_out = os.path.join(root_dir, "output", model_type_arg, position, scenario, label_type)
    os.makedirs(base_out, exist_ok=True)

    # Tamanhos dos vetores
    array_size = 1020 if position == "chest" else 450

    scenarios = {
        "Sc1_acc_T": ['magacc_time_domain_data_array.npy', (array_size, 1)],
        "Sc1_gyr_T": ['maggyr_time_domain_data_array.npy', (array_size, 1)],
        "Sc1_acc_F": ['magacc_frequency_domain_data_array.npy', (array_size // 2, 1)],
        "Sc1_gyr_F": ['maggyr_frequency_domain_data_array.npy', (array_size // 2, 1)],
        "Sc_2_acc_T": ['acc_x_y_z_axes_time_domain_data_array.npy', (array_size, 3)],
        "Sc_2_gyr_T": ['gyr_x_y_z_axes_time_domain_data_array.npy', (array_size, 3)],
        "Sc_2_acc_F": ['acc_x_y_z_axes_frequency_domain_data_array.npy', (array_size // 2, 3)],
        "Sc_2_gyr_F": ['gyr_x_y_z_axes_frequency_domain_data_array.npy', (array_size // 2, 3)],
        "Sc_3_T": ['magacc_and_maggyr_time_domain_data_array.npy', (array_size, 2)],
        "Sc_3_F": ['magacc_and_maggyr_frequency_domain_data_array.npy', (array_size // 2, 2)],
        "Sc_4_T": ['acc_and_gyr_three_axes_time_domain_data_array.npy', (array_size, 6)],
        "Sc_4_F": ['acc_and_gyr_three_axes_frequency_domain_data_array.npy', (array_size // 2, 6)],
    }

    labels_dict = {
        "multiple_one": "multiple_class_label_1.npy",
        "multiple_two": "multiple_class_label_2.npy",
        "binary_one": "binary_class_label_1.npy",
        "binary_two": "binary_class_label_2.npy",
    }

    # ----------------------------- #
    #     Carregar Dados e Split    #
    # ----------------------------- #
    X = np.load(os.path.join(data_path, scenarios[scenario][0]))
    y = np.load(os.path.join(label_path, labels_dict[label_type])).astype(np.int64)

    if model_type_arg == "LSTM":
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], -1, 1))

    # 80% para treino/validação (cross-validation), 20% para teste final
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------- #
    #     Otimização com Optuna     #
    # ----------------------------- #
    print("\n Iniciando Otimização com Optuna...\n")

    input_shape_dict = {
        "CNN1D": scenarios[scenario][1],
        "MLP": np.prod(scenarios[scenario][1]),
        "LSTM": X_trainval.reshape((X_trainval.shape[0], -1, scenarios[scenario][1][1])).shape[1:]
    }

    study = run_optuna(
        input_shape_dict=input_shape_dict,
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        output_dir=base_out,
        num_labels=num_labels,
        device=device,
        restrict_model_type=model_type_arg,
        study_name=f"{scenario}_{position}_{label_type}_{model_type_arg}" if model_type_arg else f"{scenario}_{position}_{label_type}"
    )


    best_params = study.best_params
    model_type = best_params["model_type"] if not model_type_arg else model_type_arg
    print("\n Melhor modelo:", model_type)
    print(" Melhores parâmetros:", best_params)

    # ----------------------------- #
    #      Treinamento Final        #
    # ----------------------------- #
    print("\n Iniciando Treinamento Final dos 20 Modelos com Avaliação no Teste...\n")

    # Ajustar input_shape
    if model_type == "CNN1D":
        input_shape = scenarios[scenario][1]
    elif model_type == "LSTM":
        input_shape = X.shape[1:]
    else:
        input_shape = np.prod(scenarios[scenario][1])

    for i in range(1, 21): #range(1,11)
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        random.seed(42 + i)
        print(f"\n--- Modelo {i} ---\n")
        model_dir = os.path.join(base_out, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)

        # Inicializar modelo
        if model_type == "CNN1D":
            model = CNN1DNet(
                input_shape=input_shape,
                filter_size=best_params["filter_size"],
                kernel_size=best_params["kernel_size"],
                num_layers=best_params["num_layers"],
                num_dense_layers=best_params["num_dense_layers"],
                dense_neurons=best_params["dense_units"],
                dropout=best_params["dropout"],
                number_of_labels=num_labels
            )
            batch_size = 32
        elif model_type == "MLP":
            model = MLPNet(
                input_dim=input_shape,
                num_layers=best_params["num_layers"],
                dense_neurons=best_params["dense_units"],
                dropout=best_params["dropout"],
                number_of_labels=num_labels
            )
            batch_size = 32
        elif model_type == "LSTM":
            model = LSTMNet(
                input_dim=input_shape[1],
                hidden_dim=best_params["hidden_dim"],
                num_layers=best_params["num_layers"],
                dropout=best_params["dropout"],
                number_of_labels=num_labels
            )
            batch_size = 32

        model.to(device, non_blocking=True)

        # Usar múltiplas GPUs se disponível
        if torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
            model = torch.nn.DataParallel(model)
            # Ajustar batch size para múltiplas GPUs
            batch_size = batch_size * torch.cuda.device_count()
            print(f"Batch size ajustado para {batch_size} (batch_size * num_gpus)")

        # treino final em val+train
        train_loader = DataLoader(TensorDataset(
            torch.tensor(X_trainval, dtype=torch.float32),
            torch.tensor(y_trainval, dtype=torch.long)
        ), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8
        )

        # corretamente usando teste
        test_loader = DataLoader(TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ), batch_size=batch_size, pin_memory=True, num_workers=8
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        criterion = nn.CrossEntropyLoss()

        y_pred, y_true, val_losses, train_losses = train(
            model, train_loader, test_loader,
            optimizer, criterion, device,
            epochs=25, early_stopping=True, patience=5
        )

        plot_loss_curve(train_losses, val_losses, model_dir, i)

        save_results(
            model=model,
            val_loader=test_loader,  
            y_val_onehot=y_test,     
            number_of_labels=num_labels,
            i=i,
            decision_threshold=best_params["decision_threshold"],
            output_dir=model_dir,
            device=device
        )

        with open(os.path.join(model_dir, "used_hyperparameters.json"), "w") as f:
            json.dump(best_params, f, indent=4)


    # Criar lista para armazenar métricas
    all_metrics = []

    # Caminho base onde estão salvos os modelos
    for i in range(1, 21):
        metrics_path = os.path.join(base_out, f"model_{i}", f"metrics_model_{i}.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            all_metrics.append(df.iloc[0])

    # Gerar DataFrame com todas as métricas
    metrics_df = pd.DataFrame(all_metrics)

    # Salvar todas as métricas em CSV
    all_metrics_path = os.path.join(base_out, "all_metrics.csv")
    metrics_df.to_csv(all_metrics_path, index=False)

    # Calcular média e desvio padrão
    summary = metrics_df.describe().loc[["mean", "std"]]

    # Salvar resumo estatístico
    summary_path = os.path.join(base_out, "summary_metrics.csv")
    summary.to_csv(summary_path)

    # Salvar boxplot de distribuição das principais métricas
    plt.figure(figsize=(10, 6))
    metricas_plot = ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity"]
    sns.boxplot(data=metrics_df[metricas_plot])
    plt.title("Distribuição das Métricas dos 20 Modelos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "metrics_boxplot.png"))
    plt.close()

    import shutil

    best_dir = os.path.join(base_out, "best_models")
    os.makedirs(best_dir, exist_ok=True)

    for i in range(1, 21):
        src = os.path.join(base_out, f"model_{i}", f"model_{i}.pt")
        dst = os.path.join(best_dir, f"model_{i}.pt")
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()