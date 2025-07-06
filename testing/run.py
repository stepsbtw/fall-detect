import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from testing.train import train, save_results, run_optuna
from testing.neural_networks import CNN1DNet, MLPNet, LSTMNet

# --- Argumentos CLI ---
parser = argparse.ArgumentParser(description="Otimização bayesiana + treinamento PyTorch")

parser.add_argument("--scenario", required=True, choices=[
    "Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
    "Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
    "Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
])
parser.add_argument("--position", required=True, choices=["left", "chest", "right"])
parser.add_argument("--label_type", required=True, choices=["multiple_one", "multiple_two", "binary_one", "binary_two"])
parser.add_argument("--neural_network_type", required=True, choices=["CNN1D", "MLP", "LSTM"])

args = parser.parse_args()

# --- Seleções do usuário ---
position = args.position
label_type = args.label_type
scenario = args.scenario
model_type = args.neural_network_type
num_labels = 37 if label_type == "multiple_one" else 26 if label_type == "multiple_two" else 2

# --- Diretórios e arquivos ---
root_dir = os.path.dirname(__file__)
data_path = os.path.join(root_dir, "labels_and_data", "data", position)
label_path = os.path.join(root_dir, "labels_and_data", "labels", position)

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

X = np.load(os.path.join(data_path, scenarios[scenario][0]))
y = np.load(os.path.join(label_path, labels_dict[label_type])).astype(np.int64)

# Ajustar input_shape
if model_type == "CNN1D":
    input_shape = scenarios[scenario][1]
elif model_type == "LSTM":
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], -1, 1))
    input_shape = X.shape[1:]
else:
    input_shape = X.shape[1]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# --- Dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --- Diretórios de saída ---
base_out = os.path.join(root_dir, "output", model_type.lower(), position, scenario, label_type)
os.makedirs(base_out, exist_ok=True)

# --- Otimização com Optuna ---
print("\n Iniciando Otimização\n")

# --- Precompute input shapes for all 3 model types ---
input_shape_dict = {
    "CNN1D": scenarios[scenario][1],
    "MLP": X.shape[1],
    "LSTM": X.reshape((X.shape[0], -1, X.shape[1] // scenarios[scenario][1][1])).shape[1:]
}

study = run_optuna(
    input_shape_dict,
    X_train, y_train,
    X_val, y_val,
    output_dir=base_out,
    num_labels=num_labels,
    device=device
)


best_params = study.best_params
print("\nMelhores parâmetros:", best_params)

# --- Treinar modelos com os melhores parâmetros ---
print("\n Iniciando Treinamento Final\n")

for i in range(1, 21):
    print(f"\n--- Modelo {i} ---\n")
    model_dir = os.path.join(base_out, f"model_{i}")
    os.makedirs(model_dir, exist_ok=True)

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
    elif model_type == "MLP":
        model = MLPNet(
            input_dim=input_shape,
            num_layers=best_params["num_layers"],
            dense_neurons=best_params["dense_units"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )
    elif model_type == "LSTM":
        model = LSTMNet(
            input_dim=input_shape[1],
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            number_of_labels=num_labels
        )

    model.to(device)

    # Loaders
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=32, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=32)

    # Treino
    import torch.nn as nn
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, device, epochs=25, early_stopping=True, patience=5)

    # Avaliação e salvamento
    save_results(
        model=model,
        val_loader=val_loader,
        y_val_onehot=y_val,
        number_of_labels=num_labels,
        i=i,
        decision_threshold=best_params["decision_threshold"],
        output_dir=model_dir,
        device=device
    )
