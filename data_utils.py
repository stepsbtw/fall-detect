# data_utils.py
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse

BATCH_SIZE = 64
array_sizes = {"chest": 1020, "right": 450, "left": 450}

targets_filename_and_size = {
    "multiple_one": ("multiple_class_label_1.npy", 37),
    "multiple_two": ("multiple_class_label_2.npy", 26),
    "binary_one": ("binary_class_label_1.npy", 2),
    "binary_two": ("binary_class_label_2.npy", 2),
}

def generate_datasets(data, label):
    X = torch.from_numpy(np.load(data))
    y = torch.from_numpy(np.load(label))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, stratify=y, random_state=101)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, stratify=y_test, random_state=101)

    X_train = torch.permute(X_train, (0, 2, 1))
    X_val = torch.permute(X_val, (0, 2, 1))
    X_test = torch.permute(X_test, (0, 2, 1))

    return X_train, y_train, X_val, y_val, X_test, y_test

def collect_datasets_from_input(position, target_type, scenario, label_dir, data_dir):
    array_size = array_sizes[position]

    neural_network_scenarios = {
        "Sc1_acc_T": ["magacc_time_domain_data_array.npy", (array_size, 1)],
        "Sc1_gyr_T": ["maggyr_time_domain_data_array.npy", (array_size, 1)],
        "Sc_2_acc_T": ["acc_x_y_z_axes_time_domain_data_array.npy", (array_size, 3)],
        "Sc_2_gyr_T": ["gyr_x_y_z_axes_time_domain_data_array.npy", (array_size, 3)],
        "Sc_3_T": ["magacc_and_maggyr_time_domain_data_array.npy", (array_size, 2)],
        "Sc_4_T": ["acc_and_gyr_three_axes_time_domain_data_array.npy", (array_size, 6)],
        "Sc1_acc_F": ["magacc_frequency_domain_data_array.npy", (array_size // 2, 1)],
        "Sc1_gyr_F": ["maggyr_frequency_domain_data_array.npy", (array_size // 2, 1)],
        "Sc_2_acc_F": ["acc_x_y_z_axes_frequency_domain_data_array.npy", (array_size // 2, 3)],
        "Sc_2_gyr_F": ["gyr_x_y_z_axes_frequency_domain_data_array.npy", (array_size // 2, 3)],
        "Sc_3_F": ["magacc_and_maggyr_frequency_domain_data_array.npy", (array_size // 2, 2)],
        "Sc_4_F": ["acc_and_gyr_three_axes_frequency_domain_data_array.npy", (array_size // 2, 6)],
    }

    label_file, label_size = targets_filename_and_size[target_type]
    data_file, input_shape = neural_network_scenarios[scenario]
    
    label_path = os.path.join(label_dir, label_file)
    data_path = os.path.join(data_dir, data_file)

    return (input_shape, label_size, *generate_datasets(data_path, label_path))


def generate_batches(X_train, y_train, X_val, y_val, X_test, y_test):
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    return train_dl, val_dl, test_dl


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario", required=True, choices=[...])
    parser.add_argument("-p", "--position", required=True, choices=["left", "chest", "right"])
    parser.add_argument("-nn", "--neural_network_type", required=True, choices=["CNN1D", "MLP"])

    # LSTM
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--num_dense_layers_lstm", type=int, default=2)
    parser.add_argument("--first_dense_size_lstm", type=int, default=128)
    parser.add_argument("--bidirectional", type=bool, default=True)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-c", "--n_conv", type=int, default=2)
    parser.add_argument("-d", "--n_dense", type=int, default=1)
    parser.add_argument("-l", "--label_type", choices=["binary_one", "binary_two"], default="binary_one")
    parser.add_argument("--export", action="store_true")

    return parser.parse_args()


def create_result_dir(base_dir, model_type, pos):
    result_dir = os.path.join(base_dir, "output", model_type, pos)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
