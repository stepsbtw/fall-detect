# Criado por Rodrigo Parracho - https://github.com/RodrigoKasama
# Adaptado por Caio Passos - https://github.com/stepsbtw

import csv
import json
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

import argparse
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from neural_architectures import CustomMLP, CNN1D, CustomLSTM

# Dados importantes

BATCH_SIZE = 64  # 128

# Quantidade de leituras a cada 5s -> Passo de tempo
array_sizes = {"chest": 1020, "right": 450, "left": 450}

# Nome do arquivo dos targets e quantidade de classes
targets_filename_and_size = {
    # O problema multiclasse não funciona por enquanto
    "multiple_one": ("multiple_class_label_1.npy", 37),
    # O problema multiclasse não funciona por enquanto
    "multiple_two": ("multiple_class_label_2.npy", 26),
    "binary_one": ("binary_class_label_1.npy", 2),
    "binary_two": ("binary_class_label_2.npy", 2),
}


def generate_datasets(data: str = None, label: str = None):
    # Antigo generate_training_testing_and_validation_sets()
    # Carregando os dados e os targuets
    X = np.load(data)
    y = np.load(label)

    # Convertendo para tensores
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # 60% para treinamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, random_state=101, stratify=y)

    # 20% + 20% para validação e teste
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.4, random_state=101, stratify=y_test)

    # É necessário "pivotar" o datset devido a forma como o pytorch interpreta as camadas dos tensores ([batch, features, passo_de tempo])
    X_train = torch.permute(X_train, (0, 2, 1))
    X_val = torch.permute(X_val, (0, 2, 1))
    X_test = torch.permute(X_test, (0, 2, 1))

    return X_train, y_train, X_val, y_val, X_test, y_test


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"O valor {ivalue} deve ser maior que 0")
    return ivalue


def parse_input():
    parser = argparse.ArgumentParser(description="Script for model training")

    # Argumentos obrigatórios
    parser.add_argument("-s", "--scenario",
                        type=str,
                        choices=[
                            # Cenários sem transformada de fourier
                            # Univariada
                            "Sc1_acc_T", "Sc1_gyr_T",
                            # Multivariada (x, y, z)
                            "Sc_2_acc_T", "Sc_2_gyr_T",
                            # Multivariada - Aceleração Linear e Angular (2)
                            "Sc_3_T",
                            # Multivariada - (x, y, z)-Linear e (x, y, z)-Angular (6)
                            "Sc_4_T",

                            # Cenários com transformada de fourier
                            "Sc1_acc_F", "Sc1_gyr_F", "Sc_2_acc_F", "Sc_2_gyr_F", "Sc_3_F", "Sc_4_F"
                        ],
                        required=True,
                        help="Possiveis Cenários a se trabalhar.Cenários com *_F referem-se a transformada de fourier entrada, enquanto que *_T são leituras sem transformação.\n Cenários com *_acc_* referem-se a leitura de aceleração LINEAR, enquanto que Cenários com *_gyr_* referem-se à leitura de aceleração ANGULAR.",
                        )
    parser.add_argument("-p", "--position",
                        type=str,
                        choices=["left", "chest", "right"],
                        required=True,
                        help="Referente a qual sensor será utilizado.",
                        )
    parser.add_argument("-nn", "--neural_network_type",
                        type=str,
                        choices=["CNN1D", "MLP"],
                        required=True,
                        help="Tipo de rede neural CNN1D ou MLP"
                        )
    # Parâmetros específicos do LSTM
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Tamanho do hidden state na LSTM. Default: 128")
    parser.add_argument("--num_lstm_layers", type=int, default=2,
                        help="Número de camadas LSTM. Default: 2")
    parser.add_argument("--num_dense_layers_lstm", type=int, default=2,
                        help="Número de camadas densas após a LSTM. Default: 2")
    parser.add_argument("--first_dense_size_lstm", type=int, default=128,
                        help="Tamanho inicial da primeira camada densa do LSTM. Default: 128")
    parser.add_argument("--bidirectional", type=bool, default=True,
                        help="Se a LSTM será bidirecional. Default: True")


    # Argumentos opcionais
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Taxa de aprendizado da rede neural. Default: 0.001")
    parser.add_argument("-e", "--epochs", type=check_positive, default=20,
                        help="Numero épocas de treinamento rede neural. Defalut: 20")
    parser.add_argument("-c", "--n_conv", type=check_positive, default=2,
                        help="Numero de sequencias de Convolução1D + ReLU + MaxPool1D + Dropout na rede neural. Default: 2")
    parser.add_argument("-d", "--n_dense", type=check_positive, default=1,
                        help="Numero de camadas ocultas na rede neural. Default: 1")
    parser.add_argument("-l", "--label_type", type=str, default="binary_one",
                        choices=["binary_one", "binary_two"],
                        help="Tipo de classificação binária. Default: 'binary_one'",
                        )
    parser.add_argument("--export", action="store_true",
                        help="Marcador para exportar o gráfico de aprendizado, o relatório de classificação e o modelo. Default: FALSE",
                        )

    args = parser.parse_args()

    return args


def collect_datasets_from_input(position, target_type, scenario, label_dir, data_dir):

    array_size = array_sizes[position]

    # Para cada cenário cria um dict com o diretório do dado e o shape de entrada
    neural_network_scenarios = {
        # Leitura da magnitude (SQRT(x² + y² + z²)) da aceleração linear
        "Sc1_acc_T": [os.path.join(data_dir, "magacc_time_domain_data_array.npy"), (array_size, 1)],
        # Leitura da magnitude (SQRT(x² + y² + z²)) da aceleração angular
        "Sc1_gyr_T": [os.path.join(data_dir, "maggyr_time_domain_data_array.npy"), (array_size, 1)],
        # Leitura dos exios (x, y, z) da aceleração linear - > Passa a ter 3 features | Problema multivariado
        "Sc_2_acc_T": [os.path.join(data_dir, "acc_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
        # Leitura dos exios (x, y, z) da aceleração angular - > Passa a ter 3 features | Problema multivariado
        "Sc_2_gyr_T": [os.path.join(data_dir, "gyr_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
        # Leitura da magnitude (SQRT(x² + y² + z²)) da aceleração linear e da aceleração angular - > Passa a ter 2 features | Problema multivariado
        "Sc_3_T": [os.path.join(data_dir, "magacc_and_maggyr_time_domain_data_array.npy"), (array_size, 2)],
        # Leitura dos exios (x, y, z) da aceleração linear E (x, y, z) da aceleração angular - > Passa a ter 6 features | Problema multivariado
        "Sc_4_T": [os.path.join(data_dir, "acc_and_gyr_three_axes_time_domain_data_array.npy"), (array_size, 6)],

        # Também foi realizado uma uma transformada de fourier que mostrou-se promissora na classificação
        # - Por conta da caracteristica da transformada, o resultado é uma função espelhada, para resolver esse problema segmentamos a duplicata da transformada
        "Sc1_acc_F": [os.path.join(data_dir, "magacc_frequency_domain_data_array.npy"), (int(array_size/2), 1)],
        "Sc1_gyr_F": [os.path.join(data_dir, "maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 1)],
        "Sc_2_acc_F": [os.path.join(data_dir, "acc_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],
        "Sc_2_gyr_F": [os.path.join(data_dir, "gyr_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],
        "Sc_3_F": [os.path.join(data_dir, "magacc_and_maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 2)],
        "Sc_4_F": [os.path.join(data_dir, "acc_and_gyr_three_axes_frequency_domain_data_array.npy"), (int(array_size/2), 6)],
    }

    # O nome do arquivo de dados e o formato de entrada da RN será definido de acordo com neural_network_scenarios.

    label_filename, label_size = targets_filename_and_size.get(target_type)
    data_filename, input_shape = neural_network_scenarios[scenario]

    #  O arquivo de targets é label_dir + label_filename
    label_path = os.path.join(label_dir, label_filename)

    X_train, y_train, X_val, y_val, X_test, y_test = generate_datasets(
        data_filename, label_path)

    return input_shape, label_size, X_train, y_train, X_val, y_val, X_test, y_test


def save_loss_curve(train_loss: list, valid_loss: list, image_dir: str = "./", filename: str = "plot_loss_curve"):
    fig = plt.figure(figsize=(10, 8))

    plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(image_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    pass

def get_class_report(model, test_dl):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            probs = torch.sigmoid(outputs.squeeze())

            preds = (probs >= 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_predictions, zero_division=0)
    dict_report = classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None

    return report, dict_report, conf_matrix, all_labels, all_probs, auc



def plot_confusion_matrix(cm, path, labels=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if cm is None:
        print("Aviso: matriz de confusão está vazia (None).")
        return

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels is not None else range(cm.shape[0]),
                yticklabels=labels if labels is not None else range(cm.shape[0]))
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_mlp_feature_importance(model: CustomMLP, input_shape: tuple, output_path: str):
    if input_shape[1] > 1:
        num_features = input_shape[0] * input_shape[1]
    else:
        num_features = input_shape[0]
    weights = model.layers[0].weight.detach().cpu().numpy().mean(axis=0)

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), weights)
    plt.title("Feature Importance (First Layer Weights)")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def generate_batches(X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    return train_dl, val_dl, test_dl


def create_result_dir(current_directory, model_type, pos):
    output_dir = os.path.join(current_directory, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nn_results_dir = os.path.join(output_dir, model_type, pos)
    if not os.path.exists(nn_results_dir):
        os.makedirs(nn_results_dir)

    return nn_results_dir

def export_result(scenario, neural_network_type, position, test_report):
    filename = f"results/{scenario}_{neural_network_type}_{position}.json"
    with open(filename, "w") as f:
        json.dump(test_report, f, indent=4)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def show_datasets_info(X_train, y_train, X_val, y_val, X_test, y_test):
    def class_dist(y):
        y = y.numpy()
        return {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    info = {
        "X_train": list(X_train.shape),
        "y_train_dist": class_dist(y_train),
        "X_val": list(X_val.shape),
        "y_val_dist": class_dist(y_val),
        "X_test": list(X_test.shape),
        "y_test_dist": class_dist(y_test),
    }

    msg = "\n--- Dataset Info ---\n"
    msg += f"Train Set: {info['X_train']}, Class Dist: {info['y_train_dist']}\n"
    msg += f"Val Set:   {info['X_val']}, Class Dist: {info['y_val_dist']}\n"
    msg += f"Test Set:  {info['X_test']}, Class Dist: {info['y_test_dist']}\n"
    msg += "-" * 24
    return msg



# Dispositivo para rodar (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuração do benchmark para CUDNN, se estiver usando GPU
# Isso pode melhorar o desempenho para redes neurais com entradas de tamanho fixo
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
