# Criado por Rodrigo Parracho - https://github.com/RodrigoKasama

import optuna
import csv

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

import argparse

# Dados importantes

BATCH_SIZE = 32

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=101, stratify=y)
    
    # 20% + 20% para validação e teste
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=101, stratify=y_test)

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
    parser.add_argument(
        "-s", "--scenario",
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
    parser.add_argument(
        "-p", "--position",
        type=str,
        choices=["left", "chest", "right"],
        required=True,
        help="Referente a qual sensor será utilizado.",
    )
    parser.add_argument(
        "-l", "--label_type",
        type=str,
        choices=["binary_one", "binary_two"],
        # choices=["multiple_one", "multiple_two","binary_one", "binary_two"],
        default="binary_one",
        help="Type of classification problem Multi/Binary Classes",
    )
    parser.add_argument(
        "-nn", "--neural_network_type",
        type=str,
        choices=["CNN1D", "MLP"],
        required=True,
        help="Tipo de rede neural CNN1D ou MLP"
    )
    parser.add_argument("-e", "--epochs", type=check_positive, default=20,
                        help="Numero épocas de treinamento rede neural")
    parser.add_argument("-c", "--n_conv", type=check_positive, default=1,
                        help="Numero de sequencias de Convolução1D, ReLU, MaxPool1D e Dropout na rede neural")
    parser.add_argument("-d", "--n_dense", type=check_positive,
                        default=1, help="Numero de Camadas Densas na rede neural")

    args = parser.parse_args()

    return args.position, args.label_type, args.scenario, args.neural_network_type, args.n_conv, args.n_dense, args.epochs


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
    # Listas para armazenar todos os rótulos verdadeiros e predições
    all_labels = []
    all_predictions = []

    # Para economizar memória e tempo
    with torch.no_grad():
        for inputs, labels in test_dl:
            # A saida é uma logit, então tem que aplicar sigmoide
            outputs = model(inputs.float())

            # Conversão em probabilidades
            probabilities = torch.sigmoid(outputs.squeeze())

            # Limiar para converter probabilidades em predições binárias - se >= 0.5: 1. Do contrário, 0
            predicted = (probabilities >= 0.5).int()

            # Armazena as predições e os rótulos verdadeiros
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calcula e exibe o relatório de classificação
    report = classification_report(all_labels, all_predictions)
    return report


def generate_batches(X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_dl, val_dl, test_dl


def create_result_dir(current_directory, model_type, pos):
    output_dir = os.path.join(current_directory, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nn_results_dir = os.path.join(output_dir, model_type, pos)
    if not os.path.exists(nn_results_dir):
        os.makedirs(nn_results_dir)

    return nn_results_dir


# Funções não utilizadas voltadas p otimização de hiperparametros


def create_study_object(objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, neural_network_results_dir, number_of_labels, batch_size, training_epochs=25):

    study = optuna.create_study(direction="maximize")
    # Fará chamadas da função
    study.optimize(lambda trial: objective(trial, input_shape, X_train, y_train, X_val,
                                           y_val, neural_network_type, neural_network_results_dir, number_of_labels, training_epochs, batch_size), n_trials=5)

    best_trial = study.best_trial
    best_params = best_trial.params

    return best_trial, best_params


def objective(trial, input_shape, X_train, y_train, X_val, y_val, neural_network_type, output_dir, number_of_labels, training_epochs, batch_size):
    from sklearn.metrics import matthews_corrcoef

    mcc = None

    if neural_network_type == "CNN1D":

        # Fixando momentaneamente os hiperparâmetros
        filter_size = 50
        kernel_size = 5
        num_layers = 3
        num_dense_layers = 2
        dense_neurons = 100
        dropout = 0.3
        learning_rate = 0.0001
        decision_threshold = 0.5

        # Criando a arquitetura da rede neural de acordo com os hiperparametros e retornando um modelo treinado
        model, historic = cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size,
                                             kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs, batch_size)

        # SUSPEITA DE DATA LEAKAGE - O modelo treina com os dados de treinamento e validação. Após é coletado o mcc com base novamente nos dados de validaçãp
        # Coleta a predição do modelo
        y_pred_prob = model.predict(X_val)

        # Coleta com o threshold alterado
        y_pred = (y_pred_prob[:, 1] >= decision_threshold).astype(int)
        mcc = matthews_corrcoef(y_val.argmax(axis=1), y_pred)

        optimized_params = {
            "filter_size": filter_size,
            "kernel_size": kernel_size,
            "num_layers": num_layers,
            "num_dense_layers": num_dense_layers,
            "dense_neurons": dense_neurons,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "decision_threshold": decision_threshold
        }

        # Registra o score do conj de hiperparametros
        file_path = os.path.join(output_dir, "optimization_results.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as csvfile:

            fieldnames = ["Trial", "MCC"] + list(optimized_params.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {"Trial": trial.number, "MCC": mcc}
            row.update(optimized_params)
            writer.writerow(row)

    return mcc


def cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs, batch_size):

    # print(X_train.)
    print("X_train original", X_train)
    print("Shape Original:", X_train.shape)

    a = torch.permute(X_train, (0, 2, 1))
    print()
    print("X_train", a)
    print("X_train pivotado", a.shape)

    input()
    X_train = torch.permute(X_train, (0, 2, 1))
    X_val = torch.permute(X_val, (0, 2, 1))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNN1D(input_shape=input_shape, filter_size=filter_size, kernel_size=kernel_size, num_layers=num_layers,
                  num_dense_layers=num_dense_layers, dense_neurons=dense_neurons, dropout=dropout, number_of_labels=number_of_labels)

    print(model)
    # input()
    # Retorna um modelo treinado
    model = fit(model, X_train, y_train, X_val, y_val,
                learning_rate, nn.CrossEntropyLoss, training_epochs)
    return model


if __name__ == "__main__":
    print("Este é um arquivo auxiliar e não deve ser executado dessa forma! Verifique a documentação.")
