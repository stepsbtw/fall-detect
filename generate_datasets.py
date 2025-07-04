# Criado por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Rodrigo Parracho - https://github.com/RodrigoKasama
# Adaptado por Caio Passos - https://github.com/stepsbtw

import argparse
import os
import numpy as np
from scipy.signal import resample
from builders.data_training_builders import (
    sort_by_number,
    get_file_path,
    create_dataframe,
    create_directory_if_does_not_exist
)
from builders.data_training_generators import generate_activities

data_arrays_time_domain = [[] for _ in range(8)]
data_arrays_frequency_domain = [[] for _ in range(8)]
labels_list = [[] for _ in range(4)]

# ========== ARGUMENTS ==========
parser = argparse.ArgumentParser(description="Geração de datasets e rótulos para diferentes cenários")
parser.add_argument("position", type=str, choices=["chest", "left", "right", "all"], help="Sensor position ou 'all' para fusão")
args = parser.parse_args()
position = args.position.lower()

# ========== DIRECTORIES ==========
current_directory = os.path.dirname(__file__)
main_directory = os.path.join(current_directory, "database")
label_directory = os.path.join(current_directory, "labels_and_data", "labels", position)
data_array_directory = os.path.join(current_directory, "labels_and_data", "data", position if position != "all" else "")

create_directory_if_does_not_exist(label_directory)
create_directory_if_does_not_exist(data_array_directory)

# ========== HANDLE FUSION MODE ==========
def resample_to_target(X, target_len):
    return np.array([resample(sample, target_len, axis=0) for sample in X])

def generate_fused_dataset():
    print("\n[✓] Gerando conjunto de dados combinado (tempo)...")

    base_path = os.path.join(current_directory, "labels_and_data", "data")
    X_chest = np.load(os.path.join(base_path, "chest", "acc_and_gyr_three_axes_time_domain_data_array.npy"))
    X_left  = np.load(os.path.join(base_path, "left",  "acc_and_gyr_three_axes_time_domain_data_array.npy"))
    X_right = np.load(os.path.join(base_path, "right", "acc_and_gyr_three_axes_time_domain_data_array.npy"))

    X_left_resampled = resample_to_target(X_left, target_len=X_chest.shape[1])
    X_right_resampled = resample_to_target(X_right, target_len=X_chest.shape[1])

    X_fused = np.concatenate([X_chest, X_left_resampled, X_right_resampled], axis=2)
    np.save(os.path.join(base_path, "combined_time_domain_data.npy"), X_fused)

    print(f"  ✔ Salvo: combined_time_domain_data.npy — shape: {X_fused.shape}")

def generate_fused_frequency_dataset():
    print("\n[✓] Gerando conjunto de dados combinado (frequência)...")

    base_path = os.path.join(current_directory, "labels_and_data", "data")
    X_chest = np.load(os.path.join(base_path, "chest", "acc_and_gyr_three_axes_frequency_domain_data_array.npy"))
    X_left  = np.load(os.path.join(base_path, "left",  "acc_and_gyr_three_axes_frequency_domain_data_array.npy"))
    X_right = np.load(os.path.join(base_path, "right", "acc_and_gyr_three_axes_frequency_domain_data_array.npy"))

    X_left_resampled = resample_to_target(X_left, target_len=X_chest.shape[1])
    X_right_resampled = resample_to_target(X_right, target_len=X_chest.shape[1])

    X_fused = np.concatenate([X_chest, X_left_resampled, X_right_resampled], axis=2)
    np.save(os.path.join(base_path, "combined_frequency_domain_data.npy"), X_fused)

    print(f"  ✔ Salvo: combined_frequency_domain_data.npy — shape: {X_fused.shape}")

if position == "all":
    generate_fused_dataset()
    generate_fused_frequency_dataset()
    exit()

# ========== SINGLE SENSOR MODE ==========
print("Obtendo dados da posição:", position.upper())
subdirectory_list = os.listdir(main_directory)
subdirectory_list.sort(key=sort_by_number)

print("Criando diretórios de labels e data_arrays...")
create_directory_if_does_not_exist(label_directory)
create_directory_if_does_not_exist(data_array_directory)

print("Obtendo os dados de cada usuário...")
for subdirectory in subdirectory_list:
    print(f"  {subdirectory}...", end="")
    acc, gyr, sampling = get_file_path(main_directory, subdirectory, position.upper())
    acc_df, gyr_df, sampling_df = create_dataframe(acc, gyr, sampling)
    generate_activities(acc_df, gyr_df, sampling_df, position.upper(), data_arrays_time_domain, data_arrays_frequency_domain, labels_list)
    print("OK")

print("Salvando rótulos de cada caso...")
for i in range(4):
    prefix = "multiple" if i < 2 else "binary"
    suffix = "1" if (i % 2) + 1 == 1 else "2"
    label_path = os.path.join(label_directory, f"{prefix}_class_label_{suffix}.npy")
    np.save(label_path, np.asarray(labels_list[i]))

print("Criando os arquivos .npy para cada agrupamento de dados...")
for topic, font in {"time": data_arrays_time_domain, "frequency": data_arrays_frequency_domain}.items():
    # 0 - magacc
    np.save(os.path.join(data_array_directory, f"magacc_{topic}_domain_data_array.npy"), np.asarray(font[0]))

    # 123 - acc_x_y_z_axes
    acc_xyz = np.concatenate((np.asarray(font[1]), np.asarray(font[2]), np.asarray(font[3])), axis=2)
    np.save(os.path.join(data_array_directory, f"acc_x_y_z_axes_{topic}_domain_data_array.npy"), acc_xyz)

    # 4 - maggyr
    np.save(os.path.join(data_array_directory, f"maggyr_{topic}_domain_data_array.npy"), np.asarray(font[4]))

    # 567 - gyr_x_y_z_axes
    gyr_xyz = np.concatenate((np.asarray(font[5]), np.asarray(font[6]), np.asarray(font[7])), axis=2)
    np.save(os.path.join(data_array_directory, f"gyr_x_y_z_axes_{topic}_domain_data_array.npy"), gyr_xyz)

    # 04 - magacc_and_maggyr
    mag_both = np.concatenate((np.asarray(font[0]), np.asarray(font[4])), axis=2)
    np.save(os.path.join(data_array_directory, f"magacc_and_maggyr_{topic}_domain_data_array.npy"), mag_both)

    # 123567 - acc_and_gyr_three_axes
    accgyr = np.concatenate(
        [np.asarray(font[i]) for i in [1, 2, 3, 5, 6, 7]],
        axis=2
    )
    np.save(os.path.join(data_array_directory, f"acc_and_gyr_three_axes_{topic}_domain_data_array.npy"), accgyr)

print(f"\n Finalizado. Dados disponíveis em: {os.path.join(current_directory, 'labels_and_data')}")