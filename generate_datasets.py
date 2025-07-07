import argparse
import os
import numpy as np
from scipy.signal import resample
from builders.data_training_builders import (
    sort_by_number, get_file_path, create_dataframe, create_directory_if_does_not_exist
)
from builders.data_training_generators import generate_activities

# --- CLI Argument ---
parser = argparse.ArgumentParser(description="Geração de datasets e rótulos para diferentes cenários")
parser.add_argument("position", type=str, choices=["chest", "left", "right", "all"], help="Sensor position ou 'all' para fusão")
args = parser.parse_args()
position = args.position.lower()

# --- Paths ---
current_directory = os.path.dirname(__file__)
main_directory = os.path.join(current_directory, "database")
label_directory = os.path.join(current_directory, "labels_and_data", "labels", position)
data_array_directory = os.path.join(current_directory, "labels_and_data", "data", position if position != "all" else "all")

create_directory_if_does_not_exist(label_directory)
create_directory_if_does_not_exist(data_array_directory)

# ========== Shared Utility ==========
def resample_to_target(X, target_len):
    return np.array([resample(sample, target_len, axis=0) for sample in X])

# ========== Fusion Mode ==========
def generate_all_fused_scenarios():
    print("\n[✓] Gerando todos os cenários combinados...")

    def load_fused(scenario_filename):
        def load_and_resample(pos):
            path = os.path.join(current_directory, "labels_and_data", "data", pos, scenario_filename)
            data = np.load(path)
            return data if pos == "chest" else resample_to_target(data, target_len=data_chest.shape[1])

        data_chest = np.load(os.path.join(current_directory, "labels_and_data", "data", "chest", scenario_filename))
        data_left = resample_to_target(np.load(os.path.join(current_directory, "labels_and_data", "data", "left", scenario_filename)), target_len=data_chest.shape[1])
        data_right = resample_to_target(np.load(os.path.join(current_directory, "labels_and_data", "data", "right", scenario_filename)), target_len=data_chest.shape[1])

        min_samples = min(data_chest.shape[0], data_left.shape[0], data_right.shape[0])
        fused = np.concatenate([
            data_chest[:min_samples],
            data_left[:min_samples],
            data_right[:min_samples]
        ], axis=2)

        return fused, min_samples

    scenarios = {
        "Sc1_acc_T": "magacc_time_domain_data_array.npy",
        "Sc1_gyr_T": "maggyr_time_domain_data_array.npy",
        "Sc1_acc_F": "magacc_frequency_domain_data_array.npy",
        "Sc1_gyr_F": "maggyr_frequency_domain_data_array.npy",
        "Sc_2_acc_T": "acc_x_y_z_axes_time_domain_data_array.npy",
        "Sc_2_gyr_T": "gyr_x_y_z_axes_time_domain_data_array.npy",
        "Sc_2_acc_F": "acc_x_y_z_axes_frequency_domain_data_array.npy",
        "Sc_2_gyr_F": "gyr_x_y_z_axes_frequency_domain_data_array.npy",
        "Sc_3_T": "magacc_and_maggyr_time_domain_data_array.npy",
        "Sc_3_F": "magacc_and_maggyr_frequency_domain_data_array.npy",
        "Sc_4_T": "acc_and_gyr_three_axes_time_domain_data_array.npy",
        "Sc_4_F": "acc_and_gyr_three_axes_frequency_domain_data_array.npy",
    }

    output_dir = os.path.join(current_directory, "labels_and_data", "data", "all")
    create_directory_if_does_not_exist(output_dir)

    min_samples_all = []

    for scenario_name, file_name in scenarios.items():
        fused_data, min_samples = load_fused(file_name)
        np.save(os.path.join(output_dir, file_name), fused_data)
        print(f"  ✔ {scenario_name} salvo — shape: {fused_data.shape}")
        min_samples_all.append(min_samples)

    # Save labels aligned to min sample count
    label_input = os.path.join(current_directory, "labels_and_data", "labels", "chest", "binary_class_label_1.npy")
    labels = np.load(label_input)[:min(min_samples_all)]

    label_output_dir = os.path.join(current_directory, "labels_and_data", "labels", "all")
    create_directory_if_does_not_exist(label_output_dir)
    np.save(os.path.join(label_output_dir, "binary_class_label_1.npy"), labels)

    print("\n[✓] Todos os cenários combinados foram gerados com sucesso.")

# Run fusion mode
if position == "all":
    generate_all_fused_scenarios()
    exit()

# ========== Single Sensor Mode ==========
print(f"Obtendo dados da posição: {position.upper()}")
subdirectory_list = os.listdir(main_directory)
subdirectory_list.sort(key=sort_by_number)

data_arrays_time_domain = [[] for _ in range(8)]
data_arrays_frequency_domain = [[] for _ in range(8)]
labels_list = [[] for _ in range(4)]

print("Criando diretórios de labels e data_arrays...")
create_directory_if_does_not_exist(label_directory)
create_directory_if_does_not_exist(data_array_directory)

print("Obtendo os dados de cada usuário...")
for subdirectory in subdirectory_list:
    print(f"  {subdirectory}...", end="")
    acc, gyr, sampling = get_file_path(main_directory, subdirectory, position.upper())
    acc_df, gyr_df, sampling_df = create_dataframe(acc, gyr, sampling)
    generate_activities(acc_df, gyr_df, sampling_df, position.upper(),
                        data_arrays_time_domain, data_arrays_frequency_domain, labels_list)
    print("OK")

print("Salvando rótulos de cada caso...")
for i in range(4):
    prefix = "multiple" if i < 2 else "binary"
    suffix = "1" if (i % 2) + 1 == 1 else "2"
    label_path = os.path.join(label_directory, f"{prefix}_class_label_{suffix}.npy")
    np.save(label_path, np.asarray(labels_list[i]))

print("Criando os arquivos .npy para cada agrupamento de dados...")
for topic, font in {"time": data_arrays_time_domain, "frequency": data_arrays_frequency_domain}.items():
    np.save(os.path.join(data_array_directory, f"magacc_{topic}_domain_data_array.npy"), np.asarray(font[0]))

    acc_xyz = np.concatenate((np.asarray(font[1]), np.asarray(font[2]), np.asarray(font[3])), axis=2)
    np.save(os.path.join(data_array_directory, f"acc_x_y_z_axes_{topic}_domain_data_array.npy"), acc_xyz)

    np.save(os.path.join(data_array_directory, f"maggyr_{topic}_domain_data_array.npy"), np.asarray(font[4]))

    gyr_xyz = np.concatenate((np.asarray(font[5]), np.asarray(font[6]), np.asarray(font[7])), axis=2)
    np.save(os.path.join(data_array_directory, f"gyr_x_y_z_axes_{topic}_domain_data_array.npy"), gyr_xyz)

    mag_both = np.concatenate((np.asarray(font[0]), np.asarray(font[4])), axis=2)
    np.save(os.path.join(data_array_directory, f"magacc_and_maggyr_{topic}_domain_data_array.npy"), mag_both)

    accgyr = np.concatenate([np.asarray(font[i]) for i in [1, 2, 3, 5, 6, 7]], axis=2)
    np.save(os.path.join(data_array_directory, f"acc_and_gyr_three_axes_{topic}_domain_data_array.npy"), accgyr)

print(f"\n[✓] Finalizado. Dados disponíveis em: {os.path.join(current_directory, 'labels_and_data')}")
