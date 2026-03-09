# Criado por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Rodrigo Parracho - https://github.com/RodrigoKasama
# Adaptado por Caio Passos - https://github.com/stepsbtw

import argparse
import os
from data_training_builders import (sort_by_number, get_file_path, create_dataframe, create_directory_if_does_not_exist)
from data_training_generators import generate_activities
import numpy as np

""" EXECUTION OF THE TRAINING DATA GENERATION PROGRAM """

parser = argparse.ArgumentParser(description="Script para geração de datasets e rótulos para cara estratégia de cenários")
parser.add_argument("position", type=str, nargs="?", default=None, choices=["chest", "left", "right"], help="Sensor position (omit to generate all three)")
parser.add_argument("--database", type=str, default=None, help="Path to the database directory (default: ./database relative to this script)")
parser.add_argument("--output", type=str, default=None, help="Path to the output directory (default: same as this script)")
args = parser.parse_args()

positions = [args.position] if args.position else ["chest", "left", "right"]

current_directory = os.path.dirname(__file__)
main_directory = args.database if args.database else os.path.join(current_directory, "database")
output_directory = args.output if args.output else current_directory

subdirectory_list = os.listdir(main_directory)
subdirectory_list.sort(key=sort_by_number)

for position in positions:
    position_upper = position.upper()
    data_arrays_time_domain = [[] for _ in range(8)]
    data_arrays_frequency_domain = [[] for _ in range(8)]
    labels_list = []
    groups_list = []

    label_directory = os.path.join(output_directory, position, "labels")
    data_array_directory = os.path.join(output_directory, position, "data")

    print(f"\n[{position.upper()}] Criando diretórios de labels e data_arrays...")
    create_directory_if_does_not_exist(label_directory)
    create_directory_if_does_not_exist(data_array_directory)

    print(f"[{position.upper()}] Obtendo os dados de cada usuário...")
    for subdirectory in subdirectory_list:
        group_id = sort_by_number(subdirectory)
        print(f"  {subdirectory}...", end="")
        acc, gyr, sampling = get_file_path(main_directory, subdirectory, position_upper)

        acc_dataframe, gyr_dataframe, sampling_dataframe = create_dataframe(
            acc, gyr, sampling)

        generate_activities(acc_dataframe, gyr_dataframe, sampling_dataframe, position_upper,
                            data_arrays_time_domain, data_arrays_frequency_domain, labels_list, groups_list, group_id)
        print(f"OK")

    print(f"[{position.upper()}] Salvando rótulos de cada caso...")
    np.save(os.path.join(label_directory, "labels.npy"), np.asarray(labels_list))
    np.save(os.path.join(label_directory, "groups.npy"), np.asarray(groups_list))

    print(f"[{position.upper()}] Criando o arquivo de dados...")
    # Channel order: magacc, acc_x, acc_y, acc_z, maggyr, gyr_x, gyr_y, gyr_z
    all_time = np.concatenate([np.asarray(c) for c in data_arrays_time_domain], axis=2)
    all_freq = np.concatenate([np.asarray(c) for c in data_arrays_frequency_domain], axis=2)

    np.save(os.path.join(data_array_directory, "data_time_domain.npy"), all_time)
    np.save(os.path.join(data_array_directory, "data_frequency_domain.npy"), all_freq)
    print(f"[{position.upper()}] Finalizado. Dados disponíveis em {data_array_directory}")

"""
 	# magacc - 0
		# time_domain
		# frequency_domain
  
	# acc_x_y_z_axes - 123
		# time_domain
		# frequency_domain
  
	# maggyr - 4
		# time_domain
		# frequency_domain
  
	# gyr_x_y_z_axes - 567
		# time_domain
		# frequency_domain
  
	# magacc_and_maggyr - 04
		# time_domain
		# frequency_domain
  
	# acc_and_gyr_three_axes - 123567
		# time_domain
		# frequency_domain
"""