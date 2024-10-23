# Criado por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Rodrigo Parracho - https://github.com/RodrigoKasama

import argparse
import os
from builders.data_training_builders import (
    sort_by_number, get_file_path, create_dataframe, create_directory_if_does_not_exist)
from builders.data_training_generators import generate_activities
import numpy as np

data_arrays_time_domain = [[] for _ in range(8)]
data_arrays_frequency_domain = [[] for _ in range(8)]
labels_list = [[] for _ in range(4)]

""" EXECUTION OF THE TRAINING DATA GENERATION PROGRAM """

parser = argparse.ArgumentParser(description="Script para geração de datasets e rótulos para cara estratégia de cenários")
parser.add_argument("position", type=str, choices=["chest", "left", "right"], help="Sensor position")
args = parser.parse_args()
position = args.position.upper()

current_directory = os.path.dirname(__file__)
main_directory = os.path.join(current_directory, "database")

subdirectory_list = os.listdir(main_directory)

subdirectory_list.sort(key=sort_by_number)

label_directory = os.path.join(current_directory, "labels_and_data", "labels", position.lower())
data_array_directory = os.path.join(current_directory, "labels_and_data", "data", position.lower())

print("Criando diretórios de labels e data_arrays...")
create_directory_if_does_not_exist(label_directory)
create_directory_if_does_not_exist(data_array_directory)

print("Obtendo os dados de cada usuário...")
for subdirectory in subdirectory_list:
    print(f"  {subdirectory}...", end="")
    acc, gyr, sampling = get_file_path(main_directory, subdirectory, position)

    acc_dataframe, gyr_dataframe, sampling_dataframe = create_dataframe(
        acc, gyr, sampling)

    generate_activities(acc_dataframe, gyr_dataframe, sampling_dataframe, position,
                        data_arrays_time_domain, data_arrays_frequency_domain, labels_list)
    print(f"OK")

print("Salvando rótulos de cada caso...")
for i in range(4):
    prefix = "multiple" if i < 2 else "binary"
    sufix = "1" if (i % 2) + 1 == 1 else "2"
    np.save(os.path.join(label_directory,
            f"{prefix}_class_label_{sufix}.npy"), np.asarray(labels_list[i]))

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


print("Criando os arquivos .npy referente a cada agrupamento de dados...")
for topic, font in {"time": data_arrays_time_domain, "frequency": data_arrays_frequency_domain}.items():

    # 0 - "magacc"
    np.save(os.path.join(data_array_directory,
            f"magacc_{topic}_domain_data_array.npy"), np.asarray(font[0]))
    # 123 - "acc_x_y_z_axes"
    np.save(os.path.join(data_array_directory, f"acc_x_y_z_axes_{topic}_domain_data_array.npy"),
            np.concatenate((np.asarray(font[1]), np.asarray(font[2]), np.asarray(font[3])), axis=2))
    # 4 - "maggyr"
    np.save(os.path.join(data_array_directory,
            f"maggyr_{topic}_domain_data_array.npy"), np.asarray(font[4]))

    # 567 - "gyr_x_y_z_axes"
    np.save(os.path.join(data_array_directory, f"gyr_x_y_z_axes_{topic}_domain_data_array.npy"),
            np.concatenate((np.asarray(font[5]), np.asarray(font[6]), np.asarray(font[7])), axis=2))

    # 04 - magacc_and_maggyr
    np.save(os.path.join(data_array_directory, f"magacc_and_maggyr_{topic}_domain_data_array.npy"),
            np.concatenate((np.asarray(font[0]), np.asarray(font[4])), axis=2))

    # 123567 - acc_and_gyr_three_axes
    np.save(os.path.join(data_array_directory, f"acc_and_gyr_three_axes_{topic}_domain_data_array.npy"),
            np.concatenate((
                np.asarray(font[1]), np.asarray(font[2]), np.asarray(font[3]),
                np.asarray(font[5]), np.asarray(font[6]), np.asarray(font[7])),
            axis=2))
print(f"Finalizado. Dados de treinamento disponíveis em {os.path.join(current_directory, 'labels_and_data')}")