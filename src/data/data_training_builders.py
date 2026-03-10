# Escrito por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Caio Passos - https://github.com/stepsbtw
import pandas as pd
import os
import math
import numpy as np

def create_directory_if_does_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_by_number(id):
    return int(id[2:])

def get_file_path(main_directory, subdirectory, position, preprocessing=False):
    subdirectory_path = os.path.join(main_directory, subdirectory)
    subdirectory_path_of_subdirectory = os.path.join(
        subdirectory_path, position)

    file_name = f"{subdirectory}_{position}_acceleration.csv"
    file_name_2 = f"{subdirectory}_{position}_sampling.csv"
    file_name_3 = f"{subdirectory}_{position}_angular_speed.csv"

    acc_file = os.path.join(subdirectory_path_of_subdirectory, file_name)
    sampling_file = os.path.join(
        subdirectory_path_of_subdirectory, file_name_2)
    gyr_file = os.path.join(subdirectory_path_of_subdirectory, file_name_3)

    if preprocessing:
        return acc_file, gyr_file, sampling_file, file_name, file_name_2, file_name_3
    return acc_file, gyr_file, sampling_file

def add_magnitude_column(dataframe, sensor=None):
    initial_letter = None

    if sensor == "acc":
        initial_letter = "a"
    else:
        initial_letter = "w"

    resultant_force = []
    i = 0

    while i < len(dataframe[f"{initial_letter}x"]):
        resultant = math.sqrt((dataframe[f"{initial_letter}x"][i]) ** 2 + (
            dataframe[f"{initial_letter}y"][i]) ** 2 + (dataframe[f"{initial_letter}z"][i]) ** 2)
        resultant_force.append(resultant)
        i += 1
    dataframe.insert(5, "Magnitude", resultant_force, True)

def create_dataframe(acc_file, gyr_file, sampling_file):

    acc_dataframe = pd.DataFrame(pd.read_csv(acc_file))
    gyr_dataframe = pd.DataFrame(pd.read_csv(gyr_file))
    sampling_dataframe = pd.DataFrame(pd.read_csv(sampling_file))

    return acc_dataframe, gyr_dataframe, sampling_dataframe

def fourier_transform(time_series):
    altered_time_series = []
    mean_time_series = np.mean(time_series)

    for i in time_series:
        # Subtraction from the average to Remove the DC Component (Zero Frequency Component).
        data = i - mean_time_series
        altered_time_series.append(data)
    altered_time_series = np.array(altered_time_series)

    return np.abs(np.fft.fft(altered_time_series))

def section_data_array(acc_dataframe, gyr_dataframe, i, use_in_media_generator=None):
    magacc = acc_dataframe.loc[acc_dataframe["sampling"] == i, "Magnitude"]
    magacc = magacc.reset_index(drop=True)

    xacc = acc_dataframe.loc[acc_dataframe["sampling"] == i, "ax"]
    xacc = xacc.reset_index(drop=True)

    yacc = acc_dataframe.loc[acc_dataframe["sampling"] == i, "ay"]
    yacc = yacc.reset_index(drop=True)

    zacc = acc_dataframe.loc[acc_dataframe["sampling"] == i, "az"]
    zacc = zacc.reset_index(drop=True)

    maggyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == i, "Magnitude"]
    maggyr = maggyr.reset_index(drop=True)

    xgyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == i, "wx"]
    xgyr = xgyr.reset_index(drop=True)

    ygyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == i, "wy"]
    ygyr = ygyr.reset_index(drop=True)

    zgyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == i, "wz"]
    zgyr = zgyr.reset_index(drop=True)

    timestamp_acc = acc_dataframe.loc[acc_dataframe["sampling"]
                                      == i, "timestamp"]
    timestamp_acc = timestamp_acc.reset_index(drop=True)
    timestamp_acc = timestamp_acc.drop(0, errors='ignore')
    timestamp_acc = timestamp_acc.reset_index(drop=True)

    timestamp_gyr = gyr_dataframe.loc[gyr_dataframe["sampling"]
                                      == i, "timestamp"]
    timestamp_gyr = timestamp_gyr.reset_index(drop=True)
    timestamp_gyr = timestamp_gyr.drop(0, errors='ignore')
    timestamp_gyr = timestamp_gyr.reset_index(drop=True)

    if use_in_media_generator == "yes":
        return timestamp_acc, timestamp_gyr, magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr
    return magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr

def add_data_arrays_to_time_and_frequency_data_lists(initial_index, final_index, array_size, data_array, data_array_list, fourier_transformed_data_array_list):
    data_array = data_array[initial_index:final_index]
    numpy_data_array = np.array(data_array)
    numpy_data_array = np.expand_dims(numpy_data_array, axis=1)
    data_array_list.append(numpy_data_array)

    transformed_data_array = fourier_transform(data_array)
    transformed_data_array = transformed_data_array[:int(array_size / 2)]
    numpy_transformed_data_array = np.array(transformed_data_array)
    numpy_transformed_data_array = np.expand_dims(
        numpy_transformed_data_array, axis=1)
    fourier_transformed_data_array_list.append(numpy_transformed_data_array)

def create_labels(activity):
    labels = {"ADL_1": 0, "ADL_2": 0, "ADL_3": 0, "ADL_4": 0, "ADL_5": 0, "ADL_6": 0, "ADL_7": 0, "ADL_8": 0,
                               "ADL_9": 0, "ADL_10": 0, "ADL_11": 0, "ADL_12": 0, "ADL_13": 0, 
                               "OM_1": 0, "OM_2": 0, "OM_3": 0, "OM_4": 0, "OM_5": 0, "OM_6": 0, "OM_7": 0, "OM_8": 0, "OM_9": 0, # OM_3 ao OM_8 sao possiveis falsos positivos!
                               "FALL_1": 1, "FALL_2": 1, "FALL_3": 1, "FALL_4": 1, "FALL_5": 1}

    activity_without_rifle = activity.split("_with_rifle")[0]
    label = labels.get(activity_without_rifle)

    return label

def add_labels(label, labels_list):
    labels_list.append(label)
