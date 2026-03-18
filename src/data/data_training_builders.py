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



def validate_sampling_segments(
    acc_dataframe,
    gyr_dataframe,
    sampling_dataframe,
    timestamp_tolerance_ms=2500,
    duration_tolerance_ratio=0.25,
):
    """
    Validate whether each sampling id in the manifest is consistent with the
    sensor rows assigned to that id.

    Returns a dataframe with one row per sampling id and boolean issue flags.
    This is intentionally a validator, not a slicer: the training pipeline may
    still rely on the sensor CSVs' ``sampling`` column for extraction.
    """
    report_rows = []

    for _, row in sampling_dataframe.iterrows():
        sampling_id = int(row["id"])
        expected_start = float(row["beginning"]) if "beginning" in row and pd.notna(row["beginning"]) else np.nan
        expected_end = float(row["ending"]) if "ending" in row and pd.notna(row["ending"]) else np.nan
        expected_duration = (
            expected_end - expected_start
            if not np.isnan(expected_start) and not np.isnan(expected_end)
            else np.nan
        )

        acc_slice = acc_dataframe.loc[acc_dataframe["sampling"] == sampling_id, ["timestamp"]].copy()
        gyr_slice = gyr_dataframe.loc[gyr_dataframe["sampling"] == sampling_id, ["timestamp"]].copy()

        acc_count = int(len(acc_slice))
        gyr_count = int(len(gyr_slice))

        acc_start = float(acc_slice["timestamp"].min()) if acc_count else np.nan
        acc_end = float(acc_slice["timestamp"].max()) if acc_count else np.nan
        gyr_start = float(gyr_slice["timestamp"].min()) if gyr_count else np.nan
        gyr_end = float(gyr_slice["timestamp"].max()) if gyr_count else np.nan

        observed_start = np.nanmin([acc_start, gyr_start]) if (acc_count or gyr_count) else np.nan
        observed_end = np.nanmax([acc_end, gyr_end]) if (acc_count or gyr_count) else np.nan
        observed_duration = (
            observed_end - observed_start
            if not np.isnan(observed_start) and not np.isnan(observed_end)
            else np.nan
        )

        start_gap_ms = (
            abs(observed_start - expected_start)
            if not np.isnan(observed_start) and not np.isnan(expected_start)
            else np.nan
        )
        end_gap_ms = (
            abs(observed_end - expected_end)
            if not np.isnan(observed_end) and not np.isnan(expected_end)
            else np.nan
        )
        duration_gap_ms = (
            abs(observed_duration - expected_duration)
            if not np.isnan(observed_duration) and not np.isnan(expected_duration)
            else np.nan
        )
        duration_gap_ratio = (
            duration_gap_ms / expected_duration
            if not np.isnan(duration_gap_ms) and expected_duration not in (0, np.nan)
            else np.nan
        )

        missing_acc = acc_count == 0
        missing_gyr = gyr_count == 0
        missing_any_sensor = missing_acc or missing_gyr
        start_mismatch = bool(not np.isnan(start_gap_ms) and start_gap_ms > timestamp_tolerance_ms)
        end_mismatch = bool(not np.isnan(end_gap_ms) and end_gap_ms > timestamp_tolerance_ms)
        duration_mismatch = bool(
            not np.isnan(duration_gap_ratio) and duration_gap_ratio > duration_tolerance_ratio
        )
        has_issue = missing_any_sensor or start_mismatch or end_mismatch or duration_mismatch

        report_rows.append({
            "sampling_id": sampling_id,
            "exercise": row.get("exercise"),
            "withRifle": row.get("withRifle"),
            "expected_start": expected_start,
            "expected_end": expected_end,
            "expected_duration_ms": expected_duration,
            "acc_count": acc_count,
            "gyr_count": gyr_count,
            "observed_start": observed_start,
            "observed_end": observed_end,
            "observed_duration_ms": observed_duration,
            "start_gap_ms": start_gap_ms,
            "end_gap_ms": end_gap_ms,
            "duration_gap_ms": duration_gap_ms,
            "duration_gap_ratio": duration_gap_ratio,
            "missing_acc": missing_acc,
            "missing_gyr": missing_gyr,
            "missing_any_sensor": missing_any_sensor,
            "start_mismatch": start_mismatch,
            "end_mismatch": end_mismatch,
            "duration_mismatch": duration_mismatch,
            "has_issue": has_issue,
        })

    return pd.DataFrame(report_rows)
