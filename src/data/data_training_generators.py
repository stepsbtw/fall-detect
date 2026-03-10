# Escrito por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Caio Passos - https://github.com/stepsbtw
from data_training_builders import (section_data_array, create_labels, add_labels, add_data_arrays_to_time_and_frequency_data_lists)

import math

"""
Transforms the activities "FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5", "ADL_7", "ADL_8" and "ADL_13" 
into five-second arrays and adds them to the time domain and data domain lists. frequency.   Falling 
activities originally last 10 seconds, however the fall itself occurs within the first five seconds. 
Activities ADL_7" and "ADL_8" last 6 seconds, so the last second is being discarded. It was found that 
"ADL_15" also occurs in the first 5 seconds, so we discard observations after this time.
"""


def generate_array_of_activities_lasting_5seconds(data_array, array_size, data_array_list, fourier_transformed_data_array_list):
    add_data_arrays_to_time_and_frequency_data_lists(
        0, array_size, array_size, data_array, data_array_list, fourier_transformed_data_array_list)


"""
It transforms activities in which state transitions occur (for example: transition from walking to the lying 
shooting position into an array of size equivalent to 5 seconds. The function maps the largest force peak that 
occurs during the activity and moves the start and the end of the data array as a function of this peak.
"""


def generate_array_of_transition_activities(data_array, array_size, data_array_list, fourier_transformed_data_array_list):
    maximum_value = max(data_array)
    index_of_maximum_value = int(
        data_array.loc[data_array == maximum_value].index[0])
    initial_index = int(index_of_maximum_value - (array_size / 2))
    final_index = int(index_of_maximum_value + (array_size / 2))

    if initial_index < 0:
        initial_index = 0
        final_index = array_size
    elif final_index > int(data_array.index[-1]):
        final_index = int(data_array.index[-1])
        initial_index = final_index - array_size

    add_data_arrays_to_time_and_frequency_data_lists(
        initial_index, final_index, array_size, data_array, data_array_list, fourier_transformed_data_array_list)


"""
Transforms other activities lasting more than 10 seconds into several data arrays with a size of 5 seconds.
For example, the ADL_3 activity that lasts 30 seconds turns into 6 arrays of 5 seconds (480 observations or
1050 observations).
"""


def generate_array_of_other_activities(data_array_acc, data_array_gyr, array_size, acc_data_array_list, gyr_data_array_list,
                                       acc_fourier_transformed_data_array_list, gyr_fourier_transformed_data_array_list, label,
                                       labels_list, groups_list, group_id, timestamp_acc, timestamp_gyr, generate_labels=None):

    size_acc_data_array = len(data_array_acc)
    size_gyr_data_array = len(data_array_gyr)

    if size_acc_data_array > size_gyr_data_array:
        usable_size = size_gyr_data_array
    else:
        usable_size = size_acc_data_array

    # ── Original observation-count windowing (replaced) ───────────────────────
    # parts = math.floor(usable_size / array_size)
    #
    # This tied the number of sub-windows to the number of observations the
    # sensor happened to capture, which fluctuates with the real (not nominal)
    # sampling rate.  When a sensor runs slightly faster or slower, this pipeline
    # would produce a different window count than generate_fused_dataset.py for
    # the exact same activity, making the two datasets inconsistent.
    #
    # ── Timestamp-based windowing (mirrors generate_fused_dataset.py) ─────────
    # Count how many complete 5-second intervals fit in the *recorded duration*.
    # Using wall-clock duration instead of observation count makes the window
    # count independent of the sampling rate, so single-sensor and fused datasets
    # always agree on how many sub-windows each long activity produces.
    WINDOW_MS = 5_000
    t_start = float(timestamp_acc.iloc[0]) if len(timestamp_acc) > 0 else 0.0
    t_end = min(
        float(timestamp_acc.iloc[-1]) if len(timestamp_acc) > 0 else 0.0,
        float(timestamp_gyr.iloc[-1]) if len(timestamp_gyr) > 0 else 0.0,
    )
    parts = int((t_end - t_start) / WINDOW_MS)
    # Safety cap: never index past the actual data arrays (guards against edge
    # cases where timestamps and observation counts diverge, e.g. after a gap).
    parts = min(parts, math.floor(usable_size / array_size))
    initial_index = 0
    final_index = array_size

    for i in range(parts):
        add_data_arrays_to_time_and_frequency_data_lists(
            initial_index, final_index, array_size, data_array_acc, acc_data_array_list, acc_fourier_transformed_data_array_list)
        add_data_arrays_to_time_and_frequency_data_lists(
            initial_index, final_index, array_size, data_array_gyr, gyr_data_array_list, gyr_fourier_transformed_data_array_list)
        if generate_labels == "yes":
            add_labels(label, labels_list)
            groups_list.append(group_id)

        initial_index += array_size
        final_index += array_size


"""
Populates the lists with data arrays and labels for each activity for all 
collected data files. Used inside a "for" loop in the "generate_activities" function.
"""


def create_data_sets_for_training(position, activity, magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr,
                                  list_of_data_arrays_in_the_time_domain, list_of_data_arrays_in_the_frequency_domain, labels_list, groups_list, group_id,
                                  timestamp_acc, timestamp_gyr):

    label = create_labels(activity)

    activity = activity.split("_with_rifle")[0]

    five_second_activity_list = ["FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5", "ADL_5", "ADL_6", "ADL_7", "ADL_8", "ADL_13"]
    transition_activities_list = ["OM_3", "OM_4", "OM_5", "OM_6", "OM_7", "OM_8"]

    array_size = 1100 if position == "CHEST" else 460

    if len(xgyr) >= array_size and len(xacc) >= array_size:
        if activity in five_second_activity_list:

            generate_array_of_activities_lasting_5seconds(
                magacc, array_size, list_of_data_arrays_in_the_time_domain[0], list_of_data_arrays_in_the_frequency_domain[0])
            generate_array_of_activities_lasting_5seconds(
                xacc, array_size, list_of_data_arrays_in_the_time_domain[1], list_of_data_arrays_in_the_frequency_domain[1])
            generate_array_of_activities_lasting_5seconds(
                yacc, array_size, list_of_data_arrays_in_the_time_domain[2], list_of_data_arrays_in_the_frequency_domain[2])
            generate_array_of_activities_lasting_5seconds(
                zacc, array_size, list_of_data_arrays_in_the_time_domain[3], list_of_data_arrays_in_the_frequency_domain[3])
            generate_array_of_activities_lasting_5seconds(
                maggyr, array_size, list_of_data_arrays_in_the_time_domain[4], list_of_data_arrays_in_the_frequency_domain[4])
            generate_array_of_activities_lasting_5seconds(
                xgyr, array_size, list_of_data_arrays_in_the_time_domain[5], list_of_data_arrays_in_the_frequency_domain[5])
            generate_array_of_activities_lasting_5seconds(
                ygyr, array_size, list_of_data_arrays_in_the_time_domain[6], list_of_data_arrays_in_the_frequency_domain[6])
            generate_array_of_activities_lasting_5seconds(
                zgyr, array_size, list_of_data_arrays_in_the_time_domain[7], list_of_data_arrays_in_the_frequency_domain[7])

            add_labels(label, labels_list)
            groups_list.append(group_id)

        elif activity in transition_activities_list:
            generate_array_of_transition_activities(
                magacc, array_size, list_of_data_arrays_in_the_time_domain[0], list_of_data_arrays_in_the_frequency_domain[0])
            generate_array_of_transition_activities(
                xacc, array_size, list_of_data_arrays_in_the_time_domain[1], list_of_data_arrays_in_the_frequency_domain[1])
            generate_array_of_transition_activities(
                yacc, array_size, list_of_data_arrays_in_the_time_domain[2], list_of_data_arrays_in_the_frequency_domain[2])
            generate_array_of_transition_activities(
                zacc, array_size, list_of_data_arrays_in_the_time_domain[3], list_of_data_arrays_in_the_frequency_domain[3])
            generate_array_of_transition_activities(
                maggyr, array_size, list_of_data_arrays_in_the_time_domain[4], list_of_data_arrays_in_the_frequency_domain[4])
            generate_array_of_transition_activities(
                xgyr, array_size, list_of_data_arrays_in_the_time_domain[5], list_of_data_arrays_in_the_frequency_domain[5])
            generate_array_of_transition_activities(
                ygyr, array_size, list_of_data_arrays_in_the_time_domain[6], list_of_data_arrays_in_the_frequency_domain[6])
            generate_array_of_transition_activities(
                zgyr, array_size, list_of_data_arrays_in_the_time_domain[7], list_of_data_arrays_in_the_frequency_domain[7])

            add_labels(label, labels_list)
            groups_list.append(group_id)
        else:
            generate_array_of_other_activities(magacc, maggyr, array_size, list_of_data_arrays_in_the_time_domain[0],
                                               list_of_data_arrays_in_the_time_domain[4],
                                               list_of_data_arrays_in_the_frequency_domain[0],
                                               list_of_data_arrays_in_the_frequency_domain[4],
                                               label, labels_list, groups_list, group_id, timestamp_acc, timestamp_gyr, "yes")

            generate_array_of_other_activities(xacc, xgyr, array_size, list_of_data_arrays_in_the_time_domain[1],
                                               list_of_data_arrays_in_the_time_domain[5],
                                               list_of_data_arrays_in_the_frequency_domain[1],
                                               list_of_data_arrays_in_the_frequency_domain[5],
                                               label, labels_list, groups_list, group_id, timestamp_acc, timestamp_gyr)

            generate_array_of_other_activities(yacc, ygyr, array_size, list_of_data_arrays_in_the_time_domain[2],
                                               list_of_data_arrays_in_the_time_domain[6],
                                               list_of_data_arrays_in_the_frequency_domain[2],
                                               list_of_data_arrays_in_the_frequency_domain[6],
                                               label, labels_list, groups_list, group_id, timestamp_acc, timestamp_gyr)

            generate_array_of_other_activities(zacc, zgyr, array_size, list_of_data_arrays_in_the_time_domain[3],
                                               list_of_data_arrays_in_the_time_domain[7],
                                               list_of_data_arrays_in_the_frequency_domain[3],
                                               list_of_data_arrays_in_the_frequency_domain[7],
                                               label, labels_list, groups_list, group_id, timestamp_acc, timestamp_gyr)


"""
Populates the lists with data arrays and labels for each activity 
"""


def generate_activities(acc_dataframe, gyr_dataframe, sampling_dataframe, position, list_of_data_arrays_in_the_time_domain,
                        list_of_data_arrays_in_the_frequency_domain, labels_list, groups_list, group_id):

    for i in (sampling_dataframe["id"]):

        activity = (
            sampling_dataframe.loc[sampling_dataframe["id"] == i, "exercise"].iloc[0])

        if (sampling_dataframe.loc[sampling_dataframe["id"] == i, "withRifle"].iloc[0]) == 1 and activity[:2] != "OM":
            activity = f"{activity}_with_rifle"

        magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr = section_data_array(
            acc_dataframe, gyr_dataframe, i)

        timestamp_acc = acc_dataframe.loc[acc_dataframe["sampling"] == i, "timestamp"].reset_index(drop=True)
        timestamp_gyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == i, "timestamp"].reset_index(drop=True)

        create_data_sets_for_training(position, activity, magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr,
                                      list_of_data_arrays_in_the_time_domain, list_of_data_arrays_in_the_frequency_domain,
                                      labels_list, groups_list, group_id, timestamp_acc, timestamp_gyr)
