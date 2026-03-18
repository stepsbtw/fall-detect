# Escrito por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Caio Passos - https://github.com/stepsbtw
from data_training_builders import (
    section_data_array,
    create_labels,
    add_labels,
    add_data_arrays_to_time_and_frequency_data_lists,
)

import math


"""
Transforms the activities "FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5", "ADL_7", "ADL_8" and "ADL_13"
into five-second arrays and adds them to the time domain and frequency domain lists.
"""


def generate_array_of_activities_lasting_5seconds(
    data_array,
    array_size,
    data_array_list,
    fourier_transformed_data_array_list,
):
    add_data_arrays_to_time_and_frequency_data_lists(
        0,
        array_size,
        array_size,
        data_array,
        data_array_list,
        fourier_transformed_data_array_list,
    )


"""
It transforms activities in which state transitions occur (for example: transition from walking to the lying
shooting position into an array of size equivalent to 5 seconds. The function maps the largest force peak that
occurs during the activity and moves the start and the end of the data array as a function of this peak.
"""


def generate_array_of_transition_activities(
    data_array,
    array_size,
    data_array_list,
    fourier_transformed_data_array_list,
):
    maximum_value = max(data_array)
    index_of_maximum_value = int(data_array.loc[data_array == maximum_value].index[0])
    initial_index = int(index_of_maximum_value - (array_size / 2))
    final_index = int(index_of_maximum_value + (array_size / 2))

    if initial_index < 0:
        initial_index = 0
        final_index = array_size
    elif final_index > int(data_array.index[-1]):
        final_index = int(data_array.index[-1])
        initial_index = final_index - array_size

    add_data_arrays_to_time_and_frequency_data_lists(
        initial_index,
        final_index,
        array_size,
        data_array,
        data_array_list,
        fourier_transformed_data_array_list,
    )


def build_occurrence_rank_map(sampling_dataframe):
    """
    Build a mapping:
        sampling_id -> {exercise, with_rifle, occurrence_rank}

    occurrence_rank is the repetition order of the same (exercise, withRifle)
    inside one subject/session, ordered by ascending sampling id.
    """
    rank_map = {}

    grouped = sampling_dataframe.groupby(["exercise", "withRifle"])
    for (exercise, with_rifle), group in grouped:
        group_sorted = group.sort_values("id")
        for occurrence_rank, (_, row) in enumerate(group_sorted.iterrows()):
            rank_map[int(row["id"])] = {
                "exercise": str(exercise),
                "with_rifle": int(with_rifle),
                "occurrence_rank": int(occurrence_rank),
            }

    return rank_map


def make_window_id(group_id, exercise, with_rifle, occurrence_rank, window_idx):
    return f"{int(group_id)}|{exercise}|{int(with_rifle)}|{int(occurrence_rank)}|{int(window_idx)}"


"""
Transforms other activities lasting more than 10 seconds into several data arrays with a size of 5 seconds.
For example, the ADL_3 activity that lasts 30 seconds turns into 6 arrays of 5 seconds.
"""


def generate_array_of_other_activities(
    data_array_acc,
    data_array_gyr,
    array_size,
    acc_data_array_list,
    gyr_data_array_list,
    acc_fourier_transformed_data_array_list,
    gyr_fourier_transformed_data_array_list,
    label,
    labels_list,
    groups_list,
    window_ids_list,
    group_id,
    timestamp_acc,
    timestamp_gyr,
    exercise,
    with_rifle,
    occurrence_rank,
    generate_labels=None,
):
    size_acc_data_array = len(data_array_acc)
    size_gyr_data_array = len(data_array_gyr)

    if size_acc_data_array > size_gyr_data_array:
        usable_size = size_gyr_data_array
    else:
        usable_size = size_acc_data_array

    WINDOW_MS = 5_000
    t_start = float(timestamp_acc.iloc[0]) if len(timestamp_acc) > 0 else 0.0
    t_end = min(
        float(timestamp_acc.iloc[-1]) if len(timestamp_acc) > 0 else 0.0,
        float(timestamp_gyr.iloc[-1]) if len(timestamp_gyr) > 0 else 0.0,
    )
    parts = int((t_end - t_start) / WINDOW_MS)

    # Safety cap
    parts = min(parts, math.floor(usable_size / array_size))

    initial_index = 0
    final_index = array_size

    for window_idx in range(parts):
        add_data_arrays_to_time_and_frequency_data_lists(
            initial_index,
            final_index,
            array_size,
            data_array_acc,
            acc_data_array_list,
            acc_fourier_transformed_data_array_list,
        )
        add_data_arrays_to_time_and_frequency_data_lists(
            initial_index,
            final_index,
            array_size,
            data_array_gyr,
            gyr_data_array_list,
            gyr_fourier_transformed_data_array_list,
        )

        if generate_labels == "yes":
            add_labels(label, labels_list)
            groups_list.append(group_id)
            window_ids_list.append(
                make_window_id(
                    group_id=group_id,
                    exercise=exercise,
                    with_rifle=with_rifle,
                    occurrence_rank=occurrence_rank,
                    window_idx=window_idx,
                )
            )

        initial_index += array_size
        final_index += array_size


"""
Populates the lists with data arrays and labels for each activity for all
collected data files. Used inside a "for" loop in the "generate_activities" function.
"""


def create_data_sets_for_training(
    position,
    activity,
    magacc,
    xacc,
    yacc,
    zacc,
    maggyr,
    xgyr,
    ygyr,
    zgyr,
    list_of_data_arrays_in_the_time_domain,
    list_of_data_arrays_in_the_frequency_domain,
    labels_list,
    groups_list,
    window_ids_list,
    group_id,
    timestamp_acc,
    timestamp_gyr,
    exercise,
    with_rifle,
    occurrence_rank,
    chest_array_size=460,
):
    label = create_labels(activity)

    activity = activity.split("_with_rifle")[0]

    five_second_activity_list = [
        "FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5",
        "ADL_5", "ADL_6", "ADL_7", "ADL_8", "ADL_13",
    ]
    transition_activities_list = ["OM_3", "OM_4", "OM_5", "OM_6", "OM_7", "OM_8"]

    #array_size = 1100 if position == "CHEST" else 460
    array_size = chest_array_size if position == "CHEST" else 460

    if len(xgyr) >= array_size and len(xacc) >= array_size:
        if activity in five_second_activity_list:
            generate_array_of_activities_lasting_5seconds(
                magacc, array_size,
                list_of_data_arrays_in_the_time_domain[0],
                list_of_data_arrays_in_the_frequency_domain[0],
            )
            generate_array_of_activities_lasting_5seconds(
                xacc, array_size,
                list_of_data_arrays_in_the_time_domain[1],
                list_of_data_arrays_in_the_frequency_domain[1],
            )
            generate_array_of_activities_lasting_5seconds(
                yacc, array_size,
                list_of_data_arrays_in_the_time_domain[2],
                list_of_data_arrays_in_the_frequency_domain[2],
            )
            generate_array_of_activities_lasting_5seconds(
                zacc, array_size,
                list_of_data_arrays_in_the_time_domain[3],
                list_of_data_arrays_in_the_frequency_domain[3],
            )
            generate_array_of_activities_lasting_5seconds(
                maggyr, array_size,
                list_of_data_arrays_in_the_time_domain[4],
                list_of_data_arrays_in_the_frequency_domain[4],
            )
            generate_array_of_activities_lasting_5seconds(
                xgyr, array_size,
                list_of_data_arrays_in_the_time_domain[5],
                list_of_data_arrays_in_the_frequency_domain[5],
            )
            generate_array_of_activities_lasting_5seconds(
                ygyr, array_size,
                list_of_data_arrays_in_the_time_domain[6],
                list_of_data_arrays_in_the_frequency_domain[6],
            )
            generate_array_of_activities_lasting_5seconds(
                zgyr, array_size,
                list_of_data_arrays_in_the_time_domain[7],
                list_of_data_arrays_in_the_frequency_domain[7],
            )

            add_labels(label, labels_list)
            groups_list.append(group_id)
            window_ids_list.append(
                make_window_id(
                    group_id=group_id,
                    exercise=exercise,
                    with_rifle=with_rifle,
                    occurrence_rank=occurrence_rank,
                    window_idx=0,
                )
            )

        elif activity in transition_activities_list:
            generate_array_of_transition_activities(
                magacc, array_size,
                list_of_data_arrays_in_the_time_domain[0],
                list_of_data_arrays_in_the_frequency_domain[0],
            )
            generate_array_of_transition_activities(
                xacc, array_size,
                list_of_data_arrays_in_the_time_domain[1],
                list_of_data_arrays_in_the_frequency_domain[1],
            )
            generate_array_of_transition_activities(
                yacc, array_size,
                list_of_data_arrays_in_the_time_domain[2],
                list_of_data_arrays_in_the_frequency_domain[2],
            )
            generate_array_of_transition_activities(
                zacc, array_size,
                list_of_data_arrays_in_the_time_domain[3],
                list_of_data_arrays_in_the_frequency_domain[3],
            )
            generate_array_of_transition_activities(
                maggyr, array_size,
                list_of_data_arrays_in_the_time_domain[4],
                list_of_data_arrays_in_the_frequency_domain[4],
            )
            generate_array_of_transition_activities(
                xgyr, array_size,
                list_of_data_arrays_in_the_time_domain[5],
                list_of_data_arrays_in_the_frequency_domain[5],
            )
            generate_array_of_transition_activities(
                ygyr, array_size,
                list_of_data_arrays_in_the_time_domain[6],
                list_of_data_arrays_in_the_frequency_domain[6],
            )
            generate_array_of_transition_activities(
                zgyr, array_size,
                list_of_data_arrays_in_the_time_domain[7],
                list_of_data_arrays_in_the_frequency_domain[7],
            )

            add_labels(label, labels_list)
            groups_list.append(group_id)
            window_ids_list.append(
                make_window_id(
                    group_id=group_id,
                    exercise=exercise,
                    with_rifle=with_rifle,
                    occurrence_rank=occurrence_rank,
                    window_idx=0,
                )
            )

        else:
            generate_array_of_other_activities(
                magacc, maggyr, array_size,
                list_of_data_arrays_in_the_time_domain[0],
                list_of_data_arrays_in_the_time_domain[4],
                list_of_data_arrays_in_the_frequency_domain[0],
                list_of_data_arrays_in_the_frequency_domain[4],
                label,
                labels_list,
                groups_list,
                window_ids_list,
                group_id,
                timestamp_acc,
                timestamp_gyr,
                exercise,
                with_rifle,
                occurrence_rank,
                "yes",
            )

            generate_array_of_other_activities(
                xacc, xgyr, array_size,
                list_of_data_arrays_in_the_time_domain[1],
                list_of_data_arrays_in_the_time_domain[5],
                list_of_data_arrays_in_the_frequency_domain[1],
                list_of_data_arrays_in_the_frequency_domain[5],
                label,
                labels_list,
                groups_list,
                window_ids_list,
                group_id,
                timestamp_acc,
                timestamp_gyr,
                exercise,
                with_rifle,
                occurrence_rank,
            )

            generate_array_of_other_activities(
                yacc, ygyr, array_size,
                list_of_data_arrays_in_the_time_domain[2],
                list_of_data_arrays_in_the_time_domain[6],
                list_of_data_arrays_in_the_frequency_domain[2],
                list_of_data_arrays_in_the_frequency_domain[6],
                label,
                labels_list,
                groups_list,
                window_ids_list,
                group_id,
                timestamp_acc,
                timestamp_gyr,
                exercise,
                with_rifle,
                occurrence_rank,
            )

            generate_array_of_other_activities(
                zacc, zgyr, array_size,
                list_of_data_arrays_in_the_time_domain[3],
                list_of_data_arrays_in_the_time_domain[7],
                list_of_data_arrays_in_the_frequency_domain[3],
                list_of_data_arrays_in_the_frequency_domain[7],
                label,
                labels_list,
                groups_list,
                window_ids_list,
                group_id,
                timestamp_acc,
                timestamp_gyr,
                exercise,
                with_rifle,
                occurrence_rank,
            )


"""
Populates the lists with data arrays and labels for each activity
"""


def generate_activities(
    acc_dataframe,
    gyr_dataframe,
    sampling_dataframe,
    position,
    list_of_data_arrays_in_the_time_domain,
    list_of_data_arrays_in_the_frequency_domain,
    labels_list,
    groups_list,
    window_ids_list,
    group_id,
    chest_array_size=460,
):
    occurrence_map = build_occurrence_rank_map(sampling_dataframe)

    for i in sampling_dataframe["id"]:
        base_exercise = sampling_dataframe.loc[sampling_dataframe["id"] == i, "exercise"].iloc[0]
        with_rifle = int(sampling_dataframe.loc[sampling_dataframe["id"] == i, "withRifle"].iloc[0])

        activity = base_exercise
        if with_rifle == 1 and activity[:2] != "OM":
            activity = f"{activity}_with_rifle"

        occurrence_rank = occurrence_map[int(i)]["occurrence_rank"]

        magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr = section_data_array(
            acc_dataframe, gyr_dataframe, i
        )

        timestamp_acc = acc_dataframe.loc[acc_dataframe["sampling"] == i, "timestamp"].reset_index(drop=True)
        timestamp_gyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == i, "timestamp"].reset_index(drop=True)

        create_data_sets_for_training(
            position,
            activity,
            magacc,
            xacc,
            yacc,
            zacc,
            maggyr,
            xgyr,
            ygyr,
            zgyr,
            list_of_data_arrays_in_the_time_domain,
            list_of_data_arrays_in_the_frequency_domain,
            labels_list,
            groups_list,
            window_ids_list,
            group_id,
            timestamp_acc,
            timestamp_gyr,
            base_exercise,
            with_rifle,
            occurrence_rank,
            chest_array_size,
        )