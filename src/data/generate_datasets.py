# Criado por Leandro Soares - https://github.com/SoaresLMB
# Adaptado por Rodrigo Parracho - https://github.com/RodrigoKasama
# Adaptado por Caio Passos - https://github.com/stepsbtw

import argparse
import os

import numpy as np
import pandas as pd

from data_training_builders import (
    sort_by_number,
    get_file_path,
    create_dataframe,
    create_directory_if_does_not_exist,
    validate_sampling_segments,
)
from data_training_generators import generate_activities

""" EXECUTION OF THE TRAINING DATA GENERATION PROGRAM """

parser = argparse.ArgumentParser(
    description="Script para geração de datasets e rótulos para cada estratégia de cenários"
)
parser.add_argument(
    "position",
    type=str,
    nargs="?",
    default=None,
    choices=["chest", "left", "right"],
    help="Sensor position (omit to generate all three)",
)
parser.add_argument(
    "--database",
    type=str,
    default=None,
    help="Path to the database directory (default: ./database relative to this script)",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to the output directory (default: same as this script)",
)
parser.add_argument(
    "--chest-array-size",
    type=int,
    default=460,
    choices=[460, 1100],
    help="Window size for CHEST only (default: 460). LEFT/RIGHT remain 460.",
)
parser.add_argument(
    "--dataset-name",
    type=str,
    default=None,
    help="Override output folder name (default: same as position, e.g. chest or chest_1100).",
)

parser.add_argument(
    "--strict-sampling-validation",
    action="store_true",
    help="Fail if sampling.csv boundaries disagree with sensor rows for any activity.",
)

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
    window_ids_list = []
    validation_reports = []

    if args.dataset_name and len(positions) == 1:
        dataset_dir_name = args.dataset_name
    elif position == "chest" and args.chest_array_size == 1100:
        dataset_dir_name = "chest_1100"
    else:
        dataset_dir_name = position

    label_directory = os.path.join(output_directory, dataset_dir_name, "labels")
    data_array_directory = os.path.join(output_directory, dataset_dir_name, "data")

    print(f"\n[{position.upper()}] Criando diretórios de labels e data_arrays...")
    create_directory_if_does_not_exist(label_directory)
    create_directory_if_does_not_exist(data_array_directory)

    print(f"[{position.upper()}] Obtendo os dados de cada usuário...")
    for subdirectory in subdirectory_list:
        group_id = sort_by_number(subdirectory)
        print(f"  {subdirectory}...", end="")
        acc, gyr, sampling = get_file_path(main_directory, subdirectory, position_upper)

        acc_dataframe, gyr_dataframe, sampling_dataframe = create_dataframe(acc, gyr, sampling)

        validation_report = validate_sampling_segments(
            acc_dataframe,
            gyr_dataframe,
            sampling_dataframe,
        )
        validation_report.insert(0, "group_id", group_id)
        validation_report.insert(1, "position", position_upper)
        validation_reports.append(validation_report)

        issue_count = int(validation_report["has_issue"].sum())
        if issue_count:
            print(f"VALIDATION_WARN({issue_count})", end=" ")
            if args.strict_sampling_validation:
                bad_ids = validation_report.loc[validation_report["has_issue"], "sampling_id"].tolist()
                raise ValueError(
                    f"[{position_upper}] sampling validation failed for {subdirectory}: "
                    f"{issue_count} problematic ids {bad_ids}"
                )

        generate_activities(
            acc_dataframe,
            gyr_dataframe,
            sampling_dataframe,
            position_upper,
            data_arrays_time_domain,
            data_arrays_frequency_domain,
            labels_list,
            groups_list,
            window_ids_list,
            group_id,
            chest_array_size=args.chest_array_size,
        )
        print("OK")

    print(f"[{position.upper()}] Salvando rótulos de cada caso...")
    np.save(os.path.join(label_directory, "labels.npy"), np.asarray(labels_list))
    np.save(os.path.join(label_directory, "groups.npy"), np.asarray(groups_list))
    np.save(os.path.join(label_directory, "window_ids.npy"), np.asarray(window_ids_list, dtype=object))

    print(f"[{position.upper()}] Criando o arquivo de dados...")
    all_time = np.concatenate([np.asarray(c) for c in data_arrays_time_domain], axis=2)
    all_freq = np.concatenate([np.asarray(c) for c in data_arrays_frequency_domain], axis=2)

    if len(labels_list) != len(groups_list) or len(labels_list) != len(window_ids_list):
        raise ValueError(
            f"[{position.upper()}] labels/groups/window_ids have different lengths: "
            f"{len(labels_list)} / {len(groups_list)} / {len(window_ids_list)}"
        )

    if all_time.shape[0] != len(window_ids_list) or all_freq.shape[0] != len(window_ids_list):
        raise ValueError(
            f"[{position.upper()}] data arrays and window_ids have different lengths: "
            f"time={all_time.shape[0]} freq={all_freq.shape[0]} ids={len(window_ids_list)}"
        )

    np.save(os.path.join(data_array_directory, "data_time_domain.npy"), all_time)
    np.save(os.path.join(data_array_directory, "data_frequency_domain.npy"), all_freq)
    print(f"[{position.upper()}] Finalizado. Dados disponíveis em {data_array_directory}")

    if validation_reports:
        validation_path = os.path.join(label_directory, "sampling_validation_report.csv")
        df_validation = pd.concat(validation_reports, ignore_index=True)
        df_validation.to_csv(validation_path, index=False)
        total_issues = int(df_validation["has_issue"].sum())
        print(
            f"[{position.upper()}] Sampling validation report saved to {validation_path} "
            f"({total_issues} rows with issues)."
        )

    # ── Save metadata CSV ─────────────────────────────────────────────
    metadata_path = os.path.join(data_array_directory, "metadata.csv")

    df_meta = pd.DataFrame({
        "index": list(range(len(window_ids_list))),
        "position": position,
        "window_id": window_ids_list,
        "group_id": groups_list,
        "y_true": labels_list,
    })

    if not (len(window_ids_list) == len(groups_list) == len(labels_list)):
        raise ValueError("Metadata lists are misaligned!")

    if all_time.shape[0] != len(window_ids_list):
        raise ValueError("Data and metadata length mismatch!")

    df_meta.to_csv(metadata_path, index=False)

    print(f"[{position.upper()}] Metadata saved to {metadata_path}")
