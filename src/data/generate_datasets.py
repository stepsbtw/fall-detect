import argparse
import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from data_training_builders import (
    sort_by_number,
    get_file_path,
    create_dataframe,
    create_directory_if_does_not_exist,
    validate_sampling_segments,
)
from data_training_generators import (
    generate_activities,
    build_position_activity_records,
    build_canonical_window_manifest,
    append_window_sample,
    create_metadata_row,
)

WINDOW_ID_FILE = "window_ids.npy"
LABELS_FILE = "labels.npy"
GROUPS_FILE = "groups.npy"
TIME_FILE = "data_time_domain.npy"
FREQ_FILE = "data_frequency_domain.npy"
DEFAULT_WINDOW_MODE = "legacy"
SUPPORTED_WINDOW_MODES = ["legacy", "strict_overlap", "matched_sessions"]


def _load_position_metadata(dataset_dir: str) -> pd.DataFrame:
    metadata_path = os.path.join(dataset_dir, "data", "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata at {metadata_path}")
    return pd.read_csv(metadata_path)


def _normalize_metadata_for_save(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    df.insert(0, "index", np.arange(len(df), dtype=int))
    return df


def _write_trimmed_metadata(dataset_dir: str, filtered: pd.DataFrame) -> None:
    filtered = _normalize_metadata_for_save(filtered)
    data_meta = os.path.join(dataset_dir, "data", "metadata.csv")
    labels_meta = os.path.join(dataset_dir, "labels", "metadata.csv")
    filtered.to_csv(data_meta, index=False)
    if os.path.exists(labels_meta):
        filtered.to_csv(labels_meta, index=False)


def _trim_saved_dataset_to_window_ids(dataset_dir: str, ordered_window_ids: List[str]) -> int:
    metadata = _load_position_metadata(dataset_dir)
    order_map = {str(wid): i for i, wid in enumerate(ordered_window_ids)}

    filtered = metadata[metadata["window_id"].astype(str).isin([str(w) for w in ordered_window_ids])].copy()
    filtered["__order__"] = filtered["window_id"].astype(str).map(order_map)
    filtered = filtered.sort_values("__order__").drop(columns=["__order__"])

    labels_dir = os.path.join(dataset_dir, "labels")
    data_dir = os.path.join(dataset_dir, "data")

    window_ids = np.load(os.path.join(labels_dir, WINDOW_ID_FILE), allow_pickle=True)
    id_to_idx = {str(wid): i for i, wid in enumerate(window_ids.tolist())}
    indices = [id_to_idx[str(wid)] for wid in ordered_window_ids if str(wid) in id_to_idx]

    time = np.load(os.path.join(data_dir, TIME_FILE), allow_pickle=True)[indices]
    freq = np.load(os.path.join(data_dir, FREQ_FILE), allow_pickle=True)[indices]
    labels = np.load(os.path.join(labels_dir, LABELS_FILE), allow_pickle=True)[indices]
    groups = np.load(os.path.join(labels_dir, GROUPS_FILE), allow_pickle=True)[indices]
    new_window_ids = np.asarray([window_ids[i] for i in indices], dtype=object)

    np.save(os.path.join(data_dir, TIME_FILE), time)
    np.save(os.path.join(data_dir, FREQ_FILE), freq)
    np.save(os.path.join(labels_dir, LABELS_FILE), labels)
    np.save(os.path.join(labels_dir, GROUPS_FILE), groups)
    np.save(os.path.join(labels_dir, WINDOW_ID_FILE), new_window_ids)

    _write_trimmed_metadata(dataset_dir, filtered)
    return len(filtered)


def _write_strict_trim_audit(
    output_directory: str,
    dataset_dirs_by_position: Dict[str, str],
    ordered_common_ids: List[str],
) -> str:
    common_set = set(map(str, ordered_common_ids))
    rows = []
    positions = list(dataset_dirs_by_position.keys())
    per_pos_ids: Dict[str, Set[str]] = {}
    all_ids: Set[str] = set()

    for pos, dataset_dir in dataset_dirs_by_position.items():
        meta = _load_position_metadata(dataset_dir)
        ids = set(meta["window_id"].astype(str).tolist())
        per_pos_ids[pos] = ids
        all_ids |= ids

    for wid in sorted(all_ids):
        present_in = [pos for pos in positions if wid in per_pos_ids[pos]]
        missing_in = [pos for pos in positions if wid not in per_pos_ids[pos]]
        rows.append(
            {
                "window_id": wid,
                "kept_in_strict_intersection": wid in common_set,
                "present_in": ",".join(present_in),
                "missing_in": ",".join(missing_in),
                "missing_count": len(missing_in),
                "drop_reason": "kept" if wid in common_set else f"missing_in_{'_'.join(missing_in)}",
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_path = os.path.join(output_directory, "strict_trim_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    return audit_path


def _enforce_common_window_intersection(output_directory: str, dataset_dirs_by_position: Dict[str, str]) -> None:
    metas = {pos: _load_position_metadata(path) for pos, path in dataset_dirs_by_position.items()}
    common_ids = set.intersection(*[set(df["window_id"].astype(str).tolist()) for df in metas.values()])
    ordered_common_ids = sorted(common_ids)

    print(
        f"\n[STRICT] Trimming all generated datasets to the common window_id intersection: {len(ordered_common_ids)} rows"
    )

    audit_path = _write_strict_trim_audit(output_directory, dataset_dirs_by_position, ordered_common_ids)
    print(f"[STRICT] Audit saved to {audit_path}")

    for pos, dataset_dir in dataset_dirs_by_position.items():
        before = len(metas[pos])
        after = _trim_saved_dataset_to_window_ids(dataset_dir, ordered_common_ids)
        print(f"[STRICT] {pos.upper()}: kept {after}, removed {before - after}")


def _build_records_by_group(main_directory: str, subdirectory_list: List[str], positions: List[str], strict_sampling_validation: bool):
    records_by_group: Dict[int, Dict[str, Dict]] = {}
    validation_reports_by_position: Dict[str, List[pd.DataFrame]] = {p: [] for p in positions}

    print("Loading source files and validating sampling segments...")
    for subdirectory in subdirectory_list:
        group_id = sort_by_number(subdirectory)
        print(f"  {subdirectory}...")
        records_by_group[group_id] = {}
        for position in positions:
            position_upper = position.upper()
            acc, gyr, sampling = get_file_path(main_directory, subdirectory, position_upper)
            acc_dataframe, gyr_dataframe, sampling_dataframe = create_dataframe(acc, gyr, sampling)

            validation_report = validate_sampling_segments(
                acc_dataframe,
                gyr_dataframe,
                sampling_dataframe,
            )
            validation_report.insert(0, "group_id", group_id)
            validation_report.insert(1, "position", position_upper)
            validation_reports_by_position[position].append(validation_report)

            issue_count = int(validation_report["has_issue"].sum())
            if issue_count and strict_sampling_validation:
                bad_ids = validation_report.loc[validation_report["has_issue"], "sampling_id"].tolist()
                raise ValueError(
                    f"[{position_upper}] sampling validation failed for {subdirectory}: "
                    f"{issue_count} problematic ids {bad_ids}"
                )

            records_by_group[group_id][position_upper] = build_position_activity_records(
                acc_dataframe,
                gyr_dataframe,
                sampling_dataframe,
                position_upper,
                group_id,
            )

    return records_by_group, validation_reports_by_position


def _save_validation_report(position: str, label_directory: str, validation_reports: List[pd.DataFrame]) -> None:
    if validation_reports:
        validation_path = os.path.join(label_directory, "sampling_validation_report.csv")
        df_validation = pd.concat(validation_reports, ignore_index=True)
        df_validation.to_csv(validation_path, index=False)
        total_issues = int(df_validation["has_issue"].sum())
        print(
            f"[{position.upper()}] Sampling validation report saved to {validation_path} "
            f"({total_issues} rows with issues)."
        )


def _manifest_filename(window_mode: str) -> str:
    return f"canonical_window_manifest_{window_mode}.csv"


def _legacy_metadata_frame(position: str, window_ids_list: List[str], groups_list: List[int], labels_list: List[int]) -> pd.DataFrame:
    df_meta = pd.DataFrame(
        {
            "position": position,
            "window_id": window_ids_list,
            "group_id": groups_list,
            "y_true": labels_list,
        }
    )
    return _normalize_metadata_for_save(df_meta)


def _save_position_dataset(
    position: str,
    dataset_dir: str,
    data_arrays_time_domain: List[List[np.ndarray]],
    data_arrays_frequency_domain: List[List[np.ndarray]],
    labels_list: List[int],
    groups_list: List[int],
    window_ids_list: List[str],
    metadata_rows: List[Dict[str, object]] | None,
    validation_reports: List[pd.DataFrame],
) -> None:
    label_directory = os.path.join(dataset_dir, "labels")
    data_array_directory = os.path.join(dataset_dir, "data")
    create_directory_if_does_not_exist(label_directory)
    create_directory_if_does_not_exist(data_array_directory)

    np.save(os.path.join(label_directory, LABELS_FILE), np.asarray(labels_list))
    np.save(os.path.join(label_directory, GROUPS_FILE), np.asarray(groups_list))
    np.save(os.path.join(label_directory, WINDOW_ID_FILE), np.asarray(window_ids_list, dtype=object))

    all_time = np.concatenate([np.asarray(c) for c in data_arrays_time_domain], axis=2)
    all_freq = np.concatenate([np.asarray(c) for c in data_arrays_frequency_domain], axis=2)

    np.save(os.path.join(data_array_directory, TIME_FILE), all_time)
    np.save(os.path.join(data_array_directory, FREQ_FILE), all_freq)

    _save_validation_report(position, label_directory, validation_reports)

    if metadata_rows is None:
        df_meta = _legacy_metadata_frame(position, window_ids_list, groups_list, labels_list)
    else:
        df_meta = _normalize_metadata_for_save(pd.DataFrame(metadata_rows))

    data_meta_path = os.path.join(data_array_directory, "metadata.csv")
    labels_meta_path = os.path.join(label_directory, "metadata.csv")
    df_meta.to_csv(data_meta_path, index=False)
    df_meta.to_csv(labels_meta_path, index=False)

    print(
        f"[{position.upper()}] Done. Samples={len(window_ids_list)} "
        f"time={all_time.shape} freq={all_freq.shape}"
    )
    print(f"[{position.upper()}] Metadata saved to {data_meta_path}")


def run_legacy_mode(args, positions, main_directory, output_directory, subdirectory_list):
    dataset_dirs_by_position: Dict[str, str] = {}

    for position in positions:
        position_upper = position.upper()
        data_arrays_time_domain = [[] for _ in range(8)]
        data_arrays_frequency_domain = [[] for _ in range(8)]
        labels_list: List[int] = []
        groups_list: List[int] = []
        window_ids_list: List[str] = []
        validation_reports: List[pd.DataFrame] = []

        dataset_dir_name = args.dataset_name if args.dataset_name and len(positions) == 1 else position
        dataset_dir = os.path.join(output_directory, dataset_dir_name)
        dataset_dirs_by_position[position] = dataset_dir
        label_directory = os.path.join(dataset_dir, "labels")
        data_array_directory = os.path.join(dataset_dir, "data")

        print(f"\n[{position_upper}] Criando diretórios de labels e data_arrays...")
        create_directory_if_does_not_exist(label_directory)
        create_directory_if_does_not_exist(data_array_directory)

        print(f"[{position_upper}] Obtendo os dados de cada usuário...")
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
            )
            print("OK")

        if len(labels_list) != len(groups_list) or len(labels_list) != len(window_ids_list):
            raise ValueError(
                f"[{position_upper}] labels/groups/window_ids have different lengths: "
                f"{len(labels_list)} / {len(groups_list)} / {len(window_ids_list)}"
            )

        all_time = np.concatenate([np.asarray(c) for c in data_arrays_time_domain], axis=2)
        all_freq = np.concatenate([np.asarray(c) for c in data_arrays_frequency_domain], axis=2)
        if all_time.shape[0] != len(window_ids_list) or all_freq.shape[0] != len(window_ids_list):
            raise ValueError(
                f"[{position_upper}] data arrays and window_ids have different lengths: "
                f"time={all_time.shape[0]} freq={all_freq.shape[0]} ids={len(window_ids_list)}"
            )

        _save_position_dataset(
            position=position,
            dataset_dir=dataset_dir,
            data_arrays_time_domain=data_arrays_time_domain,
            data_arrays_frequency_domain=data_arrays_frequency_domain,
            labels_list=labels_list,
            groups_list=groups_list,
            window_ids_list=window_ids_list,
            metadata_rows=None,
            validation_reports=validation_reports,
        )

    return dataset_dirs_by_position


def run_manifest_mode(args, positions, main_directory, output_directory, subdirectory_list):
    positions_upper = [p.upper() for p in positions]
    dataset_dirs_by_position: Dict[str, str] = {}

    records_by_group, validation_reports_by_position = _build_records_by_group(
        main_directory=main_directory,
        subdirectory_list=subdirectory_list,
        positions=positions,
        strict_sampling_validation=args.strict_sampling_validation,
    )

    if args.manifest_path and os.path.exists(args.manifest_path):
        manifest = pd.read_csv(args.manifest_path)
        print(f"Using manifest: {args.manifest_path} ({len(manifest)} windows)")
    else:
        manifest = build_canonical_window_manifest(
            records_by_group=records_by_group,
            required_positions=positions_upper,
            alignment_mode=args.window_mode,
        )
        manifest_path = args.manifest_path or os.path.join(output_directory, _manifest_filename(args.window_mode))
        manifest.to_csv(manifest_path, index=False)
        print(f"Saved canonical window manifest: {manifest_path} ({len(manifest)} windows)")

    for position in positions:
        position_upper = position.upper()
        dataset_dir_name = args.dataset_name if args.dataset_name and len(positions) == 1 else position
        dataset_dir = os.path.join(output_directory, dataset_dir_name)
        dataset_dirs_by_position[position] = dataset_dir
        create_directory_if_does_not_exist(os.path.join(dataset_dir, "labels"))
        create_directory_if_does_not_exist(os.path.join(dataset_dir, "data"))

        data_arrays_time_domain = [[] for _ in range(8)]
        data_arrays_frequency_domain = [[] for _ in range(8)]
        labels_list: List[int] = []
        groups_list: List[int] = []
        window_ids_list: List[str] = []
        metadata_rows: List[Dict[str, object]] = []
        skipped = 0

        print(f"\n[{position_upper}] Generating arrays from canonical windows...")
        for _, window_spec_row in manifest.iterrows():
            window_spec = window_spec_row.to_dict()
            group_id = int(window_spec["group_id"])
            match_key = (
                str(window_spec["exercise"]),
                int(window_spec["with_rifle"]),
                int(window_spec["occurrence_rank"]),
            )
            position_records = records_by_group.get(group_id, {}).get(position_upper, {})
            position_record = position_records.get(match_key)
            if position_record is None:
                skipped += 1
                continue

            try:
                extracted_bounds = append_window_sample(
                    position_record,
                    window_spec,
                    data_arrays_time_domain,
                    data_arrays_frequency_domain,
                )
            except Exception:
                skipped += 1
                continue

            meta_row = create_metadata_row(position, window_spec, extracted_bounds)
            metadata_rows.append(meta_row)
            labels_list.append(int(meta_row["y_true"]))
            groups_list.append(int(meta_row["group_id"]))
            window_ids_list.append(str(meta_row["window_id"]))

        if skipped:
            print(f"[{position_upper}] Skipped {skipped} manifest rows that could not be extracted.")

        _save_position_dataset(
            position=position,
            dataset_dir=dataset_dir,
            data_arrays_time_domain=data_arrays_time_domain,
            data_arrays_frequency_domain=data_arrays_frequency_domain,
            labels_list=labels_list,
            groups_list=groups_list,
            window_ids_list=window_ids_list,
            metadata_rows=metadata_rows,
            validation_reports=validation_reports_by_position[position],
        )

    return dataset_dirs_by_position


def main():
    parser = argparse.ArgumentParser(
        description="Script para geração de datasets e rótulos para cada estratégia de cenários"
    )
    parser.add_argument(
        "position",
        type=str,
        nargs="?",
        default=None,
        choices=["chest", "left", "right"],
        help="Sensor position (omit to generate all requested positions)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the dataset directory (default: ./dataset relative to this script)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output directory (default: same as this script)",
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
    parser.add_argument(
        "--strict-common-intersection",
        action="store_true",
        help="After generating all requested positions, trim them to the common window_id intersection and save an audit CSV.",
    )
    parser.add_argument(
        "--window-mode",
        type=str,
        default=DEFAULT_WINDOW_MODE,
        choices=SUPPORTED_WINDOW_MODES,
        help="Window generation mode: legacy, strict_overlap, or matched_sessions.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional manifest CSV path. For manifest modes, reuse if present or save to this path if missing.",
    )

    args = parser.parse_args()

    positions = [args.position] if args.position else ["chest", "left", "right"]
    current_directory = os.path.dirname(__file__)
    main_directory = args.input if args.input else os.path.join(current_directory, "dataset")
    output_directory = args.output if args.output else current_directory

    subdirectory_list = os.listdir(main_directory)
    subdirectory_list.sort(key=sort_by_number)

    if args.window_mode == "legacy":
        dataset_dirs_by_position = run_legacy_mode(args, positions, main_directory, output_directory, subdirectory_list)
    else:
        print(f"Window mode: {args.window_mode}")
        dataset_dirs_by_position = run_manifest_mode(args, positions, main_directory, output_directory, subdirectory_list)

    if args.strict_common_intersection and len(positions) > 1:
        _enforce_common_window_intersection(output_directory, dataset_dirs_by_position)


if __name__ == "__main__":
    main()
