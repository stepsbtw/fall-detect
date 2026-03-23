from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_training_builders import (
    add_data_arrays_to_time_and_frequency_data_lists,
    add_timestamp_window_to_time_and_frequency_data_lists,
    add_labels,
    create_labels,
    section_data_array,
)

WINDOW_MS = 5_000
OUTPUT_ARRAY_SIZE = 460

# legacy mode lists
LEGACY_FIVE_SECOND_ACTIVITIES = frozenset([
    "FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5",
    "ADL_7", "ADL_8", "ADL_13",
])
LEGACY_TRANSITION_ACTIVITIES = frozenset([
    "ADL_1", "ADL_2", "ADL_4", "ADL_5", "ADL_6",
])

# manifest modes lists
FIVE_SECOND_ACTIVITIES = frozenset([
    "FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5",
    "ADL_5", "ADL_6", "ADL_7", "ADL_8", "ADL_13",
])
TRANSITION_ACTIVITIES = frozenset([
    "OM_3", "OM_4", "OM_5", "OM_6", "OM_7", "OM_8",
])
REFERENCE_POSITION_ORDER = ["CHEST", "LEFT", "RIGHT"]
CHANNEL_SPECS = [
    ("acc", "Magnitude"),
    ("acc", "ax"),
    ("acc", "ay"),
    ("acc", "az"),
    ("gyr", "Magnitude"),
    ("gyr", "wx"),
    ("gyr", "wy"),
    ("gyr", "wz"),
]


# ---------------- legacy-compatible API ----------------
def generate_array_of_activities_lasting_5seconds(
    data_array: pd.Series,
    raw_array_size: int,
    output_array_size: int,
    data_array_list: List[np.ndarray],
    fourier_transformed_data_array_list: List[np.ndarray],
) -> None:
    add_data_arrays_to_time_and_frequency_data_lists(
        0,
        raw_array_size,
        raw_array_size,
        data_array,
        data_array_list,
        fourier_transformed_data_array_list,
        output_size=output_array_size,
    )


def generate_array_of_transition_activities(
    data_array: pd.Series,
    raw_array_size: int,
    output_array_size: int,
    data_array_list: List[np.ndarray],
    fourier_transformed_data_array_list: List[np.ndarray],
) -> None:
    maximum_value = max(data_array)
    index_of_maximum_value = int(data_array.loc[data_array == maximum_value].index[0])
    initial_index = int(index_of_maximum_value - (raw_array_size / 2))
    final_index = initial_index + raw_array_size

    data_len = len(data_array)
    if initial_index < 0:
        initial_index = 0
        final_index = raw_array_size
    elif final_index > data_len:
        final_index = data_len
        initial_index = final_index - raw_array_size

    add_data_arrays_to_time_and_frequency_data_lists(
        initial_index,
        final_index,
        raw_array_size,
        data_array,
        data_array_list,
        fourier_transformed_data_array_list,
        output_size=output_array_size,
    )


def build_occurrence_rank_map(sampling_dataframe: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    occurrence_map: Dict[int, Dict[str, int]] = {}
    grouped = sampling_dataframe.groupby(["exercise", "withRifle"])
    for (exercise, with_rifle), group in grouped:
        group_sorted = group.sort_values("id")
        for occurrence_rank, (_, row) in enumerate(group_sorted.iterrows()):
            occurrence_map[int(row["id"])] = {
                "exercise": str(exercise),
                "with_rifle": int(with_rifle),
                "occurrence_rank": int(occurrence_rank),
            }
    return occurrence_map


def make_window_id(group_id, exercise, with_rifle, occurrence_rank, window_idx):
    return f"{int(group_id)}|{exercise}|{int(with_rifle)}|{int(occurrence_rank)}|{int(window_idx)}"


def generate_array_of_other_activities(
    data_array_acc,
    data_array_gyr,
    raw_array_size,
    output_array_size,
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
    usable_size = min(size_acc_data_array, size_gyr_data_array)

    t_start = float(timestamp_acc.iloc[0]) if len(timestamp_acc) > 0 else 0.0
    t_end = min(
        float(timestamp_acc.iloc[-1]) if len(timestamp_acc) > 0 else 0.0,
        float(timestamp_gyr.iloc[-1]) if len(timestamp_gyr) > 0 else 0.0,
    )
    parts = int((t_end - t_start) / WINDOW_MS)
    parts = min(parts, math.floor(usable_size / raw_array_size))

    initial_index = 0
    final_index = raw_array_size

    for window_idx in range(parts):
        add_data_arrays_to_time_and_frequency_data_lists(
            initial_index,
            final_index,
            raw_array_size,
            data_array_acc,
            acc_data_array_list,
            acc_fourier_transformed_data_array_list,
            output_size=output_array_size,
        )
        add_data_arrays_to_time_and_frequency_data_lists(
            initial_index,
            final_index,
            raw_array_size,
            data_array_gyr,
            gyr_data_array_list,
            gyr_fourier_transformed_data_array_list,
            output_size=output_array_size,
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

        initial_index += raw_array_size
        final_index += raw_array_size


def create_data_sets_for_training(
    position: str,
    activity: str,
    magacc: pd.Series,
    xacc: pd.Series,
    yacc: pd.Series,
    zacc: pd.Series,
    maggyr: pd.Series,
    xgyr: pd.Series,
    ygyr: pd.Series,
    zgyr: pd.Series,
    list_of_data_arrays_in_the_time_domain: List[List[np.ndarray]],
    list_of_data_arrays_in_the_frequency_domain: List[List[np.ndarray]],
    labels_list: List[int],
    groups_list: List[int],
    window_ids_list: List[str],
    group_id: int,
    timestamp_acc: pd.Series,
    timestamp_gyr: pd.Series,
    exercise: str,
    with_rifle: int,
    occurrence_rank: int,
) -> None:
    label = create_labels(activity)

    raw_array_size = 1100 if position == "CHEST" else 460
    output_array_size = OUTPUT_ARRAY_SIZE

    if len(xgyr) < raw_array_size or len(xacc) < raw_array_size:
        return

    if activity in LEGACY_FIVE_SECOND_ACTIVITIES:
        for channel_idx, channel_data in enumerate([magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr]):
            generate_array_of_activities_lasting_5seconds(
                channel_data,
                raw_array_size,
                output_array_size,
                list_of_data_arrays_in_the_time_domain[channel_idx],
                list_of_data_arrays_in_the_frequency_domain[channel_idx],
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
        return

    if activity in LEGACY_TRANSITION_ACTIVITIES:
        for channel_idx, channel_data in enumerate([magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr]):
            generate_array_of_transition_activities(
                channel_data,
                raw_array_size,
                output_array_size,
                list_of_data_arrays_in_the_time_domain[channel_idx],
                list_of_data_arrays_in_the_frequency_domain[channel_idx],
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
        return

    paired_channels = [
        (magacc, maggyr, 0, 4, "yes"),
        (xacc, xgyr, 1, 5, None),
        (yacc, ygyr, 2, 6, None),
        (zacc, zgyr, 3, 7, None),
    ]
    for acc_data, gyr_data, acc_idx, gyr_idx, generate_labels_flag in paired_channels:
        generate_array_of_other_activities(
            acc_data,
            gyr_data,
            raw_array_size,
            output_array_size,
            list_of_data_arrays_in_the_time_domain[acc_idx],
            list_of_data_arrays_in_the_time_domain[gyr_idx],
            list_of_data_arrays_in_the_frequency_domain[acc_idx],
            list_of_data_arrays_in_the_frequency_domain[gyr_idx],
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
            generate_labels_flag,
        )


def generate_activities(
    acc_dataframe: pd.DataFrame,
    gyr_dataframe: pd.DataFrame,
    sampling_dataframe: pd.DataFrame,
    position: str,
    list_of_data_arrays_in_the_time_domain: List[List[np.ndarray]],
    list_of_data_arrays_in_the_frequency_domain: List[List[np.ndarray]],
    labels_list: List[int],
    groups_list: List[int],
    window_ids_list: List[str],
    group_id: int,
) -> None:
    occurrence_map = build_occurrence_rank_map(sampling_dataframe)

    for sampling_id in sampling_dataframe["id"]:
        base_exercise = sampling_dataframe.loc[sampling_dataframe["id"] == sampling_id, "exercise"].iloc[0]
        with_rifle = int(sampling_dataframe.loc[sampling_dataframe["id"] == sampling_id, "withRifle"].iloc[0])

        activity = base_exercise
        if with_rifle == 1 and activity[:2] != "OM":
            activity = f"{activity}_with_rifle"

        occurrence_rank = occurrence_map[int(sampling_id)]["occurrence_rank"]

        magacc, xacc, yacc, zacc, maggyr, xgyr, ygyr, zgyr = section_data_array(
            acc_dataframe,
            gyr_dataframe,
            sampling_id,
        )

        timestamp_acc = acc_dataframe.loc[acc_dataframe["sampling"] == sampling_id, "timestamp"].reset_index(drop=True)
        timestamp_gyr = gyr_dataframe.loc[gyr_dataframe["sampling"] == sampling_id, "timestamp"].reset_index(drop=True)

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
        )


# ---------------- manifest API ----------------
def make_timestamp_window_id(
    group_id: int,
    exercise: str,
    with_rifle: int,
    occurrence_rank: int,
    window_start_ms: float,
    window_end_ms: float,
) -> str:
    return (
        f"{int(group_id)}|{exercise}|{int(with_rifle)}|{int(occurrence_rank)}|"
        f"{int(round(window_start_ms))}|{int(round(window_end_ms))}"
    )


def make_matched_window_id(
    group_id: int,
    exercise: str,
    with_rifle: int,
    occurrence_rank: int,
    subwindow_idx: int,
) -> str:
    return f"{int(group_id)}|{exercise}|{int(with_rifle)}|{int(occurrence_rank)}|{int(subwindow_idx)}"


def _effective_timestamps(series: pd.Series) -> pd.Series:
    ts = series.reset_index(drop=True)
    if len(ts) > 1:
        ts = ts.drop(0, errors="ignore").reset_index(drop=True)
    return ts


def _signal_overlap_bounds(acc_slice: pd.DataFrame, gyr_slice: pd.DataFrame) -> Optional[Tuple[float, float]]:
    acc_ts = _effective_timestamps(acc_slice["timestamp"])
    gyr_ts = _effective_timestamps(gyr_slice["timestamp"])
    if len(acc_ts) == 0 or len(gyr_ts) == 0:
        return None
    start = max(float(acc_ts.iloc[0]), float(gyr_ts.iloc[0]))
    end = min(float(acc_ts.iloc[-1]), float(gyr_ts.iloc[-1]))
    if end <= start:
        return None
    return start, end


def _activity_type(base_exercise: str) -> str:
    if base_exercise in FIVE_SECOND_ACTIVITIES:
        return "five_second"
    if base_exercise in TRANSITION_ACTIVITIES:
        return "transition"
    return "long"


def build_position_activity_records(
    acc_dataframe: pd.DataFrame,
    gyr_dataframe: pd.DataFrame,
    sampling_dataframe: pd.DataFrame,
    position: str,
    group_id: int,
) -> Dict[Tuple[str, int, int], Dict[str, object]]:
    position_upper = position.upper()
    occurrence_map = build_occurrence_rank_map(sampling_dataframe)
    records: Dict[Tuple[str, int, int], Dict[str, object]] = {}

    for _, row in sampling_dataframe.iterrows():
        sampling_id = int(row["id"])
        base_exercise = str(row["exercise"])
        with_rifle = int(row["withRifle"])
        occurrence_rank = int(occurrence_map[sampling_id]["occurrence_rank"])
        match_key = (base_exercise, with_rifle, occurrence_rank)

        acc_slice = acc_dataframe.loc[acc_dataframe["sampling"] == sampling_id].reset_index(drop=True)
        gyr_slice = gyr_dataframe.loc[gyr_dataframe["sampling"] == sampling_id].reset_index(drop=True)
        overlap = _signal_overlap_bounds(acc_slice, gyr_slice)
        if overlap is None:
            continue

        records[match_key] = {
            "group_id": int(group_id),
            "position": position_upper,
            "sampling_id": sampling_id,
            "base_exercise": base_exercise,
            "with_rifle": with_rifle,
            "occurrence_rank": occurrence_rank,
            "activity_type": _activity_type(base_exercise),
            "acc": acc_slice,
            "gyr": gyr_slice,
            "overlap_start_ms": float(overlap[0]),
            "overlap_end_ms": float(overlap[1]),
        }
    return records


def _pick_reference_position(positions_present: List[str]) -> str:
    for ref in REFERENCE_POSITION_ORDER:
        if ref in positions_present:
            return ref
    return positions_present[0]


def _strict_bounds_for_record(record: Dict[str, object], subwindow_idx: int = 0) -> Optional[Tuple[float, float]]:
    start = float(record["overlap_start_ms"])
    end = float(record["overlap_end_ms"])
    activity_type = str(record["activity_type"])
    if activity_type == "five_second":
        if end - start < WINDOW_MS:
            return None
        return start, start + WINDOW_MS
    if activity_type == "transition":
        acc_df = record["acc"]
        peak_idx = int(acc_df["Magnitude"].reset_index(drop=True).idxmax())
        center_ts = float(acc_df["timestamp"].iloc[peak_idx])
        ws = center_ts - (WINDOW_MS / 2)
        we = center_ts + (WINDOW_MS / 2)
        if ws < start or we > end:
            return None
        return ws, we
    ws = start + (subwindow_idx * WINDOW_MS)
    we = ws + WINDOW_MS
    if we > end:
        return None
    return ws, we


def _matched_bounds_for_record(record: Dict[str, object], subwindow_idx: int = 0) -> Optional[Tuple[float, float]]:
    return _strict_bounds_for_record(record, subwindow_idx=subwindow_idx)


def _count_extractable_windows(record: Dict[str, object], alignment_mode: str) -> int:
    activity_type = str(record["activity_type"])
    if activity_type in {"five_second", "transition"}:
        bounds = _matched_bounds_for_record(record, 0) if alignment_mode == "matched_sessions" else _strict_bounds_for_record(record, 0)
        return 1 if bounds is not None else 0

    start = float(record["overlap_start_ms"])
    end = float(record["overlap_end_ms"])
    rough_n = int((end - start) / WINDOW_MS)
    if rough_n <= 0:
        return 0
    count = 0
    for idx in range(rough_n):
        bounds = _matched_bounds_for_record(record, idx) if alignment_mode == "matched_sessions" else _strict_bounds_for_record(record, idx)
        if bounds is None:
            break
        count += 1
    return count


def _build_windows_for_match_key_strict(
    group_id: int,
    match_key: Tuple[str, int, int],
    records_by_position: Dict[str, Dict[str, object]],
    required_positions: List[str],
) -> List[Dict[str, object]]:
    if any(position not in records_by_position for position in required_positions):
        return []

    base_exercise, with_rifle, occurrence_rank = match_key
    records = [records_by_position[position] for position in required_positions]
    common_start = max(float(record["overlap_start_ms"]) for record in records)
    common_end = min(float(record["overlap_end_ms"]) for record in records)
    if common_end <= common_start:
        return []

    activity_type = str(records[0]["activity_type"])
    windows: List[Dict[str, object]] = []

    if activity_type == "five_second":
        if (common_end - common_start) < WINDOW_MS:
            return []
        window_start = common_start
        window_end = window_start + WINDOW_MS
        windows.append({
            "group_id": int(group_id),
            "exercise": base_exercise,
            "with_rifle": int(with_rifle),
            "occurrence_rank": int(occurrence_rank),
            "subwindow_idx": 0,
            "window_start_ms": float(window_start),
            "window_end_ms": float(window_end),
            "window_id": make_timestamp_window_id(group_id, base_exercise, with_rifle, occurrence_rank, window_start, window_end),
            "alignment_mode": "strict_overlap",
        })
        return windows

    if activity_type == "transition":
        ref_position = _pick_reference_position(list(records_by_position.keys()))
        ref_record = records_by_position[ref_position]
        ref_acc = ref_record["acc"]
        peak_idx = int(ref_acc["Magnitude"].reset_index(drop=True).idxmax())
        center_ts = float(ref_acc["timestamp"].iloc[peak_idx])
        window_start = center_ts - (WINDOW_MS / 2)
        window_end = center_ts + (WINDOW_MS / 2)
        if window_start < common_start or window_end > common_end:
            return []
        windows.append({
            "group_id": int(group_id),
            "exercise": base_exercise,
            "with_rifle": int(with_rifle),
            "occurrence_rank": int(occurrence_rank),
            "subwindow_idx": 0,
            "window_start_ms": float(window_start),
            "window_end_ms": float(window_end),
            "window_id": make_timestamp_window_id(group_id, base_exercise, with_rifle, occurrence_rank, window_start, window_end),
            "alignment_mode": "strict_overlap",
        })
        return windows

    n_windows = int((common_end - common_start) / WINDOW_MS)
    for subwindow_idx in range(n_windows):
        window_start = common_start + (subwindow_idx * WINDOW_MS)
        window_end = window_start + WINDOW_MS
        windows.append({
            "group_id": int(group_id),
            "exercise": base_exercise,
            "with_rifle": int(with_rifle),
            "occurrence_rank": int(occurrence_rank),
            "subwindow_idx": int(subwindow_idx),
            "window_start_ms": float(window_start),
            "window_end_ms": float(window_end),
            "window_id": make_timestamp_window_id(group_id, base_exercise, with_rifle, occurrence_rank, window_start, window_end),
            "alignment_mode": "strict_overlap",
        })
    return windows


def _build_windows_for_match_key_matched(
    group_id: int,
    match_key: Tuple[str, int, int],
    records_by_position: Dict[str, Dict[str, object]],
    required_positions: List[str],
) -> List[Dict[str, object]]:
    if any(position not in records_by_position for position in required_positions):
        return []

    base_exercise, with_rifle, occurrence_rank = match_key
    records = [records_by_position[position] for position in required_positions]
    counts = [_count_extractable_windows(record, alignment_mode="matched_sessions") for record in records]
    n_windows = min(counts) if counts else 0
    windows: List[Dict[str, object]] = []
    for subwindow_idx in range(n_windows):
        windows.append({
            "group_id": int(group_id),
            "exercise": base_exercise,
            "with_rifle": int(with_rifle),
            "occurrence_rank": int(occurrence_rank),
            "subwindow_idx": int(subwindow_idx),
            "window_start_ms": np.nan,
            "window_end_ms": np.nan,
            "window_id": make_matched_window_id(group_id, base_exercise, with_rifle, occurrence_rank, subwindow_idx),
            "alignment_mode": "matched_sessions",
        })
    return windows


def build_canonical_window_manifest(
    records_by_group: Dict[int, Dict[str, Dict[Tuple[str, int, int], Dict[str, object]]]],
    required_positions: List[str],
    alignment_mode: str = "strict_overlap",
) -> pd.DataFrame:
    manifest_rows: List[Dict[str, object]] = []

    for group_id in sorted(records_by_group):
        group_records = records_by_group[group_id]
        common_keys = None
        for position in required_positions:
            position_records = group_records.get(position, {})
            keys = set(position_records.keys())
            common_keys = keys if common_keys is None else (common_keys & keys)
        if not common_keys:
            continue

        for match_key in sorted(common_keys):
            records_for_key = {
                position: group_records[position][match_key]
                for position in required_positions
                if match_key in group_records.get(position, {})
            }
            if alignment_mode == "matched_sessions":
                manifest_rows.extend(_build_windows_for_match_key_matched(group_id, match_key, records_for_key, required_positions))
            else:
                manifest_rows.extend(_build_windows_for_match_key_strict(group_id, match_key, records_for_key, required_positions))

    cols = [
        "manifest_index",
        "group_id",
        "exercise",
        "with_rifle",
        "occurrence_rank",
        "subwindow_idx",
        "window_start_ms",
        "window_end_ms",
        "window_id",
        "alignment_mode",
    ]
    if not manifest_rows:
        return pd.DataFrame(columns=cols)

    manifest = pd.DataFrame(manifest_rows)
    manifest = manifest.sort_values(
        ["group_id", "exercise", "with_rifle", "occurrence_rank", "subwindow_idx", "window_id"]
    ).reset_index(drop=True)
    manifest.insert(0, "manifest_index", np.arange(len(manifest), dtype=int))
    return manifest


def _extract_window_bounds(position_record: Dict[str, object], window_spec: Dict[str, object]) -> Tuple[float, float]:
    alignment_mode = str(window_spec.get("alignment_mode", "strict_overlap"))
    subwindow_idx = int(window_spec.get("subwindow_idx", 0))

    if alignment_mode == "matched_sessions":
        bounds = _matched_bounds_for_record(position_record, subwindow_idx)
    else:
        ws = float(window_spec["window_start_ms"])
        we = float(window_spec["window_end_ms"])
        bounds = (ws, we)

    if bounds is None:
        raise ValueError(
            f"Could not derive window bounds for position={position_record['position']} subwindow_idx={subwindow_idx}"
        )
    return bounds


def append_window_sample(
    position_record: Dict[str, object],
    window_spec: Dict[str, object],
    list_of_data_arrays_in_the_time_domain: List[List[np.ndarray]],
    list_of_data_arrays_in_the_frequency_domain: List[List[np.ndarray]],
) -> Dict[str, float]:
    acc_df = position_record["acc"]
    gyr_df = position_record["gyr"]
    start_ms, end_ms = _extract_window_bounds(position_record, window_spec)

    temp_time: List[np.ndarray] = []
    temp_freq: List[np.ndarray] = []
    channel_sources = [acc_df, acc_df, acc_df, acc_df, gyr_df, gyr_df, gyr_df, gyr_df]

    for source_df, (_, column_name) in zip(channel_sources, CHANNEL_SPECS):
        one_time: List[np.ndarray] = []
        one_freq: List[np.ndarray] = []
        add_timestamp_window_to_time_and_frequency_data_lists(
            source_df["timestamp"],
            source_df[column_name],
            start_ms,
            end_ms,
            one_time,
            one_freq,
            output_size=OUTPUT_ARRAY_SIZE,
        )
        temp_time.append(one_time[0])
        temp_freq.append(one_freq[0])

    for channel_idx in range(8):
        list_of_data_arrays_in_the_time_domain[channel_idx].append(temp_time[channel_idx])
        list_of_data_arrays_in_the_frequency_domain[channel_idx].append(temp_freq[channel_idx])

    return {
        "window_start_ms": float(start_ms),
        "window_end_ms": float(end_ms),
    }


def create_metadata_row(position: str, window_spec: Dict[str, object], extracted_bounds: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    activity_name = str(window_spec["exercise"])
    if int(window_spec["with_rifle"]) == 1 and not activity_name.startswith("OM"):
        activity_name = f"{activity_name}_with_rifle"

    start_ms = float(window_spec["window_start_ms"]) if pd.notna(window_spec.get("window_start_ms", np.nan)) else np.nan
    end_ms = float(window_spec["window_end_ms"]) if pd.notna(window_spec.get("window_end_ms", np.nan)) else np.nan
    if extracted_bounds is not None:
        start_ms = float(extracted_bounds["window_start_ms"])
        end_ms = float(extracted_bounds["window_end_ms"])

    duration_ms = np.nan
    if pd.notna(start_ms) and pd.notna(end_ms):
        duration_ms = float(end_ms) - float(start_ms)

    return {
        "position": position.lower(),
        "window_id": window_spec["window_id"],
        "group_id": int(window_spec["group_id"]),
        "y_true": int(create_labels(activity_name)),
        "exercise": str(window_spec["exercise"]),
        "with_rifle": int(window_spec["with_rifle"]),
        "occurrence_rank": int(window_spec["occurrence_rank"]),
        "subwindow_idx": int(window_spec["subwindow_idx"]),
        "alignment_mode": str(window_spec.get("alignment_mode", "strict_overlap")),
        "window_start_ms": start_ms,
        "window_end_ms": end_ms,
        "duration_ms": duration_ms,
    }
