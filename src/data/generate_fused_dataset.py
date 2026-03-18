# Criado por Caio Passos - https://github.com/stepsbtw
#
# Generates a fused multi-sensor dataset by matching activities across sensor
# positions and resampling all signals to a common window size.
#
# Positions are matched by (exercise, withRifle, rank_within_individual).
#
# The dataset was collected across two campaigns: wrist sensors (LEFT/RIGHT) for
# some individuals were recorded in 2022, while CHEST was only introduced in
# 2024.  For those individuals the three positions come from separate sessions,
# so absolute timestamps cannot be used for cross-sensor alignment.
# For individuals where all positions were recorded on the same day (truly
# simultaneous, within ~2 seconds of each other) the rank-based matching still
# holds because activities are always performed in the same order.
#
# Long activities are split into 5-second chunks using each sensor's own
# timestamps (instead of observation counts), which guarantees the same number
# of sub-windows regardless of sampling rate.
#
# Usage examples:
#   python generate_fused_dataset.py
#       → fuses chest + left + right, target size 450, default paths
#   python generate_fused_dataset.py --positions chest left --target-size 450 \
#       --database path/to/raw --output path/to/out

import argparse
import os

import numpy as np
import pandas as pd

from data_training_builders import (
    create_directory_if_does_not_exist,
    create_labels,
    fourier_transform,
    get_file_path,
    sort_by_number,
    validate_sampling_segments,
)

# ── Activity type sets (same definition as the original pipeline) ─────────────
FIVE_SEC_ACTIVITIES = frozenset([
    "FALL_1", "FALL_2", "FALL_3", "FALL_4", "FALL_5",
    "ADL_5", "ADL_6", "ADL_7", "ADL_8", "ADL_13",
])
TRANSITION_ACTIVITIES = frozenset([
    "OM_3", "OM_4", "OM_5", "OM_6", "OM_7", "OM_8",
])

# Nominal window sizes used by the original pipeline (obs per 5-second window)
#DEFAULT_ARRAY_SIZES = {"CHEST": 1100, "LEFT": 460, "RIGHT": 460}
DEFAULT_ARRAY_SIZES = {"CHEST": 460, "LEFT": 460, "RIGHT": 460}


WINDOW_MS = 5_000  # 5 seconds expressed in milliseconds


# ── Core helpers ──────────────────────────────────────────────────────────────

def resample_channel(ts: np.ndarray, vals: np.ndarray, target_n: int) -> np.ndarray:
    """Linearly interpolate one channel onto a uniform grid of target_n points."""
    t_uniform = np.linspace(ts[0], ts[-1], target_n)
    return np.interp(t_uniform, ts, vals)


def slices_to_array(acc_s: pd.DataFrame, gyr_s: pd.DataFrame, target_n: int):
    """
    Resample acc + gyr slices to (target_n, 8) using each sensor's own timestamps.
    Channel order: magacc, acc_x, acc_y, acc_z, maggyr, gyr_x, gyr_y, gyr_z
    Returns None if there is not enough data to interpolate.
    """
    if len(acc_s) < 2 or len(gyr_s) < 2:
        return None
    acc_ts = acc_s["timestamp"].values.astype(float)
    gyr_ts = gyr_s["timestamp"].values.astype(float)
    try:
        channels = [
            resample_channel(acc_ts, acc_s["Magnitude"].values, target_n),
            resample_channel(acc_ts, acc_s["ax"].values, target_n),
            resample_channel(acc_ts, acc_s["ay"].values, target_n),
            resample_channel(acc_ts, acc_s["az"].values, target_n),
            resample_channel(gyr_ts, gyr_s["Magnitude"].values, target_n),
            resample_channel(gyr_ts, gyr_s["wx"].values, target_n),
            resample_channel(gyr_ts, gyr_s["wy"].values, target_n),
            resample_channel(gyr_ts, gyr_s["wz"].values, target_n),
        ]
    except Exception:
        return None
    return np.stack(channels, axis=1)  # (target_n, 8)


def extract_windows(
    acc_df: pd.DataFrame,
    gyr_df: pd.DataFrame,
    sampling_id: int,
    base_activity: str,
    target_n: int,
    array_size: int,
) -> list:
    """
    Extract all windows for one activity recording and return them as a list of
    (target_n, 8) arrays.  Mirrors the three-branch logic of the original pipeline
    but uses timestamp-based splitting for long activities so that the window count
    is independent of the sensor's sampling rate.
    """
    acc_act = acc_df[acc_df["sampling"] == sampling_id]
    gyr_act = gyr_df[gyr_df["sampling"] == sampling_id]

    if len(acc_act) < 2 or len(gyr_act) < 2:
        return []

    # ── Branch 1: fixed 5-second activities ──────────────────────────────────
    if base_activity in FIVE_SEC_ACTIVITIES:
        acc_s = acc_act.iloc[:array_size]
        gyr_s = gyr_act.iloc[:array_size]
        if len(acc_s) < array_size or len(gyr_s) < array_size:
            return []
        arr = slices_to_array(acc_s, gyr_s, target_n)
        return [arr] if arr is not None else []

    # ── Branch 2: transition activities (center on magacc peak) ──────────────
    elif base_activity in TRANSITION_ACTIVITIES:
        magacc = acc_act["Magnitude"].reset_index(drop=True)
        peak_idx = int(magacc.idxmax())
        half = array_size // 2
        start = max(0, peak_idx - half)
        end = start + array_size
        if end > len(acc_act):
            end = len(acc_act)
            start = max(0, end - array_size)
        acc_s = acc_act.iloc[start:end]
        gyr_s = gyr_act.iloc[start : start + array_size]
        if len(acc_s) < array_size or len(gyr_s) < array_size:
            return []
        arr = slices_to_array(acc_s, gyr_s, target_n)
        return [arr] if arr is not None else []

    # ── Branch 3: long activities — split by 5-second timestamp chunks ────────
    else:
        t_start = float(acc_act["timestamp"].iloc[0])
        t_end = min(float(acc_act["timestamp"].iloc[-1]),
                    float(gyr_act["timestamp"].iloc[-1]))
        n_windows = int((t_end - t_start) / WINDOW_MS)
        # Safety cap: mirrors the single-sensor pipeline in data_training_generators.py.
        # When a sensor runs slightly above its nominal rate it accumulates just
        # enough extra observations to push the timestamp duration past an extra
        # WINDOW_MS boundary, giving one more window than the single-sensor
        # pipeline (which caps at floor(min_obs / array_size)).  Capping here
        # keeps both pipelines in agreement so fused never produces more
        # sub-windows than single-sensor for the same activity.
        n_windows = min(n_windows, min(len(acc_act), len(gyr_act)) // array_size)
        windows = []
        for i in range(n_windows):
            ws = t_start + i * WINDOW_MS
            we = ws + WINDOW_MS
            acc_s = acc_act[(acc_act["timestamp"] >= ws) & (acc_act["timestamp"] < we)]
            gyr_s = gyr_act[(gyr_act["timestamp"] >= ws) & (gyr_act["timestamp"] < we)]
            arr = slices_to_array(acc_s, gyr_s, target_n)
            if arr is not None:
                windows.append(arr)
        return windows


def build_rank_map(sampling_df: pd.DataFrame) -> dict:
    """
    Build a mapping  (exercise, withRifle, rank_within_group) -> sampling_id.

    Rank is determined by ascending sampling_id, which corresponds to recording
    order within the session.  This key is used to match the same repetition of an
    activity across different sensor positions.
    """
    rank_map = {}
    for (exercise, with_rifle), group in sampling_df.groupby(["exercise", "withRifle"]):
        for rank, (_, row) in enumerate(group.sort_values("id").iterrows()):
            rank_map[(exercise, int(with_rifle), rank)] = int(row["id"])
    return rank_map


def compute_freq(time_arr: np.ndarray, target_n: int) -> np.ndarray:
    """Return the FFT magnitude for each channel (first target_n//2 components)."""
    half = target_n // 2
    freq = np.zeros((half, time_arr.shape[1]))
    for c in range(time_arr.shape[1]):
        freq[:, c] = fourier_transform(time_arr[:, c])[:half]
    return freq


# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate a fused multi-sensor dataset from multiple sensor positions"
)
parser.add_argument(
    "--positions", type=str, nargs="+", default=["chest", "left", "right"],
    choices=["chest", "left", "right"],
    help="Sensor positions to fuse (default: chest left right)",
)
parser.add_argument(
    "--database", type=str, default=None,
    help="Path to the database directory (default: ./database relative to this script)",
)
parser.add_argument(
    "--output", type=str, default=None,
    help="Path to the output directory (default: same as this script)",
)
parser.add_argument(
    "--target-size", type=int, default=460,
    help="Common window length after resampling (default: 460)",
)
parser.add_argument(
    "--strict-sampling-validation",
    action="store_true",
    help="Fail if sampling.csv boundaries disagree with sensor rows for any activity.",
)
args = parser.parse_args()

positions = [p.lower() for p in args.positions]
target_n = args.target_size
current_directory = os.path.dirname(os.path.abspath(__file__))
main_directory = args.database or os.path.join(current_directory, "database")
output_directory = args.output or current_directory

# Output folder is named after the fused positions, e.g. "chest_left_right"
fused_name = "_".join(positions)
data_out_dir = os.path.join(output_directory, fused_name, "data")
label_out_dir = os.path.join(output_directory, fused_name, "labels")
create_directory_if_does_not_exist(data_out_dir)
create_directory_if_does_not_exist(label_out_dir)

subdirectory_list = sorted(os.listdir(main_directory), key=sort_by_number)

data_time_list: list = []
data_freq_list: list = []
labels_list: list = []
groups_list: list = []
validation_reports: list = []

n_channels = 8 * len(positions)
print(f"Positions : {positions}")
print(f"Target size: {target_n}  |  Output channels: {n_channels}")
print(f"Database  : {main_directory}")
print(f"Output    : {os.path.join(output_directory, fused_name)}")
print()

# ── Main loop ─────────────────────────────────────────────────────────────────
for subdirectory in subdirectory_list:
    group_id = sort_by_number(subdirectory)
    print(f"  {subdirectory}...", end="", flush=True)

    # Load data and build rank maps for all required positions
    pos_data: dict = {}
    pos_rank_maps: dict = {}
    try:
        for pos in positions:
            acc_file, gyr_file, samp_file = get_file_path(
                main_directory, subdirectory, pos.upper()
            )
            acc_df = pd.read_csv(acc_file)
            gyr_df = pd.read_csv(gyr_file)
            sampling_df = pd.read_csv(samp_file)
            pos_data[pos] = (acc_df, gyr_df)
            pos_rank_maps[pos] = build_rank_map(sampling_df)

            validation_report = validate_sampling_segments(acc_df, gyr_df, sampling_df)
            validation_report.insert(0, "group_id", group_id)
            validation_report.insert(1, "position", pos.upper())
            validation_reports.append(validation_report)

            issue_count = int(validation_report["has_issue"].sum())
            if issue_count:
                print(f" VALIDATION_WARN[{pos.upper()}]={issue_count}", end="", flush=True)
                if args.strict_sampling_validation:
                    bad_ids = validation_report.loc[validation_report["has_issue"], "sampling_id"].tolist()
                    raise ValueError(
                        f"sampling validation failed for {subdirectory} / {pos.upper()}: "
                        f"{issue_count} problematic ids {bad_ids}"
                    )
    except FileNotFoundError as exc:
        print(f" SKIP ({exc})")
        continue

    # Activities that have a matching recording in every requested position
    common_keys = set(pos_rank_maps[positions[0]])
    for pos in positions[1:]:
        common_keys &= set(pos_rank_maps[pos])

    n_added = 0
    for key in sorted(common_keys):
        exercise, with_rifle, _ = key
        activity_name = (
            f"{exercise}_with_rifle"
            if with_rifle == 1 and not exercise.startswith("OM")
            else exercise
        )
        label = create_labels(activity_name)
        if label is None:
            continue

        # Extract windows independently for each position
        pos_windows: dict = {}
        for pos in positions:
            sid = pos_rank_maps[pos][key]
            acc_df, gyr_df = pos_data[pos]
            array_size = DEFAULT_ARRAY_SIZES[pos.upper()]
            pos_windows[pos] = extract_windows(
                acc_df, gyr_df, sid, exercise, target_n, array_size
            )

        # Only fuse up to the minimum number of sub-windows available
        min_count = min(len(pos_windows[pos]) for pos in positions)
        if min_count == 0:
            continue

        for w_idx in range(min_count):
            time_parts = [pos_windows[pos][w_idx] for pos in positions]
            fused_time = np.concatenate(time_parts, axis=1)          # (target_n, 8*n_pos)
            fused_freq = np.concatenate(
                [compute_freq(t, target_n) for t in time_parts], axis=1
            )                                                          # (target_n//2, 8*n_pos)

            data_time_list.append(fused_time)
            data_freq_list.append(fused_freq)
            labels_list.append(label)
            groups_list.append(group_id)
            n_added += 1

    print(f" OK ({n_added} windows)")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving...")
all_time = np.stack(data_time_list, axis=0)   # (N, target_n,   8*n_pos)
all_freq = np.stack(data_freq_list, axis=0)   # (N, target_n//2, 8*n_pos)

np.save(os.path.join(data_out_dir,   "data_time_domain.npy"),      all_time)
np.save(os.path.join(data_out_dir,   "data_frequency_domain.npy"),  all_freq)
np.save(os.path.join(label_out_dir,  "labels.npy"),  np.asarray(labels_list))
np.save(os.path.join(label_out_dir,  "groups.npy"),  np.asarray(groups_list))

window_ids = [
    f"{group_id}_{i}"
    for i, group_id in enumerate(groups_list)
]

df_meta = pd.DataFrame({
    "index": list(range(len(groups_list))),
    "window_id": window_ids,
    "group_id": groups_list,
    "y_true": labels_list,
})

df_meta.to_csv(os.path.join(label_out_dir, "metadata.csv"), index=False)

if validation_reports:
    validation_path = os.path.join(label_out_dir, "sampling_validation_report.csv")
    df_validation = pd.concat(validation_reports, ignore_index=True)
    df_validation.to_csv(validation_path, index=False)
    print(
        f"Sampling validation report saved to {validation_path} "
        f"({int(df_validation['has_issue'].sum())} rows with issues)."
    )

print(f"\nDone.")
print(f"  Samples     : {len(labels_list)}")
print(f"  Time shape  : {all_time.shape}   (N, {target_n}, {n_channels})")
print(f"  Freq shape  : {all_freq.shape}  (N, {target_n // 2}, {n_channels})")
print(f"  Labels      : {dict(zip(*np.unique(np.asarray(labels_list), return_counts=True)))}")
print(f"  Output dir  : {os.path.join(output_directory, fused_name)}")

