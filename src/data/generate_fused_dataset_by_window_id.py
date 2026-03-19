import argparse
import os
from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_training_builders import create_directory_if_does_not_exist


DEFAULT_POSITIONS = ["chest", "left", "right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse pre-generated per-position datasets by inner-joining on window_id "
            "and concatenating channels across positions."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Root directory containing the per-position dataset folders (default: script directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Root output directory (default: same as --input).",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=DEFAULT_POSITIONS,
        choices=DEFAULT_POSITIONS,
        help="Positions to fuse, in concatenation order (default: chest left right).",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=None,
        help=(
            "Dataset folder names to read, in the same order as --positions. "
            "Defaults to the position names themselves. Example: --positions chest left right "
            "--dataset-dirs chest_460 left right"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output fused dataset folder name (default: joined position names, e.g. chest_left_right).",
    )
    parser.add_argument(
        "--require-label-match",
        action="store_true",
        help="Fail if matched rows disagree on y_true across positions.",
    )
    parser.add_argument(
        "--require-group-match",
        action="store_true",
        help="Fail if matched rows disagree on group_id across positions.",
    )
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> Tuple[str, str, List[str], List[str], str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = args.input or script_dir
    output_root = args.output or input_root
    positions = [p.lower() for p in args.positions]

    if args.dataset_dirs is None:
        dataset_dirs = positions
    else:
        dataset_dirs = args.dataset_dirs
        if len(dataset_dirs) != len(positions):
            raise ValueError("--dataset-dirs must have the same length as --positions")

    fused_name = args.name or "_".join(positions)
    return input_root, output_root, positions, dataset_dirs, fused_name


def load_metadata(dataset_root: str) -> pd.DataFrame:
    candidates = [
        os.path.join(dataset_root, "data", "metadata.csv"),
        os.path.join(dataset_root, "labels", "metadata.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            meta = pd.read_csv(path)
            if "window_id" not in meta.columns:
                raise ValueError(f"Metadata at {path} is missing required column 'window_id'")
            if "index" not in meta.columns:
                meta = meta.reset_index().rename(columns={"index": "index"})
            return meta

    window_ids_path = os.path.join(dataset_root, "labels", "window_ids.npy")
    labels_path = os.path.join(dataset_root, "labels", "labels.npy")
    groups_path = os.path.join(dataset_root, "labels", "groups.npy")
    if os.path.exists(window_ids_path) and os.path.exists(labels_path) and os.path.exists(groups_path):
        window_ids = np.load(window_ids_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
        groups = np.load(groups_path, allow_pickle=True)
        if not (len(window_ids) == len(labels) == len(groups)):
            raise ValueError(
                f"Fallback metadata arrays in {dataset_root} are misaligned: "
                f"window_ids={len(window_ids)}, labels={len(labels)}, groups={len(groups)}"
            )
        return pd.DataFrame(
            {
                "index": np.arange(len(window_ids), dtype=int),
                "window_id": window_ids.astype(object),
                "group_id": groups,
                "y_true": labels,
            }
        )

    raise FileNotFoundError(
        f"Could not find metadata.csv nor labels/window_ids.npy in {dataset_root}"
    )


def load_arrays(dataset_root: str) -> Tuple[np.ndarray, np.ndarray]:
    time_path = os.path.join(dataset_root, "data", "data_time_domain.npy")
    freq_path = os.path.join(dataset_root, "data", "data_frequency_domain.npy")
    if not os.path.exists(time_path):
        raise FileNotFoundError(f"Missing time-domain array: {time_path}")
    if not os.path.exists(freq_path):
        raise FileNotFoundError(f"Missing frequency-domain array: {freq_path}")

    time_arr = np.load(time_path)
    freq_arr = np.load(freq_path)
    return time_arr, freq_arr


def prepare_position_table(position: str, dataset_dir: str, dataset_root: str) -> pd.DataFrame:
    meta = load_metadata(dataset_root).copy()
    required_cols = ["index", "window_id"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata for {dataset_dir} is missing columns: {missing}")

    keep_cols = ["index", "window_id"]
    if "group_id" in meta.columns:
        keep_cols.append("group_id")
    if "y_true" in meta.columns:
        keep_cols.append("y_true")

    meta = meta[keep_cols].rename(
        columns={
            "index": f"index_{position}",
            "group_id": f"group_id_{position}",
            "y_true": f"y_true_{position}",
        }
    )

    if meta["window_id"].duplicated().any():
        dupes = meta.loc[meta["window_id"].duplicated(), "window_id"].astype(str).head(10).tolist()
        raise ValueError(
            f"Dataset '{dataset_dir}' has duplicate window_id values; cannot fuse safely. "
            f"Examples: {dupes}"
        )

    return meta


def build_joined_index(
    positions: List[str],
    dataset_dirs: List[str],
    input_root: str,
    require_label_match: bool,
    require_group_match: bool,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    tables = []
    roots: Dict[str, str] = {}

    for position, dataset_dir in zip(positions, dataset_dirs):
        dataset_root = os.path.join(input_root, dataset_dir)
        roots[position] = dataset_root
        tables.append(prepare_position_table(position, dataset_dir, dataset_root))

    merged = reduce(lambda left, right: pd.merge(left, right, on="window_id", how="inner"), tables)
    if merged.empty:
        raise ValueError("No common window_id values were found across the requested datasets")

    group_cols = [c for c in merged.columns if c.startswith("group_id_")]
    label_cols = [c for c in merged.columns if c.startswith("y_true_")]

    if require_group_match and group_cols:
        group_ok = merged[group_cols].nunique(axis=1) == 1
        if not bool(group_ok.all()):
            bad = merged.loc[~group_ok, ["window_id", *group_cols]].head(10)
            raise ValueError(
                "Matched rows disagree on group_id across positions. Examples:\n"
                f"{bad.to_string(index=False)}"
            )

    if require_label_match and label_cols:
        label_ok = merged[label_cols].nunique(axis=1) == 1
        if not bool(label_ok.all()):
            bad = merged.loc[~label_ok, ["window_id", *label_cols]].head(10)
            raise ValueError(
                "Matched rows disagree on y_true across positions. Examples:\n"
                f"{bad.to_string(index=False)}"
            )

    sort_cols = [f"group_id_{positions[0]}", "window_id"] if f"group_id_{positions[0]}" in merged.columns else ["window_id"]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged, roots


def select_reference_column(frame: pd.DataFrame, prefix: str, positions: List[str]) -> str:
    for position in positions:
        column = f"{prefix}_{position}"
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find any column with prefix '{prefix}'")


def main() -> None:
    args = parse_args()
    input_root, output_root, positions, dataset_dirs, fused_name = resolve_dirs(args)

    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"Positions  : {positions}")
    print(f"Datasets   : {dataset_dirs}")

    joined, dataset_roots = build_joined_index(
        positions=positions,
        dataset_dirs=dataset_dirs,
        input_root=input_root,
        require_label_match=args.require_label_match,
        require_group_match=args.require_group_match,
    )

    time_parts = []
    freq_parts = []

    for position in positions:
        dataset_root = dataset_roots[position]
        time_arr, freq_arr = load_arrays(dataset_root)

        index_col = f"index_{position}"
        indices = joined[index_col].to_numpy(dtype=int)

        if np.any(indices < 0) or np.any(indices >= time_arr.shape[0]):
            raise IndexError(
                f"Selected indices for position '{position}' fall outside time array bounds "
                f"(max index {time_arr.shape[0] - 1})"
            )
        if np.any(indices < 0) or np.any(indices >= freq_arr.shape[0]):
            raise IndexError(
                f"Selected indices for position '{position}' fall outside freq array bounds "
                f"(max index {freq_arr.shape[0] - 1})"
            )

        time_parts.append(time_arr[indices])
        freq_parts.append(freq_arr[indices])

    time_shapes = {part.shape[:2] for part in time_parts}
    freq_shapes = {part.shape[:2] for part in freq_parts}
    if len(time_shapes) != 1:
        raise ValueError(
            "Matched time-domain arrays do not have compatible leading shapes. "
            f"Found: {sorted(time_shapes)}"
        )
    if len(freq_shapes) != 1:
        raise ValueError(
            "Matched frequency-domain arrays do not have compatible leading shapes. "
            f"Found: {sorted(freq_shapes)}"
        )

    fused_time = np.concatenate(time_parts, axis=2)
    fused_freq = np.concatenate(freq_parts, axis=2)

    label_ref = select_reference_column(joined, "y_true", positions) if any(
        c.startswith("y_true_") for c in joined.columns
    ) else None
    group_ref = select_reference_column(joined, "group_id", positions) if any(
        c.startswith("group_id_") for c in joined.columns
    ) else None

    labels = joined[label_ref].to_numpy() if label_ref else np.array([], dtype=int)
    groups = joined[group_ref].to_numpy() if group_ref else np.array([], dtype=int)
    window_ids = joined["window_id"].to_numpy(dtype=object)

    out_root = os.path.join(output_root, fused_name)
    data_out_dir = os.path.join(out_root, "data")
    label_out_dir = os.path.join(out_root, "labels")
    create_directory_if_does_not_exist(data_out_dir)
    create_directory_if_does_not_exist(label_out_dir)

    np.save(os.path.join(data_out_dir, "data_time_domain.npy"), fused_time)
    np.save(os.path.join(data_out_dir, "data_frequency_domain.npy"), fused_freq)
    np.save(os.path.join(label_out_dir, "labels.npy"), labels)
    np.save(os.path.join(label_out_dir, "groups.npy"), groups)
    np.save(os.path.join(label_out_dir, "window_ids.npy"), window_ids)

    metadata = pd.DataFrame(
        {
            "index": np.arange(len(window_ids), dtype=int),
            "positions": ["|".join(positions)] * len(window_ids),
            "window_id": window_ids,
            "group_id": groups,
            "y_true": labels,
        }
    )
    metadata.to_csv(os.path.join(data_out_dir, "metadata.csv"), index=False)
    metadata.to_csv(os.path.join(label_out_dir, "metadata.csv"), index=False)

    counts_by_position = {position: int(joined[f"index_{position}"].notna().sum()) for position in positions}
    print("\nDone.")
    print(f"  Matched windows : {len(window_ids)}")
    print(f"  Time shape      : {fused_time.shape}")
    print(f"  Freq shape      : {fused_freq.shape}")
    print(f"  Output dir      : {out_root}")
    print(f"  Per-position rows used: {counts_by_position}")


if __name__ == "__main__":
    main()
