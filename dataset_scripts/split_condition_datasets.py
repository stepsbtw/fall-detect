import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


REQUIRED_ARRAYS: List[Tuple[str, str]] = [
    ("data", "data_time_domain.npy"),
    ("labels", "labels.npy"),
    ("labels", "groups.npy"),
]
OPTIONAL_ARRAYS: List[Tuple[str, str]] = [
    ("data", "data_frequency_domain.npy"),
    ("labels", "window_ids.npy"),
]


def _window_id_without_with_rifle(window_id) -> str:
    text = str(window_id)
    parts = text.split("|")
    # Expected formats:
    #   group|exercise|with_rifle|occurrence_rank|window_idx
    #   group|exercise|with_rifle|occurrence_rank|window_start|window_end
    if len(parts) >= 5 and parts[2] in {"0", "1"}:
        return "|".join(parts[:2] + parts[3:])
    return text


def _rewrite_saved_window_ids_without_with_rifle(target_dir: str) -> None:
    window_ids_path = os.path.join(target_dir, "labels", "window_ids.npy")
    if not os.path.isfile(window_ids_path):
        return

    window_ids = np.load(window_ids_path, allow_pickle=True)
    rewritten = np.asarray([_window_id_without_with_rifle(wid) for wid in window_ids], dtype=object)
    np.save(window_ids_path, rewritten)


def _metadata_path_for_dataset(dataset_dir: str) -> str:
    candidates = [
        os.path.join(dataset_dir, "data", "metadata.csv"),
        os.path.join(dataset_dir, "labels", "metadata.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Missing metadata.csv in {dataset_dir}")


def _has_required_arrays(dataset_dir: str) -> bool:
    for subdir, filename in REQUIRED_ARRAYS:
        if not os.path.isfile(os.path.join(dataset_dir, subdir, filename)):
            return False
    return True


def _discover_candidate_datasets(dataset_root: str, include_existing_splits: bool) -> List[str]:
    datasets: List[str] = []
    if not os.path.isdir(dataset_root):
        return datasets

    for entry in sorted(os.listdir(dataset_root)):
        dataset_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(dataset_dir):
            continue
        if not include_existing_splits and (entry.endswith("_armed") or entry.endswith("_unarmed")):
            continue
        if not _has_required_arrays(dataset_dir):
            continue
        try:
            _metadata_path_for_dataset(dataset_dir)
        except FileNotFoundError:
            continue
        datasets.append(entry)
    return datasets


def _load_metadata_with_source_index(dataset_dir: str) -> pd.DataFrame:
    metadata_path = _metadata_path_for_dataset(dataset_dir)
    metadata = pd.read_csv(metadata_path)
    if "with_rifle" not in metadata.columns:
        raise ValueError(f"Column 'with_rifle' is required in {metadata_path}")

    out = metadata.copy()
    if "index" in out.columns:
        out["__src_idx"] = pd.to_numeric(out["index"], errors="raise").astype(int)
    else:
        out["__src_idx"] = np.arange(len(out), dtype=int)

    if out["__src_idx"].duplicated().any():
        raise ValueError(f"Duplicate source indices found in {metadata_path}")

    return out


def _slice_and_save_arrays(dataset_dir: str, out_dir: str, indices: np.ndarray) -> None:
    for subdir, filename in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
        src_path = os.path.join(dataset_dir, subdir, filename)
        if not os.path.isfile(src_path):
            if (subdir, filename) in REQUIRED_ARRAYS:
                raise FileNotFoundError(f"Missing required file: {src_path}")
            continue

        arr = np.load(src_path, allow_pickle=True)
        if arr.ndim == 0:
            raise ValueError(f"Expected array with sample axis in {src_path}, got scalar.")
        if indices.size > 0 and int(indices.max()) >= int(arr.shape[0]):
            raise IndexError(
                f"Source index out of bounds for {src_path}: max_idx={int(indices.max())}, n_samples={int(arr.shape[0])}"
            )

        dst_path = os.path.join(out_dir, subdir, filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, arr[indices])


def _split_single_dataset(
    dataset_root: str,
    dataset_name: str,
    armed_value: int,
    armed_suffix: str,
    unarmed_suffix: str,
    overwrite: bool,
    drop_with_rifle_from_window_id: bool,
) -> None:
    dataset_dir = os.path.join(dataset_root, dataset_name)
    metadata = _load_metadata_with_source_index(dataset_dir)

    split_specs = [
        (armed_suffix, int(armed_value)),
        (unarmed_suffix, 1 - int(armed_value)),
    ]

    for split_name, split_value in split_specs:
        mask = pd.to_numeric(metadata["with_rifle"], errors="raise").astype(int) == int(split_value)
        subset = metadata.loc[mask].copy()
        subset = subset.sort_values("__src_idx").reset_index(drop=True)
        indices = subset["__src_idx"].to_numpy(dtype=int)

        target_name = f"{dataset_name}_{split_name}"
        target_dir = os.path.join(dataset_root, target_name)

        if os.path.isdir(target_dir) and not overwrite:
            print(f"[SKIP] {target_name} already exists. Use --overwrite to regenerate.")
            continue

        if len(indices) == 0:
            print(f"[WARN] {target_name}: no samples found for with_rifle={split_value}, skipping.")
            continue

        _slice_and_save_arrays(dataset_dir, target_dir, indices)

        subset = subset.drop(columns=["__src_idx"])
        if drop_with_rifle_from_window_id and "window_id" in subset.columns:
            subset["window_id"] = subset["window_id"].map(_window_id_without_with_rifle)

        if "index" in subset.columns:
            subset["index"] = np.arange(len(subset), dtype=int)
        else:
            subset.insert(0, "index", np.arange(len(subset), dtype=int))

        if drop_with_rifle_from_window_id:
            _rewrite_saved_window_ids_without_with_rifle(target_dir)

        if "window_id" in subset.columns and subset["window_id"].duplicated().any():
            dup_count = int(subset["window_id"].duplicated().sum())
            print(
                f"[WARN] {target_name}: {dup_count} duplicated condition-agnostic window_id values "
                "after dropping with_rifle."
            )

        data_meta_path = os.path.join(target_dir, "data", "metadata.csv")
        labels_meta_path = os.path.join(target_dir, "labels", "metadata.csv")
        os.makedirs(os.path.dirname(data_meta_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_meta_path), exist_ok=True)
        subset.to_csv(data_meta_path, index=False)
        subset.to_csv(labels_meta_path, index=False)

        print(
            f"[OK] {target_name}: {len(subset)} samples "
            f"(with_rifle={split_value})"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split datasets into armed/unarmed subsets using metadata.with_rifle"
    )
    parser.add_argument(
        "--dataset-root",
        default=os.path.join(os.path.dirname(__file__), "..", "dataset"),
        help="Root folder containing dataset directories",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to split (default: auto-discover)",
    )
    parser.add_argument(
        "--armed-value",
        type=int,
        default=1,
        choices=[0, 1],
        help="Value in with_rifle considered as armed (default: 1)",
    )
    parser.add_argument(
        "--armed-suffix",
        default="armed",
        help="Suffix for armed split",
    )
    parser.add_argument(
        "--unarmed-suffix",
        default="unarmed",
        help="Suffix for unarmed split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously generated split datasets",
    )
    parser.add_argument(
        "--include-existing-splits",
        action="store_true",
        help="Include datasets already ending with _armed/_unarmed in auto-discovery",
    )
    parser.add_argument(
        "--drop-with-rifle-from-window-id",
        action="store_true",
        help=(
            "Rewrite split output window IDs by removing the with_rifle token "
            "(enables armed/unarmed alignment in cross-condition evaluation)."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if args.datasets is None:
        datasets = _discover_candidate_datasets(
            dataset_root=dataset_root,
            include_existing_splits=bool(args.include_existing_splits),
        )
    else:
        datasets = list(args.datasets)

    if not datasets:
        print("No datasets found to split.")
        return

    print(f"Dataset root: {dataset_root}")
    print(f"Datasets to split: {datasets}")

    for dataset_name in datasets:
        print(f"\n== Splitting {dataset_name} ==")
        _split_single_dataset(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            armed_value=int(args.armed_value),
            armed_suffix=str(args.armed_suffix),
            unarmed_suffix=str(args.unarmed_suffix),
            overwrite=bool(args.overwrite),
            drop_with_rifle_from_window_id=bool(args.drop_with_rifle_from_window_id),
        )


if __name__ == "__main__":
    main()
