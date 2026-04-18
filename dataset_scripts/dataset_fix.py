from __future__ import annotations

import argparse
import csv
import io
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CORRECT_YEAR = 2024
OUTLIER_THRESHOLD = 10_000
TIMESTAMP_FILES = (
    ("sampling", ["beginning", "ending"]),
    ("acceleration", ["timestamp"]),
    ("angular_speed", ["timestamp"]),
)
TRUNCATED_CASES = (("ID5", "LEFT"), ("ID6", "RIGHT"))  # legacy recovery for datasets damaged by older fixes
LABEL_MAPPING = {
    "ADL_11": "ADL_9",
    "ADL_12": "ADL_10",
    "ADL_13": "ADL_11",
    "ADL_14": "ADL_12",
    "ADL_15": "ADL_13",
    "FALL_5": "FALL_4",
    "FALL_6": "FALL_5",
    "ADL_11_R": "ADL_9_R",
    "ADL_12_R": "ADL_10_R",
    "ADL_13_R": "ADL_11_R",
    "ADL_14_R": "ADL_12_R",
    "ADL_15_R": "ADL_13_R",
    "FALL_5_R": "FALL_4_R",
    "FALL_6_R": "FALL_5_R",
    "Rigth": "Right",
}


@dataclass
class Summary:
    csv_rows: int = 0
    ods_cells: int = 0
    files_written: int = 0


@dataclass
class ChangeCount:
    files_changed: int = 0
    values_changed: int = 0

    def add(self, files: int, values: int) -> None:
        self.files_changed += files
        self.values_changed += values


def find_default_dataset_root(script_path: pathlib.Path) -> pathlib.Path:
    return (script_path.parent.parent / "dataset").resolve()


def resolve_raw_root(dataset_root: pathlib.Path) -> pathlib.Path:
    for candidate in (dataset_root / "raw", dataset_root / "0_raw"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find raw root under dataset/raw or dataset/0_raw")


def iter_users(raw_root: pathlib.Path) -> list[str]:
    users = [p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("ID")]
    return sorted(users, key=lambda name: int(name[2:]))


def sensor_positions(user_root: pathlib.Path) -> list[str]:
    return sorted(p.name for p in user_root.iterdir() if p.is_dir())


def maybe_backup(path: pathlib.Path, use_backup: bool, dry_run: bool, suffix: str = ".bak") -> None:
    if not use_backup or dry_run:
        return
    bak = pathlib.Path(str(path) + suffix)
    if not bak.exists() and path.exists():
        shutil.copy2(path, bak)


def read_csv_rows(path: pathlib.Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv_rows(path: pathlib.Path, rows: list[dict[str, str]], fieldnames: list[str], dry_run: bool) -> bool:
    if dry_run:
        return False
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def _new_target_map(mapping: dict[str, str]) -> dict[str, str]:
    return {new: old for old, new in mapping.items() if new not in mapping}


def _find_files_for_labels(dataset_root: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    return sorted(dataset_root.rglob("*_sampling.csv")), sorted(dataset_root.rglob("*_README.ods"))

def rename_labels(dataset_root: pathlib.Path, dry_run: bool) -> Summary:
    summary = Summary()
    csv_files, ods_files = _find_files_for_labels(dataset_root)
    print(f"Found {len(csv_files)} sampling CSV file(s) and {len(ods_files)} README ODS file(s).")

    for path in csv_files:
        df = pd.read_csv(path)
        if "exercise" not in df.columns:
            continue

        old = df["exercise"].copy()
        df["exercise"] = df["exercise"].replace(LABEL_MAPPING)
        changed = int((old != df["exercise"]).sum())

        if changed:
            if not dry_run:
                df.to_csv(path, index=False)
                summary.files_written += 1
            summary.csv_rows += changed
            print(f"  CSV {path.relative_to(dataset_root)}: {changed} row(s) changed")

    try:
        import odf.opendocument as opendoc
        from odf.table import Table, TableCell, TableRow
        from odf.text import P
    except ImportError:
        print("  Skipping ODS rename: odfpy not installed")
        return summary

    for path in ods_files:
        doc = opendoc.load(str(path))
        changed = 0

        for sheet in doc.getElementsByType(Table):
            for row in sheet.getElementsByType(TableRow):
                for cell in row.getElementsByType(TableCell):
                    for para in cell.getElementsByType(P):
                        for node in para.childNodes:
                            if getattr(node, "nodeType", None) == node.TEXT_NODE:
                                new = LABEL_MAPPING.get(node.data, node.data)
                                if new != node.data:
                                    node.data = new
                                    changed += 1

        if changed:
            if not dry_run:
                doc.save(str(path))
                summary.files_written += 1
            summary.ods_cells += changed
            print(f"  ODS {path.relative_to(dataset_root)}: {changed} cell(s) changed")

    return summary


def _file_for(uid: str, pos: str, kind: str) -> str:
    return f"{uid}_{pos}_{kind}.csv"


def _apply_timestamp_offset(
    base_dir: pathlib.Path,
    uid: str,
    pos: str,
    offset_ms: int,
    dry_run: bool,
    backup: bool,
    wrong_year_only: bool,
) -> tuple[int, int]:
    files_changed = 0
    values_changed = 0

    for kind, columns in TIMESTAMP_FILES:
        path = base_dir / _file_for(uid, pos, kind)
        if not path.exists():
            continue

        df = pd.read_csv(path)
        changed_here = 0

        for col in columns:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce")
            valid = s.notna()

            if wrong_year_only:
                years = pd.to_datetime(s[valid], unit="ms", errors="coerce").dt.year
                mask = pd.Series(False, index=df.index)
                mask.loc[years.index] = years != CORRECT_YEAR
            else:
                mask = valid

            count = int(mask.sum())
            if count:
                df.loc[mask, col] = (s.loc[mask] + offset_ms).astype("int64")
                changed_here += count

        if changed_here:
            maybe_backup(path, backup, dry_run)
            if not dry_run:
                df.to_csv(path, index=False)
                files_changed += 1
            values_changed += changed_here
            scope = "wrong-year rows only" if wrong_year_only else "all rows"
            print(f"  {path.name}: {changed_here} timestamp value(s) shifted ({scope})")

    return files_changed, values_changed


def patch_wrong_year(raw_root: pathlib.Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    total = ChangeCount()

    for uid in iter_users(raw_root):
        user_root = raw_root / uid
        chest_path = user_root / "CHEST" / _file_for(uid, "CHEST", "sampling")
        if not chest_path.exists():
            continue

        chest_beginning = pd.read_csv(chest_path, usecols=["beginning"])["beginning"]
        chest_ref = int(chest_beginning.iloc[0])
        chest_year = pd.to_datetime(chest_ref, unit="ms").year
        if chest_year < CORRECT_YEAR:
            print(f"  WARNING: {uid}/CHEST year={chest_year}, skipping user")
            continue

        for pos in sensor_positions(user_root):
            if pos == "CHEST":
                continue

            samp_path = user_root / pos / _file_for(uid, pos, "sampling")
            if not samp_path.exists():
                continue

            beginning = pd.read_csv(samp_path, usecols=["beginning"])["beginning"]
            years = pd.to_datetime(beginning, unit="ms", errors="coerce").dt.year
            wrong_mask = years != CORRECT_YEAR
            if not wrong_mask.any():
                continue

            first_wrong = int(beginning[wrong_mask].iloc[0])
            offset = chest_ref - first_wrong
            mixed = bool(wrong_mask.sum() != len(beginning))
            before = pd.to_datetime(first_wrong, unit="ms").date()
            after = pd.to_datetime(first_wrong + offset, unit="ms").date()
            mode = "wrong-year rows only" if mixed else "entire file"
            print(f"{uid}/{pos}: offset={offset:+d} ms ({before} -> {after}) [{mode}]")

            files, values = _apply_timestamp_offset(
                user_root / pos,
                uid,
                pos,
                offset,
                dry_run,
                backup,
                wrong_year_only=mixed,
            )
            total.add(files, values)

    return total.files_changed, total.values_changed


def restore_truncated(raw_root: pathlib.Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    total = ChangeCount()

    for uid, pos in TRUNCATED_CASES:
        base = raw_root / uid / pos
        ang_path = base / _file_for(uid, pos, "angular_speed")
        ang_bak = pathlib.Path(str(ang_path) + ".bak")
        samp_path = base / _file_for(uid, pos, "sampling")
        samp_bak = pathlib.Path(str(samp_path) + ".bak")

        if not ang_bak.exists() or not samp_bak.exists():
            print(f"{uid}/{pos}: missing .bak files, skipping")
            continue

        cur_ts = int(float(pd.read_csv(samp_path)["beginning"].iloc[0]))
        bak_ts = int(float(pd.read_csv(samp_bak)["beginning"].iloc[0]))
        offset = cur_ts - bak_ts
        print(f"{uid}/{pos}: restoring angular_speed from .bak with offset {offset:+d} ms")

        rows, fieldnames = read_csv_rows(ang_bak)
        changed = 0
        for row in rows:
            raw = (row.get("timestamp") or "").strip()
            if not raw:
                continue
            row["timestamp"] = str(int(float(raw)) + offset)
            changed += 1

        if changed:
            maybe_backup(ang_path, backup, dry_run, suffix=".truncated.bak")
            if write_csv_rows(ang_path, rows, fieldnames, dry_run):
                total.files_changed += 1
            total.values_changed += changed
            print(f"  {ang_path.name}: restored {len(rows)} row(s), shifted {changed} timestamp value(s)")

    return total.files_changed, total.values_changed


def _clean_outliers_in_df(
    df: pd.DataFrame, threshold: int
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    time_cols = {"timestamp", "beginning", "ending"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in time_cols]
    if not numeric_cols:
        return df, 0, pd.DataFrame()

    num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    mask = num.abs() > threshold
    changes = int(mask.sum().sum())

    if changes == 0:
        return df, 0, pd.DataFrame()

    outliers = []
    for col in numeric_cols:
        bad_rows = df.loc[mask[col], ["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame(index=df.index[mask[col]])
        bad_rows["column"] = col
        bad_rows["value"] = df.loc[mask[col], col].values
        bad_rows["row_index"] = df.index[mask[col]]
        outliers.append(bad_rows)

    outliers_df = pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame()

    num = num.mask(mask)
    num = num.interpolate(method="linear", limit_direction="both", axis=0)
    df[numeric_cols] = num
    return df, changes, outliers_df


def clean_outliers(raw_root: pathlib.Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    total = ChangeCount()

    for uid in iter_users(raw_root):
        for pos in sensor_positions(raw_root / uid):
            base = raw_root / uid / pos
            for kind in ("acceleration", "angular_speed"):
                path = base / _file_for(uid, pos, kind)
                if not path.exists():
                    continue

                df = pd.read_csv(path)
                df, changes, outliers_df = _clean_outliers_in_df(df, OUTLIER_THRESHOLD)

                if changes:
                    print(f"\n  {path.name}: {changes} outlier value(s) found")
                    print(outliers_df.to_string(index=False))

                    maybe_backup(path, backup, dry_run)
                    if not dry_run:
                        df.to_csv(path, index=False)
                        total.files_changed += 1
                    total.values_changed += changes
                    print(f"  {path.name}: {changes} outlier value(s) fixed")

    return total.files_changed, total.values_changed


def run_all(dataset_root: pathlib.Path, raw_root: pathlib.Path, dry_run: bool, backup: bool) -> None:
    print("== Step 1: rename labels ==")
    renamed = rename_labels(dataset_root, dry_run=dry_run)
    print(f"Summary: csv_rows={renamed.csv_rows}, ods_cells={renamed.ods_cells}, files_written={renamed.files_written}\n")

    print("== Step 2: patch wrong-year timestamps ==")
    files, values = patch_wrong_year(raw_root, dry_run=dry_run, backup=backup)
    print(f"Summary: files_changed={files}, values_changed={values}\n")

    print("== Step 3: clean outliers ==")
    files, values = clean_outliers(raw_root, dry_run=dry_run, backup=backup)
    print(f"Summary: files_changed={files}, values_changed={values}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified dataset fixes runner")
    parser.add_argument(
        "--dataset-root",
        default=str(find_default_dataset_root(pathlib.Path(__file__))),
        help="Dataset root that contains README.ods and raw folders (default: ../dataset)",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Raw data root (default: auto-detect dataset/raw or dataset/0_raw)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak safety files where applicable.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rename-labels", help="Run label remap fix over sampling CSV and README ODS files")
    sub.add_parser("patch-wrong-year", help="Patch wrong-year timestamp files using CHEST as reference")
    sub.add_parser("clean-outliers", help="Fix extreme sensor outliers via interpolation")
    sub.add_parser("run-all", help="Execute all fixes in the expected order")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = pathlib.Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {dataset_root}")
        return 1

    try:
        raw_root = pathlib.Path(args.raw_root).resolve() if args.raw_root else resolve_raw_root(dataset_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not raw_root.is_dir():
        print(f"ERROR: raw root does not exist: {raw_root}")
        return 1

    dry_run = not args.apply
    backup = not args.no_backup
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"dataset_root: {dataset_root}")
    print(f"raw_root:     {raw_root}")

    if args.command == "rename-labels":
        result = rename_labels(dataset_root, dry_run=dry_run)
        print(f"Summary: csv_rows={result.csv_rows}, ods_cells={result.ods_cells}, files_written={result.files_written}")
        return 0

    if args.command == "patch-wrong-year":
        files, values = patch_wrong_year(raw_root, dry_run=dry_run, backup=backup)
        print(f"Summary: files_changed={files}, values_changed={values}")
        return 0

    if args.command == "clean-outliers":
        files, values = clean_outliers(raw_root, dry_run=dry_run, backup=backup)
        print(f"Summary: files_changed={files}, values_changed={values}")
        return 0

    if args.command == "run-all":
        run_all(dataset_root, raw_root, dry_run=dry_run, backup=backup)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
