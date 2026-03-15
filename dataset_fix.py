"""
Unified dataset fixer for fall-detect.

This script aggregates the behaviors from:
  - fix.py
  - fix_timestamps.py
  - _patch_remaining.py
  - _fix_rowlevel.py
  - _restore_truncated.py

Examples:
  python aggregate_fixes.py run-all --apply
  python aggregate_fixes.py rename-labels --apply
  python aggregate_fixes.py patch-wrong-year --apply
  python aggregate_fixes.py row-level-fix --apply
  python aggregate_fixes.py restore-truncated --apply
"""

from __future__ import annotations

import argparse
import csv
import io
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

CORRECT_YEAR = 2024

LABEL_MAPPING = {
    "ADL_11": "ADL_9",
    "ADL_12": "ADL_10",
    "ADL_13": "ADL_11",
    "ADL_14": "ADL_12",
    "ADL_15": "ADL_13",
    "FALL_5": "FALL_4",
    "ADL_11_R": "ADL_9_R",
    "ADL_12_R": "ADL_10_R",
    "ADL_13_R": "ADL_11_R",
    "ADL_14_R": "ADL_12_R",
    "ADL_15_R": "ADL_13_R",
    "FALL_5_R": "FALL_4_R",
    "FALL_6": "FALL_5",
    "FALL_6_R": "FALL_5_R",
    "Rigth": "Right",
}

TIMESTAMP_FILES = (
    ("sampling", ["beginning", "ending"]),
    ("acceleration", ["timestamp"]),
    ("angular_speed", ["timestamp"]),
)


@dataclass
class Summary:
    csv_rows: int = 0
    ods_cells: int = 0
    files_written: int = 0


def find_default_dataset_root(script_path: pathlib.Path) -> pathlib.Path:
    return (script_path.parent.parent / "dataset").resolve()


def resolve_raw_root(dataset_root: pathlib.Path) -> pathlib.Path:
    candidates = [dataset_root / "raw", dataset_root / "0_raw"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find raw root. Tried: {', '.join(str(x) for x in candidates)}"
    )


def iter_users(raw_root: pathlib.Path) -> list[str]:
    users = [p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("ID")]
    return sorted(users, key=lambda x: int(x[2:]))


def maybe_backup(path: pathlib.Path, use_backup: bool, dry_run: bool) -> None:
    if not use_backup:
        return
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        return
    if dry_run:
        return
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
    return {v: k for k, v in mapping.items() if v not in mapping}


def _find_files_for_labels(dataset_root: pathlib.Path) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    csv_files = sorted(dataset_root.rglob("*_sampling.csv"))
    ods_files = sorted(dataset_root.rglob("*_README.ods"))
    return csv_files, ods_files


def _check_csv_conflicts(csv_files: Iterable[pathlib.Path], mapping: dict[str, str]) -> list[tuple[pathlib.Path, str]]:
    new_target_to_source = _new_target_map(mapping)
    conflicts: list[tuple[pathlib.Path, str]] = []

    for path in csv_files:
        labels = {row.get("exercise", "") for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))}
        for target, source in new_target_to_source.items():
            if source in labels and target in labels:
                conflicts.append((path, f"{source} -> {target}"))
    return conflicts


def rename_labels(dataset_root: pathlib.Path, dry_run: bool) -> Summary:
    summary = Summary()
    csv_files, ods_files = _find_files_for_labels(dataset_root)
    print(f"Found {len(csv_files)} sampling CSV file(s) and {len(ods_files)} README ODS file(s).")

    conflicts = _check_csv_conflicts(csv_files, LABEL_MAPPING)
    if conflicts:
        print("Conflict(s) detected in CSV labels. Aborting rename-labels.")
        for path, desc in conflicts:
            print(f"  {path.relative_to(dataset_root)}  {desc}")
        return summary

    for path in csv_files:
        rows, fieldnames = read_csv_rows(path)
        changed = 0
        for row in rows:
            old = row.get("exercise", "")
            new = LABEL_MAPPING.get(old, old)
            if new != old:
                row["exercise"] = new
                changed += 1
        if changed:
            if write_csv_rows(path, rows, fieldnames, dry_run):
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
        for sheet in doc.spreadsheet.getElementsByType(Table):
            for row in sheet.getElementsByType(TableRow):
                for cell in row.getElementsByType(TableCell):
                    for para in cell.getElementsByType(P):
                        for node in [c for c in para.childNodes if c.nodeType == c.TEXT_NODE]:
                            old = node.data
                            new = LABEL_MAPPING.get(old, old)
                            if new != old:
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


def _shift_timestamp_files(
    base_dir: pathlib.Path,
    uid: str,
    pos: str,
    offset_ms: int,
    dry_run: bool,
    backup: bool,
    only_wrong_year: bool = False,
) -> tuple[int, int]:
    files_changed = 0
    values_changed = 0

    for kind, ts_cols in TIMESTAMP_FILES:
        path = base_dir / _file_for(uid, pos, kind)
        if not path.exists():
            continue

        rows, fieldnames = read_csv_rows(path)
        changed_here = 0
        for row in rows:
            for col in ts_cols:
                raw_val = (row.get(col) or "").strip()
                if not raw_val:
                    continue
                ts = int(float(raw_val))
                if only_wrong_year and pd.to_datetime(ts, unit="ms").year == CORRECT_YEAR:
                    continue
                row[col] = str(ts + offset_ms)
                changed_here += 1

        if changed_here:
            maybe_backup(path, backup, dry_run)
            if write_csv_rows(path, rows, fieldnames, dry_run):
                files_changed += 1
            values_changed += changed_here
            print(f"  {path.name}: {changed_here} timestamp value(s) shifted")

    return files_changed, values_changed


def patch_wrong_year(raw_root: pathlib.Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    files_changed = 0
    values_changed = 0

    for uid in iter_users(raw_root):
        chest_path = raw_root / uid / "CHEST" / _file_for(uid, "CHEST", "sampling")
        if not chest_path.exists():
            continue

        chest = pd.read_csv(chest_path)
        chest_ref = int(chest["beginning"].iloc[0])
        chest_year = pd.to_datetime(chest_ref, unit="ms").year
        if chest_year < CORRECT_YEAR:
            print(f"  WARNING: {uid}/CHEST year={chest_year}, skipping user")
            continue

        for pos in ("LEFT", "RIGHT"):
            samp_path = raw_root / uid / pos / _file_for(uid, pos, "sampling")
            if not samp_path.exists():
                continue

            samp = pd.read_csv(samp_path)
            first_ts = int(samp["beginning"].iloc[0])
            year = pd.to_datetime(first_ts, unit="ms").year
            if year >= CORRECT_YEAR:
                continue

            offset = chest_ref - first_ts
            before = pd.to_datetime(first_ts, unit="ms").date()
            after = pd.to_datetime(first_ts + offset, unit="ms").date()
            print(f"{uid}/{pos}: offset={offset:+d} ms ({before} -> {after})")

            base = raw_root / uid / pos
            f, v = _shift_timestamp_files(base, uid, pos, offset, dry_run, backup)
            files_changed += f
            values_changed += v

    return files_changed, values_changed


def _offset_from_sampling_bak(raw_root: pathlib.Path, uid: str, pos: str) -> int:
    cur_path = raw_root / uid / pos / _file_for(uid, pos, "sampling")
    bak_path = pathlib.Path(str(cur_path) + ".bak")
    if not bak_path.exists():
        raise FileNotFoundError(f"Missing backup file for offset reconstruction: {bak_path}")
    cur_ts = int(float(pd.read_csv(cur_path)["beginning"].iloc[0]))
    bak_ts = int(float(pd.read_csv(bak_path)["beginning"].iloc[0]))
    return cur_ts - bak_ts


def row_level_fix(raw_root: pathlib.Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    files_changed = 0
    values_changed = 0

    # ID2/LEFT: undo over-shift for rows not in correct year
    offset_id2 = _offset_from_sampling_bak(raw_root, "ID2", "LEFT")
    print(f"ID2/LEFT row-level correction with offset {-offset_id2:+d} ms on wrong-year rows")
    f, v = _shift_timestamp_files(
        raw_root / "ID2" / "LEFT",
        "ID2",
        "LEFT",
        -offset_id2,
        dry_run,
        backup,
        only_wrong_year=True,
    )
    files_changed += f
    values_changed += v

    # ID7/LEFT: shift remaining wrong-year rows to correct year
    chest_ref = int(
        pd.read_csv(raw_root / "ID7" / "CHEST" / _file_for("ID7", "CHEST", "sampling"))["beginning"].iloc[0]
    )
    samp_id7 = pd.read_csv(raw_root / "ID7" / "LEFT" / _file_for("ID7", "LEFT", "sampling"))
    wrong_rows = samp_id7[pd.to_datetime(samp_id7["beginning"], unit="ms").dt.year != CORRECT_YEAR]
    if not wrong_rows.empty:
        first_wrong = int(wrong_rows["beginning"].iloc[0])
        offset_id7 = chest_ref - first_wrong
        print(f"ID7/LEFT row-level correction with offset {offset_id7:+d} ms on wrong-year rows")
        f, v = _shift_timestamp_files(
            raw_root / "ID7" / "LEFT",
            "ID7",
            "LEFT",
            offset_id7,
            dry_run,
            backup,
            only_wrong_year=True,
        )
        files_changed += f
        values_changed += v
    else:
        print("ID7/LEFT: no wrong-year rows found, nothing to patch")

    return files_changed, values_changed


def restore_truncated(raw_root: pathlib.Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    files_changed = 0
    values_changed = 0

    for uid, pos in (("ID5", "LEFT"), ("ID6", "RIGHT")):
        base = raw_root / uid / pos
        ang_path = base / _file_for(uid, pos, "angular_speed")
        ang_bak = pathlib.Path(str(ang_path) + ".bak")
        samp_path = base / _file_for(uid, pos, "sampling")
        samp_bak = pathlib.Path(str(samp_path) + ".bak")

        if not ang_bak.exists() or not samp_bak.exists():
            print(f"{uid}/{pos}: required .bak files not found, skipping")
            continue

        cur_ts = int(float(pd.read_csv(samp_path)["beginning"].iloc[0]))
        bak_ts = int(float(pd.read_csv(samp_bak)["beginning"].iloc[0]))
        offset = cur_ts - bak_ts
        print(f"{uid}/{pos}: restoring angular_speed from .bak with offset {offset:+d} ms")

        rows, fieldnames = read_csv_rows(ang_bak)
        changed = 0
        for row in rows:
            raw_val = (row.get("timestamp") or "").strip()
            if not raw_val:
                continue
            row["timestamp"] = str(int(float(raw_val)) + offset)
            changed += 1

        if changed:
            if backup:
                trunc_bak = pathlib.Path(str(ang_path) + ".truncated.bak")
                if not trunc_bak.exists() and not dry_run and ang_path.exists():
                    shutil.copy2(ang_path, trunc_bak)
            if write_csv_rows(ang_path, rows, fieldnames, dry_run):
                files_changed += 1
            values_changed += changed
            print(f"  {ang_path.name}: restored {len(rows)} row(s), shifted {changed} timestamp value(s)")

    return files_changed, values_changed


def run_all(dataset_root: pathlib.Path, raw_root: pathlib.Path, dry_run: bool, backup: bool) -> None:
    print("== Step 1: rename labels ==")
    r = rename_labels(dataset_root, dry_run=dry_run)
    print(
        f"Summary: csv_rows={r.csv_rows}, ods_cells={r.ods_cells}, files_written={r.files_written}\n"
    )

    print("== Step 2: patch wrong-year timestamps ==")
    f, v = patch_wrong_year(raw_root, dry_run=dry_run, backup=backup)
    print(f"Summary: files_changed={f}, values_changed={v}\n")

    print("== Step 3: row-level special fixes ==")
    f, v = row_level_fix(raw_root, dry_run=dry_run, backup=backup)
    print(f"Summary: files_changed={f}, values_changed={v}\n")

    print("== Step 4: restore truncated angular speed files ==")
    f, v = restore_truncated(raw_root, dry_run=dry_run, backup=backup)
    print(f"Summary: files_changed={f}, values_changed={v}\n")


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
        help="Do not create .bak/.truncated.bak safety files where applicable.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rename-labels", help="Run label remap fix over sampling CSV and README ODS files")
    sub.add_parser("patch-wrong-year", help="Patch wrong-year LEFT/RIGHT timestamp files")
    sub.add_parser("row-level-fix", help="Apply row-level timestamp correction for known mixed-session files")
    sub.add_parser("restore-truncated", help="Restore known truncated angular_speed files from .bak and reapply offset")
    sub.add_parser("run-all", help="Execute all fixes in the expected order")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = pathlib.Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {dataset_root}")
        return 1

    if args.raw_root:
        raw_root = pathlib.Path(args.raw_root).resolve()
    else:
        try:
            raw_root = resolve_raw_root(dataset_root)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1

    if not raw_root.is_dir():
        print(f"ERROR: raw root does not exist: {raw_root}")
        return 1

    dry_run = not args.apply
    backup = not args.no_backup
    mode = "APPLY" if args.apply else "DRY-RUN"

    print(f"Mode: {mode}")
    print(f"dataset_root: {dataset_root}")
    print(f"raw_root:     {raw_root}")

    if args.command == "rename-labels":
        r = rename_labels(dataset_root, dry_run=dry_run)
        print(f"Summary: csv_rows={r.csv_rows}, ods_cells={r.ods_cells}, files_written={r.files_written}")
        return 0

    if args.command == "patch-wrong-year":
        f, v = patch_wrong_year(raw_root, dry_run=dry_run, backup=backup)
        print(f"Summary: files_changed={f}, values_changed={v}")
        return 0

    if args.command == "row-level-fix":
        f, v = row_level_fix(raw_root, dry_run=dry_run, backup=backup)
        print(f"Summary: files_changed={f}, values_changed={v}")
        return 0

    if args.command == "restore-truncated":
        f, v = restore_truncated(raw_root, dry_run=dry_run, backup=backup)
        print(f"Summary: files_changed={f}, values_changed={v}")
        return 0

    if args.command == "run-all":
        run_all(dataset_root, raw_root, dry_run=dry_run, backup=backup)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
