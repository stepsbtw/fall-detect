#!/usr/bin/env python3
"""Build per-subject tables that compare all model families."""

import argparse
import re
import zipfile
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd

METRIC_COLUMNS = [
    "acc",
    "prec",
    "rec",
    "f1",
    "roc_auc",
    "pr_auc",
    "tp",
    "fp",
    "tn",
    "fn",
    "threshold",
]
CANONICAL_METADATA_COLUMNS = ["subject_id", "gender", "height_cm", "weight_kg", "age", "bmi"]
DEFAULT_RUNS = ["chest", "left", "right", "chest_left_right", "bagging_chest_left_right"]
DEFAULT_FAMILIES = ["CNN1D", "LogisticRegression", "MLP", "XGBoost"]
MODEL_LABELS = {
    "CNN1D": "CNN1D",
    "LogisticRegression": "Logistic Regression",
    "MLP": "MLP",
    "XGBoost": "XGBoost",
}
RAW_ODS_PATTERN = "ID*/ID*_README.ods"
ODF_NAMESPACES = {
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
METADATA_ALIASES = {
    "subject_id": ["subject_id", "subject", "id", "participant_id", "person_id", "user_id", "userid", "group_id"],
    "gender": ["gender", "sex"],
    "height_cm": ["height_cm", "height", "altura"],
    "weight_kg": ["weight_kg", "weight", "massa", "peso"],
    "age": ["age", "idade"],
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def get_first_matching_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    normalized_map = {normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized_map:
            return normalized_map[key]
    return None


def fold_to_subject_id(fold: str) -> Optional[int]:
    match = re.search(r"(\d+)", str(fold))
    if not match:
        return None
    return int(match.group(1))


def parse_decimal_number(text: str) -> Optional[float]:
    if text is None:
        return None
    match = re.search(r"(-?\d+(?:[.,]\d+)?)", str(text))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def normalize_gender(text: str) -> Optional[str]:
    if text is None:
        return None
    token = str(text).strip().lower()
    if not token:
        return None
    if token.startswith("m"):
        return "Male"
    if token.startswith("f"):
        return "Female"
    return str(text).strip()


def parse_height_cm(text: str) -> Optional[float]:
    value = parse_decimal_number(text)
    if value is None:
        return None
    lower = str(text).lower()
    if "cm" in lower or value > 3.0:
        return value
    return value * 100.0


def parse_weight_kg(text: str) -> Optional[float]:
    return parse_decimal_number(text)


def build_metadata_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)

    meta = pd.DataFrame(records)
    for col in ["subject_id", "height_cm", "weight_kg", "age"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    if "subject_id" in meta.columns:
        meta["subject_id"] = meta["subject_id"].astype("Int64")

    if "height_cm" in meta.columns and "weight_kg" in meta.columns:
        valid = (meta["height_cm"] > 0) & (meta["weight_kg"] > 0)
        bmi = pd.Series([pd.NA] * len(meta), dtype="Float64")
        bmi.loc[valid] = meta.loc[valid, "weight_kg"] / ((meta.loc[valid, "height_cm"] / 100.0) ** 2)
        meta["bmi"] = bmi

    for col in CANONICAL_METADATA_COLUMNS:
        if col not in meta.columns:
            meta[col] = pd.NA

    meta = meta[CANONICAL_METADATA_COLUMNS]
    meta = meta.dropna(subset=["subject_id"])
    meta = meta.drop_duplicates(subset=["subject_id"], keep="first")
    meta = meta.sort_values("subject_id").reset_index(drop=True)
    return meta


def parse_ods_metadata(ods_path: Path) -> dict:
    with zipfile.ZipFile(ods_path, "r") as archive:
        if "content.xml" not in archive.namelist():
            return {}
        xml_bytes = archive.read("content.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for p in root.findall(".//text:p", ODF_NAMESPACES):
        text = " ".join("".join(p.itertext()).split())
        if text:
            paragraphs.append(text)

    meta: dict[str, object] = {}
    for text in paragraphs:
        if "subject_id" not in meta:
            m = re.match(r"(?i)^id\s*:\s*(\d+)", text)
            if m:
                meta["subject_id"] = int(m.group(1))
                continue

        if "age" not in meta:
            m = re.match(r"(?i)^age\s*:\s*(.+)$", text)
            if m:
                age_val = parse_decimal_number(m.group(1))
                if age_val is not None:
                    meta["age"] = age_val
                continue

        if "height_cm" not in meta:
            m = re.match(r"(?i)^height\s*:\s*(.+)$", text)
            if m:
                h = parse_height_cm(m.group(1))
                if h is not None:
                    meta["height_cm"] = h
                continue

        if "weight_kg" not in meta:
            m = re.match(r"(?i)^weight\s*:\s*(.+)$", text)
            if m:
                w = parse_weight_kg(m.group(1))
                if w is not None:
                    meta["weight_kg"] = w
                continue

        if "gender" not in meta:
            m = re.match(r"(?i)^gender\s*:\s*(.+)$", text)
            if m:
                g = normalize_gender(m.group(1))
                if g:
                    meta["gender"] = g
                continue

    return meta


def extract_metadata_from_raw(raw_dir: Path) -> pd.DataFrame:
    records = []
    for ods_path in sorted(raw_dir.glob(RAW_ODS_PATTERN)):
        row = parse_ods_metadata(ods_path)

        folder_match = re.search(r"ID(\d+)", ods_path.parent.name, flags=re.IGNORECASE)
        folder_subject = int(folder_match.group(1)) if folder_match else None

        if row.get("subject_id") is None and folder_subject is not None:
            row["subject_id"] = folder_subject

        if folder_subject is not None and row.get("subject_id") is not None:
            if int(row["subject_id"]) != folder_subject:
                row["subject_id"] = folder_subject

        records.append(row)

    return build_metadata_frame(records)


def load_metadata_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)
    meta = pd.read_csv(path)
    if meta.empty:
        return pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)

    selected = {}
    for target, aliases in METADATA_ALIASES.items():
        src = get_first_matching_column(meta, aliases)
        if src is not None:
            selected[target] = meta[src]

    if "subject_id" not in selected:
        return pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)

    return build_metadata_frame(pd.DataFrame(selected).to_dict(orient="records"))


def merge_metadata(raw_meta: pd.DataFrame, csv_meta: pd.DataFrame) -> pd.DataFrame:
    if raw_meta.empty and csv_meta.empty:
        return pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)
    if raw_meta.empty:
        return csv_meta.copy()
    if csv_meta.empty:
        return raw_meta.copy()

    raw_idx = raw_meta.set_index("subject_id")
    csv_idx = csv_meta.set_index("subject_id")
    merged = csv_idx.combine_first(raw_idx).reset_index()

    for col in CANONICAL_METADATA_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[CANONICAL_METADATA_COLUMNS]
    merged = merged.sort_values("subject_id").reset_index(drop=True)
    return merged


def resolve_metadata(metadata_path: Path, raw_dir: Path, metadata_source: str) -> pd.DataFrame:
    source = metadata_source.lower()
    csv_meta = pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)
    raw_meta = pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)

    if source in {"auto", "csv"}:
        csv_meta = load_metadata_csv(metadata_path)
        if not csv_meta.empty:
            print(f"[INFO] Loaded metadata CSV rows: {len(csv_meta)} from {metadata_path}")

    if source in {"auto", "raw"}:
        if raw_dir.exists():
            raw_meta = extract_metadata_from_raw(raw_dir)
            print(f"[INFO] Extracted metadata rows from raw ODS: {len(raw_meta)}")
        else:
            print(f"[WARN] Raw directory not found: {raw_dir}")

    if source == "csv":
        return csv_meta
    if source == "raw":
        return raw_meta
    return merge_metadata(raw_meta=raw_meta, csv_meta=csv_meta)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def format_cell_value(value, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.{decimals}f}"
    text = str(value)
    if re.fullmatch(r"-?\d+", text):
        return text
    try:
        num = float(text)
    except ValueError:
        return text
    if num.is_integer():
        return str(int(num))
    return f"{num:.{decimals}f}"


def table_rules(use_booktabs: bool) -> tuple[str, str, str]:
    if use_booktabs:
        return r"\toprule", r"\midrule", r"\bottomrule"
    return r"\hline", r"\hline", r"\hline"


def discover_families(output_root: Path) -> list[str]:
    families = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "chest").exists() or (child / "chest_left_right").exists():
            families.append(child.name)
    return families


def load_run_summary(summary_path: Path, family: str, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["family"] = family
    out["model"] = MODEL_LABELS.get(family, family)
    out["run"] = run_name
    out["subject_id"] = out["fold"].apply(fold_to_subject_id)
    out = out.dropna(subset=["subject_id"]).copy()
    out["subject_id"] = out["subject_id"].astype("Int64")

    for col in METRIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def collect_all_rows(output_root: Path, families: list[str], runs: list[str]) -> pd.DataFrame:
    rows = []
    for family in families:
        family_dir = output_root / family
        if not family_dir.exists():
            print(f"[WARN] Family directory not found: {family_dir}")
            continue

        for run_name in runs:
            summary_path = family_dir / run_name / "summary_metrics.csv"
            if not summary_path.exists():
                print(f"[WARN] Missing summary for {family}/{run_name}: {summary_path}")
                continue
            part = load_run_summary(summary_path, family, run_name)
            if part.empty:
                print(f"[WARN] Empty summary for {family}/{run_name}: {summary_path}")
                continue
            rows.append(part)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_table(df: pd.DataFrame, columns: list[str], caption: str, use_booktabs: bool, decimals: int) -> str:
    top, mid, bottom = table_rules(use_booktabs)
    colspec = "ll" + "c" * (len(columns) - 2)

    header_labels = {
        "f1": "F1",
        "tp": "TP",
        "fn": "FN",
        "tn": "TN",
        "fp": "FP",
    }
    header = " & ".join(latex_escape(header_labels.get(col, col)) for col in columns) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        top,
        header,
        mid,
    ]

    grouped = df.groupby("model", sort=False, dropna=False)
    total_groups = grouped.ngroups
    for group_idx, (model_name, group_df) in enumerate(grouped):
        group_rows = group_df[columns].reset_index(drop=True)
        group_size = len(group_rows)

        for row_idx, (_, row) in enumerate(group_rows.iterrows()):
            model_cell = rf"\multirow{{{group_size}}}{{*}}{{{latex_escape(format_cell_value(model_name, decimals))}}}" if row_idx == 0 else ""
            other_cells = [latex_escape(format_cell_value(row[col], decimals)) for col in columns[1:]]
            lines.append(" & ".join([model_cell, *other_cells]) + r" \\")

        if group_idx != total_groups - 1:
            lines.append(mid)

    lines += [bottom, r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def metadata_caption_fragment(metadata_df: pd.DataFrame, subject_id: int) -> str:
    row = metadata_df[metadata_df["subject_id"] == subject_id]
    if row.empty:
        return ""

    item = row.iloc[0]
    parts = []
    if pd.notna(item.get("gender")):
        parts.append(str(item["gender"]))
    if pd.notna(item.get("age")):
        parts.append(f"{format_cell_value(item['age'], 0)}y")
    if pd.notna(item.get("height_cm")):
        parts.append(f"{format_cell_value(item['height_cm'], 0)}cm")
    if pd.notna(item.get("weight_kg")):
        parts.append(f"{format_cell_value(item['weight_kg'], 0)}kg")
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"


def build_subject_tables(
    all_rows: pd.DataFrame,
    metadata_df: pd.DataFrame,
    out_dir: Path,
    families: list[str],
    runs: list[str],
    decimals: int,
    use_booktabs: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    family_order = {f: i for i, f in enumerate(families)}
    run_order = {r: i for i, r in enumerate(runs)}

    all_rows = all_rows.copy()
    all_rows["family_order"] = all_rows["family"].map(family_order).fillna(999)
    all_rows["run_order"] = all_rows["run"].map(run_order).fillna(999)

    combined_cols = ["subject_id", "model", "run", "fold"] + [c for c in METRIC_COLUMNS if c in all_rows.columns]
    combined_df = all_rows.sort_values(["subject_id", "family_order", "run_order"])[combined_cols].reset_index(drop=True)

    subjects = sorted(int(s) for s in combined_df["subject_id"].dropna().unique())
    all_subject_tables: list[str] = []
    for subject_id in subjects:
        subject_dir = out_dir / f"subject_s{subject_id}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        sub = combined_df[combined_df["subject_id"] == subject_id].copy()

        tex_cols = ["model", "run", "f1", "tp", "fn", "tn", "fp"]
        tex_cols = [c for c in tex_cols if c in sub.columns]
        caption = f"Subject s{subject_id}{metadata_caption_fragment(metadata_df, subject_id)}: model comparison across selected runs"
        tex_body = render_table(sub, tex_cols, caption, use_booktabs=use_booktabs, decimals=decimals)
        tex_path = subject_dir / "model_run_metrics.tex"
        tex_path.write_text(tex_body + "\n", encoding="utf-8")
        all_subject_tables.append(tex_body)

        print(f"[OK] Subject s{subject_id}: wrote {tex_path}")

    merged_tex = out_dir / "subject_tables.tex"
    merged_tex.write_text(("\n\n\\clearpage\n\n").join(all_subject_tables).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote merged LaTeX file: {merged_tex}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one comparison table per subject across model families.")
    parser.add_argument("output_root", type=Path, help="Output root directory, e.g. output")
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Model families to parse (default: auto-discover, then filter to common defaults)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run directories to include",
    )
    parser.add_argument("--metadata", type=Path, default=Path("dataset/participant_metadata.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--metadata-source", choices=["auto", "csv", "raw"], default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <output_root>/person_tables_by_subject)",
    )
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--no-booktabs", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    if args.families:
        families = args.families
    else:
        found = discover_families(output_root)
        defaults_found = [f for f in DEFAULT_FAMILIES if f in found]
        families = defaults_found if defaults_found else found

    metadata_df = resolve_metadata(
        metadata_path=args.metadata.resolve(),
        raw_dir=args.raw_dir.resolve(),
        metadata_source=args.metadata_source,
    )
    if metadata_df.empty:
        metadata_df = pd.DataFrame(columns=CANONICAL_METADATA_COLUMNS)
        print("[WARN] Metadata could not be resolved; subject info files will be empty.")

    all_rows = collect_all_rows(output_root, families=families, runs=args.runs)
    if all_rows.empty:
        raise RuntimeError("No run metrics were found for the selected families/runs.")

    out_dir = args.out_dir.resolve() if args.out_dir else (output_root / "person_tables_by_subject")
    build_subject_tables(
        all_rows=all_rows,
        metadata_df=metadata_df,
        out_dir=out_dir,
        families=families,
        runs=args.runs,
        decimals=args.decimals,
        use_booktabs=not args.no_booktabs,
    )


if __name__ == "__main__":
    main()
