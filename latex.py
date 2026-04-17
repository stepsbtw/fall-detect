#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

DATASETS = [
    "chest_left_right",
    "chest_left",
    "chest_right",
    "left_right",
    "chest",
    "left",
    "right",
]
DATASETS_BY_LEN = sorted(DATASETS, key=len, reverse=True)

REPORT_MODELS = ["LogisticRegression", "MLP", "CNN1D", "XGBoost"]
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "MLP": "MLP",
    "CNN1D": "CNN1D",
    "XGBoost": "XGBoost",
}

SHORT_SENSOR = {"chest": "C", "left": "L", "right": "R"}

EARLY_SCENARIOS = ["chest_left_right", "chest_left", "chest_right", "left_right", "chest", "left", "right"]
CROSS_SENSOR_PAIRS = [
    ("chest", "left"),
    ("chest", "right"),
    ("left", "chest"),
    ("left", "right"),
    ("right", "chest"),
    ("right", "left"),
]
MISSING_TARGETS = ["chest_left", "chest_right", "left_right", "chest", "left", "right"]
ARMED_SENSOR_SCENARIOS = ["chest", "left", "right"]
ABLATION_ROWS = [
    ("acc_gyr", "Accelerometer + Gyroscope"),
    ("acc_magacc", "Accelerometer + Magnitude Acc"),
    ("gyr_maggyr", "Gyroscope + Magnitude Gyr"),
    ("magacc_maggyr", "Magnitude Acc + Magnitude Gyr"),
]

ABLATION_MAP = {
    "": None,
    "none": None,
    "acc_gyr": "acc_gyr",
    "acc_magacc": "acc_magacc",
    "gyr_maggyr": "gyr_maggyr",
    "magacc_maggyr": "magacc_maggyr",
}

PLACEHOLDER_F1 = "-- $\\pm$ --"
PLACEHOLDER_COUNT = "--"


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


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


def normalize_ablation(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    return ABLATION_MAP.get(token, token)


def split_condition_suffix(dataset: str) -> tuple[str, Optional[str]]:
    text = str(dataset or "").strip()
    if text.endswith("_armed"):
        return text[: -len("_armed")], "armed"
    if text.endswith("_unarmed"):
        return text[: -len("_unarmed")], "unarmed"
    return text, None


def parse_dataset_tag(value: Optional[str]) -> dict:
    text = str(value or "").strip()
    if not text:
        return {"dataset": None, "ablation": None, "sensor_dropout": False, "sensor_dropout_p": None}

    dataset = next((d for d in DATASETS_BY_LEN if text.startswith(d)), None)
    if dataset is None:
        return {
            "dataset": None,
            "ablation": normalize_ablation(text),
            "sensor_dropout": False,
            "sensor_dropout_p": None,
        }

    tail = text[len(dataset) :]
    sensor_dropout = False
    sensor_dropout_p = None

    m = re.match(r"^_SDP([0-9]+(?:p[0-9]+)?)?(.*)$", tail)
    if m:
        sensor_dropout = True
        sensor_dropout_p = float(m.group(1).replace("p", ".")) if m.group(1) else 0.5
        tail = m.group(2)

    tail_token = tail.lstrip("_")
    dataset_tag = dataset

    condition_match = re.match(r"^(armed|unarmed)(?:_(.*))?$", tail_token)
    if condition_match:
        dataset_tag = f"{dataset}_{condition_match.group(1)}"
        tail_token = (condition_match.group(2) or "").strip("_")

    ablation = normalize_ablation(tail_token) if tail_token else None
    return {
        "dataset": dataset_tag,
        "ablation": ablation,
        "sensor_dropout": sensor_dropout,
        "sensor_dropout_p": sensor_dropout_p,
    }


def infer_kind(run_name: str, status_mode: Optional[str]) -> str:
    mode = str(status_mode or "").strip().lower()
    if mode in {"train", "cross_sensor", "missing_sensor", "bagging"}:
        return mode
    if mode in {"stacking", "stacking_train"}:
        return "stacking"
    if run_name.startswith("cross_sensor_"):
        return "cross_sensor"
    if run_name.startswith("missing_sensor_"):
        return "missing_sensor"
    if run_name.startswith("bagging_"):
        return "bagging"
    if run_name.startswith("stacking_"):
        return "stacking"
    return "train"


def parse_train_test_from_name(kind: str, run_name: str) -> tuple[Optional[str], Optional[str]]:
    if kind == "train":
        return run_name, run_name
    if kind == "cross_sensor":
        m = re.match(r"^cross_sensor_(.+)_to_(.+)$", run_name)
        return (m.group(1), m.group(2)) if m else (None, None)
    if kind == "missing_sensor":
        m = re.match(r"^missing_sensor_(.+)_on_(.+)$", run_name)
        return (m.group(1), m.group(2)) if m else (None, None)
    if kind == "bagging":
        m = re.match(r"^missing_sensor_bagging_(.+)_on_(.+)$", run_name)
        if m:
            return m.group(1), m.group(2)
        m = re.match(r"^bagging_(.+)$", run_name)
        return (m.group(1), m.group(1)) if m else (None, None)
    if kind == "stacking":
        m = re.match(r"^missing_sensor_stacking_(.+)_on_(.+)$", run_name)
        if m:
            return m.group(1), m.group(2)
        m = re.match(r"^stacking_(.+)$", run_name)
        return (m.group(1), m.group(1)) if m else (None, None)
    return None, None


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_record(model: str, run_dir: Path, cfg: dict, status: dict, df: pd.DataFrame) -> Optional[dict]:
    kind = infer_kind(run_dir.name, status.get("mode"))
    inferred_train, inferred_test = parse_train_test_from_name(kind, run_dir.name)

    train_tag = first_non_empty(
        cfg.get("train_data"),
        cfg.get("training_data"),
        cfg.get("source_dataset"),
        status.get("source_dataset"),
        inferred_train,
        run_dir.name if kind == "train" else None,
    )
    test_tag = first_non_empty(
        cfg.get("test_data"),
        cfg.get("target_dataset"),
        status.get("target_dataset"),
        inferred_test,
    )

    if kind in {"train", "stacking", "bagging"} and test_tag is None:
        test_tag = train_tag

    train_meta = parse_dataset_tag(train_tag)
    test_meta = parse_dataset_tag(test_tag)

    train_dataset = train_meta["dataset"]
    test_dataset = test_meta["dataset"] or train_dataset
    if not train_dataset or not test_dataset:
        return None

    ablation = normalize_ablation(first_non_empty(cfg.get("ablation"), train_meta["ablation"], test_meta["ablation"]))
    sensor_dropout = bool(
        cfg.get("sensor_dropout", False)
        or train_meta["sensor_dropout"]
        or test_meta["sensor_dropout"]
    )
    sensor_dropout_p = first_non_empty(
        cfg.get("sensor_dropout_p"),
        train_meta["sensor_dropout_p"],
        test_meta["sensor_dropout_p"],
    )
    sensor_dropout_p = float(sensor_dropout_p) if sensor_dropout_p is not None else None

    return {
        "model": model,
        "kind": kind,
        "train": train_dataset,
        "test": test_dataset,
        "ablation": ablation,
        "sensor_dropout": sensor_dropout,
        "sensor_dropout_p": sensor_dropout_p,
        "df": df,
    }


def collect_runs(root: Path) -> list[dict]:
    runs = []
    for summary_csv in sorted(root.rglob("summary_metrics.csv")):
        rel = summary_csv.relative_to(root)
        if len(rel.parts) < 3:
            continue
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            continue
        if df.empty:
            continue

        model = rel.parts[0]
        run_dir = summary_csv.parent
        cfg = read_json(run_dir / "run_config.json")
        status = read_json(run_dir / "status.json")
        record = build_record(model, run_dir, cfg, status, df)
        if record is not None:
            runs.append(record)
    return runs


def short_dataset_label(dataset: str) -> str:
    base_dataset, condition = split_condition_suffix(dataset)
    parts = [SHORT_SENSOR[p] for p in base_dataset.split("_") if p in SHORT_SENSOR]
    base_label = "+".join(parts) if parts else base_dataset

    if condition == "armed":
        return f"{base_label} (Armed)"
    if condition == "unarmed":
        return f"{base_label} (Unarmed)"
    return base_label


def train_test_label(train: str, test: str) -> str:
    return f"{short_dataset_label(train)} $\\rightarrow$ {short_dataset_label(test)}"


def summarize_metric(df: pd.DataFrame, metric: str, decimals: int) -> str:
    if metric not in df.columns:
        return PLACEHOLDER_COUNT if metric != "f1" else PLACEHOLDER_F1
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(vals) == 0:
        return PLACEHOLDER_COUNT if metric != "f1" else PLACEHOLDER_F1
    if metric == "f1":
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    return str(int(round(vals.sum())))


def metric_cells(run: Optional[dict], decimals: int) -> list[str]:
    if run is None:
        return [PLACEHOLDER_F1, PLACEHOLDER_COUNT, PLACEHOLDER_COUNT, PLACEHOLDER_COUNT, PLACEHOLDER_COUNT]
    return [
        summarize_metric(run["df"], "f1", decimals),
        summarize_metric(run["df"], "tp", decimals),
        summarize_metric(run["df"], "fn", decimals),
        summarize_metric(run["df"], "tn", decimals),
        summarize_metric(run["df"], "fp", decimals),
    ]


def pick_run(
    runs: list[dict],
    *,
    kind: str,
    model: str,
    train: str,
    test: str,
    ablation: Optional[str] = None,
    sensor_dropout: Optional[bool] = None,
    sensor_dropout_p: Optional[float] = None,
) -> Optional[dict]:
    want_ablation = normalize_ablation(ablation)
    candidates = []

    for run in runs:
        if run["kind"] != kind or run["model"] != model:
            continue
        if run["train"] != train or run["test"] != test:
            continue

        if want_ablation is None and run["ablation"] is not None:
            continue
        if want_ablation is not None and run["ablation"] != want_ablation:
            continue

        if sensor_dropout is not None and run["sensor_dropout"] != sensor_dropout:
            continue

        if sensor_dropout_p is not None:
            run_p = run["sensor_dropout_p"]
            if run_p is None and run["sensor_dropout"]:
                run_p = 0.5
            if run_p is None or abs(run_p - sensor_dropout_p) > 1e-9:
                continue

        candidates.append(run)

    if not candidates:
        return None
    candidates.sort(key=lambda r: len(r["df"]), reverse=True)
    return candidates[0]


def table_rules(use_booktabs: bool) -> tuple[str, str, str]:
    if use_booktabs:
        return r"\toprule", r"\midrule", r"\bottomrule"
    return r"\hline", r"\hline", r"\hline"


def render_table(
    caption: str,
    colspec: str,
    header: str,
    model_rows: list[tuple[str, list[list[str]]]],
    use_booktabs: bool,
) -> str:
    top, mid, bottom = table_rules(use_booktabs)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        top,
        header,
        mid,
    ]

    for model_idx, (model, rows) in enumerate(model_rows):
        label = latex_escape(MODEL_LABELS.get(model, model))
        for row_idx, row in enumerate(rows):
            model_cell = rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}" if row_idx == 0 else ""
            lines.append(f"{model_cell} & {' & '.join(row)} \\\\")
        if model_idx != len(model_rows) - 1:
            lines.append(mid)

    lines += [bottom, r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def build_pair_rows(
    runs: list[dict],
    models: list[str],
    pairs: list[tuple[str, str, str]],
    decimals: int,
    *,
    sensor_dropout: Optional[bool] = None,
    sensor_dropout_p: Optional[float] = None,
    drop_empty_models: bool = False,
) -> list[tuple[str, list[list[str]]]]:
    out = []
    for model in models:
        rows = []
        has_data = False
        for train, test, kind in pairs:
            run = pick_run(
                runs,
                kind=kind,
                model=model,
                train=train,
                test=test,
                ablation=None,
                sensor_dropout=sensor_dropout,
                sensor_dropout_p=sensor_dropout_p,
            )
            if run is not None:
                has_data = True
            rows.append([train_test_label(train, test), *metric_cells(run, decimals)])
        if not drop_empty_models or has_data:
            out.append((model, rows))
    return out


def build_ablation_rows(runs: list[dict], models: list[str], decimals: int) -> list[tuple[str, list[list[str]]]]:
    out = []
    pair = train_test_label("chest", "chest")

    for model in models:
        rows = []
        has_data = False
        for ablation_key, ablation_label in ABLATION_ROWS:
            run = pick_run(
                runs,
                kind="train",
                model=model,
                train="chest",
                test="chest",
                ablation=ablation_key,
                sensor_dropout=False,
            )
            if run is not None:
                has_data = True
            rows.append([pair, latex_escape(ablation_label), *metric_cells(run, decimals)])

        if has_data:
            out.append((model, rows))

    return out


def build_armed_condition_pairs() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    direct_pairs: list[tuple[str, str, str]] = []
    mixed_pairs: list[tuple[str, str, str]] = []

    for dataset in ARMED_SENSOR_SCENARIOS:
        armed = f"{dataset}_armed"
        unarmed = f"{dataset}_unarmed"
        direct_pairs.extend(
            [
                (armed, armed, "train"),
                (unarmed, unarmed, "train"),
                (armed, unarmed, "cross_sensor"),
                (unarmed, armed, "cross_sensor"),
            ]
        )
        mixed_pairs.extend(
            [
                (dataset, armed, "cross_sensor"),
                (dataset, unarmed, "cross_sensor"),
            ]
        )

    return direct_pairs, mixed_pairs


def build_tables(root: Path, out_path: Path, decimals: int, use_booktabs: bool, clearpage_between_groups: bool) -> None:
    runs = collect_runs(root)
    if not runs:
        raise FileNotFoundError(f"No summary_metrics.csv files found under {root}")

    present_models = {run["model"] for run in runs}
    models = [m for m in REPORT_MODELS if m in present_models] or REPORT_MODELS

    sections = []

    early_pairs = [(s, s, "train") for s in EARLY_SCENARIOS]
    sections.append(
        render_table(
            caption="Individual Generalization - Sensor Placement and Early Fusion",
            colspec="llccccc",
            header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
            model_rows=build_pair_rows(runs, models, early_pairs, decimals, sensor_dropout=False),
            use_booktabs=use_booktabs,
        )
    )

    if clearpage_between_groups:
        sections.append(r"\clearpage")

    ablation_rows = build_ablation_rows(runs, models, decimals)
    if ablation_rows:
        sections.append(
            render_table(
                caption="Individual Generalization - Feature Ablation on Early Fusion",
                colspec="lllccccc",
                header=r"Model & Train $\rightarrow$ Test & Ablation & F1 & TP & FN & TN & FP \\",
                model_rows=ablation_rows,
                use_booktabs=use_booktabs,
            )
        )
        if clearpage_between_groups:
            sections.append(r"\clearpage")

    bagging_pairs = [(s, s, "bagging") for s in EARLY_SCENARIOS]
    sections.append(
        render_table(
            caption="Individual Generalization - Late Fusion Ensemble: Bagging",
            colspec="llccccc",
            header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
            model_rows=build_pair_rows(runs, models, bagging_pairs, decimals, sensor_dropout=False),
            use_booktabs=use_booktabs,
        )
    )

    if clearpage_between_groups:
        sections.append(r"\clearpage")

    stacking_pairs = [(s, s, "stacking") for s in EARLY_SCENARIOS]
    sections.append(
        render_table(
            caption="Individual Generalization - Late Fusion Ensemble: Stacking",
            colspec="llccccc",
            header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
            model_rows=build_pair_rows(runs, models, stacking_pairs, decimals, sensor_dropout=False),
            use_booktabs=use_booktabs,
        )
    )

    cross_pairs = [(tr, te, "cross_sensor") for tr, te in CROSS_SENSOR_PAIRS]
    sections.append(
        render_table(
            caption="Cross Sensor Evaluation",
            colspec="llccccc",
            header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
            model_rows=build_pair_rows(runs, models, cross_pairs, decimals, sensor_dropout=False),
            use_booktabs=use_booktabs,
        )
    )

    armed_direct_pairs, armed_mixed_pairs = build_armed_condition_pairs()
    armed_direct_rows = build_pair_rows(
        runs,
        models,
        armed_direct_pairs,
        decimals,
        sensor_dropout=False,
        drop_empty_models=True,
    )
    if armed_direct_rows:
        if clearpage_between_groups:
            sections.append(r"\clearpage")
        sections.append(
            render_table(
                caption="Armed vs Unarmed Condition Generalization",
                colspec="llccccc",
                header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
                model_rows=armed_direct_rows,
                use_booktabs=use_booktabs,
            )
        )

    armed_mixed_rows = build_pair_rows(
        runs,
        models,
        armed_mixed_pairs,
        decimals,
        sensor_dropout=False,
        drop_empty_models=True,
    )
    if armed_mixed_rows:
        if clearpage_between_groups:
            sections.append(r"\clearpage")
        sections.append(
            render_table(
                caption="Mixed Condition Training to Armed/Unarmed Targets",
                colspec="llccccc",
                header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
                model_rows=armed_mixed_rows,
                use_booktabs=use_booktabs,
            )
        )

    if clearpage_between_groups:
        sections.append(r"\clearpage")

    missing_pairs = [("chest_left_right", t, "missing_sensor") for t in MISSING_TARGETS]
    sections.append(
        render_table(
            caption="Missing Sensors for Early Fusion",
            colspec="llccccc",
            header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
            model_rows=build_pair_rows(runs, models, missing_pairs, decimals, sensor_dropout=False),
            use_booktabs=use_booktabs,
        )
    )

    sdp_pairs = [("chest_left_right", "chest_left_right", "train")] + [
        ("chest_left_right", t, "missing_sensor") for t in MISSING_TARGETS
    ]
    sdp_rows = build_pair_rows(
        runs,
        models,
        sdp_pairs,
        decimals,
        sensor_dropout=True,
        sensor_dropout_p=0.5,
        drop_empty_models=True,
    )

    if sdp_rows:
        if clearpage_between_groups:
            sections.append(r"\clearpage")
        sections.append(
            render_table(
                caption="Missing Sensors for Early Fusion with Sensor Dropout (0.5)",
                colspec="llccccc",
                header=r"Model & Train $\rightarrow$ Test & F1 & TP & FN & TN & FP \\",
                model_rows=sdp_rows,
                use_booktabs=use_booktabs,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote grouped tables to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX report tables from experiment outputs.")
    parser.add_argument("root", type=Path, help="Root output directory, e.g. output/")
    parser.add_argument("--out", type=Path, default=None, help="Output .tex path (default: <root>/grouped_auto_tables.tex)")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--no-booktabs", action="store_true")
    parser.add_argument("--no-clearpage", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_path = args.out.resolve() if args.out else root / "grouped_auto_tables.tex"

    build_tables(
        root=root,
        out_path=out_path,
        decimals=args.decimals,
        use_booktabs=not args.no_booktabs,
        clearpage_between_groups=not args.no_clearpage,
    )


if __name__ == "__main__":
    main()
