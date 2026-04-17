#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_METRICS = ["f1", "prec", "rec", "tp", "fn", "tn", "fp"]
DEFAULT_MODEL_ORDER = ["LogisticRegression", "MLP", "CNN1D", "LSTM", "GRU", "RF", "SVM", "XGBoost", "CatBoost"]
KIND_ORDER = ["train", "cross_sensor", "fused_missing", "multisensor_ensemble", "multisensor_stacking"]


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


def pretty_model_name(model: str) -> str:
    mapping = {
        "LogisticRegression": "Logistic Regression",
        "CNN1D": "CNN1D",
        "MLP": "MLP",
        "LSTM": "LSTM",
        "GRU": "GRU",
        "RF": "Random Forest",
        "SVM": "SVM",
        "XGBoost": "XGBoost",
        "CatBoost": "CatBoost",
    }
    return mapping.get(model, model)


def metric_header_name(metric: str) -> str:
    mapping = {
        "f1": "f1",
        "prec": "prec",
        "rec": "rec",
        "acc": "acc",
        "roc_auc": "roc\\_auc",
        "pr_auc": "pr\\_auc",
        "tp": "TP",
        "fn": "FN",
        "tn": "TN",
        "fp": "FP",
        "threshold": "threshold",
    }
    return mapping.get(metric, latex_escape(metric))


def scenario_label(name: str) -> str:
    name = str(name)
    if name.endswith("_T") or name.endswith("_F"):
        name = name[:-2]
    return name


def sort_key_from_sensors(name: str) -> tuple[int, str]:
    base = scenario_label(name)
    sensors = [p for p in base.split("_") if p]
    return (-len(sensors), base)


def parse_sdp_prob(name: str) -> Optional[float]:
    m = re.search(r"_SDP([0-9p]+)", str(name))
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def summarize_metric(df: pd.DataFrame, metric: str, decimals: int = 3) -> str:
    metric = metric.lower()
    if metric not in df.columns:
        return "--"
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(vals) == 0:
        return "--"
    if metric in {"tp", "tn", "fp", "fn"}:
        return str(int(round(vals.sum())))
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def variant_suffix(meta: dict) -> str:
    parts = []
    if meta.get("loss") == "unweighted":
        parts.append("unweighted")
    if meta.get("scale"):
        parts.append("scaled")
    if meta.get("no_mag"):
        parts.append("no-mag")
    if meta.get("only_mag"):
        parts.append("only-mag")
    if meta.get("tune_threshold"):
        metric = meta.get("threshold_metric")
        parts.append(f"tuned-threshold{f' ({metric})' if metric else ''}")
    if meta.get("sensor_dropout"):
        p = meta.get("sensor_dropout_p")
        parts.append(f"sensor dropout ({p:g})" if p is not None else "sensor dropout")
    return " | ".join(parts)


def collect_runs(root: Path) -> list[dict]:
    runs = []
    for summary_csv in sorted(root.rglob("summary_metrics.csv")):
        run_dir = summary_csv.parent
        run_config = run_dir / "run_config.json"
        df = pd.read_csv(summary_csv)
        if df.empty:
            continue

        rel = summary_csv.relative_to(root)
        model = rel.parts[0] if len(rel.parts) >= 2 else "UnknownModel"

        cfg = {}
        if run_config.exists():
            try:
                cfg = json.loads(run_config.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        kind = detect_kind_from_outputs(run_dir, cfg, df)
        meta = build_meta_from_outputs(run_dir, cfg, df, kind)

        runs.append({
            "model": model,
            "run_dir": run_dir,
            "summary_csv": summary_csv,
            "config": cfg,
            "df": df,
            "kind": kind,
            "meta": meta,
        })
    return runs


def detect_kind_from_outputs(run_dir: Path, cfg: dict, df: pd.DataFrame) -> str:
    status = {}
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    mode = str(status.get("mode", "")).strip().lower()
    if mode == "train":
        return "train"
    if mode == "cross_sensor":
        return "cross_sensor"
    if mode == "fused_missing":
        return "fused_missing"

    name = run_dir.name
    if name.startswith("cross_sensor_"):
        return "cross_sensor"
    if name.startswith("padded_eval_"):
        return "fused_missing"
    if name.startswith("multisensor_ensemble_"):
        return "multisensor_ensemble"
    if name.startswith("multisensor_stacking_"):
        return "multisensor_stacking"

    cols = {c.lower() for c in df.columns}
    if "condition" in cols and "method" in cols:
        method_col = next(c for c in df.columns if c.lower() == "method")
        methods = set(df[method_col].astype(str).str.lower())
        if "ensemble" in methods:
            return "multisensor_ensemble"
        if "stacking" in methods:
            return "multisensor_stacking"

    return "train"


def build_meta_from_outputs(run_dir: Path, cfg: dict, df: pd.DataFrame, kind: str) -> dict:
    meta = {
        "scenario": cfg.get("scenario"),
        "test_scenario": cfg.get("test_scenario"),
        "loss": cfg.get("loss", "weighted"),
        "scale": bool(cfg.get("scale", False)),
        "sensor_dropout": bool(cfg.get("sensor_dropout", False)),
        "sensor_dropout_p": cfg.get("sensor_dropout_p"),
        "no_mag": bool(cfg.get("no_mag", False)),
        "only_mag": bool(cfg.get("only_mag", False)),
        "tune_threshold": bool(cfg.get("tune_threshold", False)),
        "threshold_metric": cfg.get("threshold_metric"),
        "tag": cfg.get("tag"),
        "run_name": run_dir.name,
    }

    if meta["sensor_dropout_p"] is None:
        meta["sensor_dropout_p"] = parse_sdp_prob(run_dir.name)

    meta["variant_suffix"] = variant_suffix(meta)

    if kind in {"cross_sensor", "fused_missing"}:
        meta["train"] = cfg.get("scenario")
        meta["test"] = cfg.get("test_scenario")

    if kind in {"multisensor_ensemble", "multisensor_stacking"} and not meta["tag"]:
        meta["tag"] = run_dir.name

    return meta


def make_caption(kind: str, model: str, meta: dict) -> str:
    base = pretty_model_name(model)

    if kind == "train":
        caption = f"Individual Generalization - {base}"
    elif kind == "cross_sensor":
        caption = f"Cross Sensor Evaluation - {base}"
    elif kind == "fused_missing":
        caption = f"Missing Sensor Simulation - {base}"
    elif kind == "multisensor_ensemble":
        caption = f"Multisensor Ensemble - {base}"
    elif kind == "multisensor_stacking":
        caption = f"Multisensor Stacking - {base}"
    else:
        caption = f"{kind} - {base}"

    suffix = meta.get("variant_suffix")
    if suffix:
        caption += f" [{suffix}]"
    return caption


def group_key(run: dict) -> tuple:
    meta = run["meta"]
    kind = run["kind"]
    model = run["model"]

    if kind == "train":
        return (
            kind,
            model,
            meta.get("loss"),
            meta.get("scale"),
            meta.get("no_mag"),
            meta.get("only_mag"),
            meta.get("sensor_dropout"),
            meta.get("sensor_dropout_p"),
            meta.get("tune_threshold"),
            meta.get("threshold_metric"),
        )

    if kind == "cross_sensor":
        return (
            kind,
            model,
            meta.get("loss"),
            meta.get("scale"),
            meta.get("no_mag"),
            meta.get("only_mag"),
            meta.get("sensor_dropout"),
            meta.get("sensor_dropout_p"),
            meta.get("tune_threshold"),
            meta.get("threshold_metric"),
        )

    if kind == "fused_missing":
        return (
            kind,
            model,
            meta.get("train"),
            meta.get("loss"),
            meta.get("scale"),
            meta.get("no_mag"),
            meta.get("only_mag"),
            meta.get("sensor_dropout"),
            meta.get("sensor_dropout_p"),
            meta.get("tune_threshold"),
            meta.get("threshold_metric"),
        )

    if kind in {"multisensor_ensemble", "multisensor_stacking"}:
        return (
            kind,
            model,
            meta.get("tag"),
            meta.get("loss"),
            meta.get("scale"),
            meta.get("sensor_dropout"),
            meta.get("sensor_dropout_p"),
            meta.get("tune_threshold"),
            meta.get("threshold_metric"),
        )

    return (kind, model, meta.get("run_name"))


def representative_meta(group_runs: list[dict]) -> dict:
    return dict(group_runs[0]["meta"])


def render_grouped_table(kind: str, model: str, meta: dict, runs: list[dict], metrics: list[str], decimals: int, use_booktabs: bool) -> str:
    if kind == "train":
        first_headers = ["Data"]
        runs = sorted(runs, key=lambda r: sort_key_from_sensors(r["meta"].get("scenario", "")))
        row_labels = [[scenario_label(r["meta"].get("scenario", ""))] for r in runs]
        midrules = {0, 3} if len(runs) >= 7 else ({0} if len(runs) >= 4 else set())

    elif kind == "cross_sensor":
        first_headers = ["Train", "Test"]
        runs = sorted(
            runs,
            key=lambda r: (
                sort_key_from_sensors(r["meta"].get("train", "")),
                sort_key_from_sensors(r["meta"].get("test", "")),
            ),
        )
        row_labels = [[scenario_label(r["meta"].get("train", "")), scenario_label(r["meta"].get("test", ""))] for r in runs]
        midrules = set()
        if len(runs) > 1:
            current_train = scenario_label(runs[0]["meta"].get("train", ""))
            for idx, run in enumerate(runs[:-1]):
                nxt = scenario_label(runs[idx + 1]["meta"].get("train", ""))
                if nxt != current_train:
                    midrules.add(idx)
                    current_train = nxt

    elif kind == "fused_missing":
        first_headers = ["Test"]
        runs = sorted(runs, key=lambda r: sort_key_from_sensors(r["meta"].get("test", "")))
        row_labels = [[scenario_label(r["meta"].get("test", ""))] for r in runs]
        midrules = {1} if len(runs) > 2 else set()

    elif kind in {"multisensor_ensemble", "multisensor_stacking"}:
        cond_rows = []
        for run in runs:
            df = run["df"]
            cols = {c.lower(): c for c in df.columns}
            if "condition" in cols:
                cond_col = cols["condition"]
                for cond in df[cond_col].astype(str).tolist():
                    cond_rows.append((run, [cond]))
            else:
                cond_rows.append((run, [run["meta"].get("tag", run["meta"].get("run_name", "run"))]))
        first_headers = ["Condition"]
        headers = first_headers + [metric_header_name(m) for m in metrics]
        colspec = "l" * len(headers)
        lines = [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{{latex_escape(make_caption(kind, model, meta))}}}",
            r"\scriptsize",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule" if use_booktabs else r"\hline",
            " & ".join(headers) + r" \\",
            r"\midrule" if use_booktabs else r"\hline",
        ]
        for run, keys in cond_rows:
            df = run["df"]
            cols = {c.lower(): c for c in df.columns}
            if "condition" in cols:
                row_df = df[df[cols["condition"]].astype(str) == keys[0]]
            else:
                row_df = df
            vals = [latex_escape(keys[0])] + [summarize_metric(row_df, m, decimals=decimals) for m in metrics]
            lines.append(" & ".join(vals) + r" \\")
        lines += [
            r"\bottomrule" if use_booktabs else r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    else:
        first_headers = ["Data"]
        row_labels = [[meta.get("run_name", "run")]]
        midrules = set()

    headers = first_headers + [metric_header_name(m) for m in metrics]
    colspec = "l" * len(headers)
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{latex_escape(make_caption(kind, model, meta))}}}",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule" if use_booktabs else r"\hline",
        " & ".join(headers) + r" \\",
        r"\midrule" if use_booktabs else r"\hline",
    ]

    for idx, (run, labels) in enumerate(zip(runs, row_labels)):
        vals = [latex_escape(x) for x in labels] + [summarize_metric(run["df"], m, decimals=decimals) for m in metrics]
        lines.append(" & ".join(vals) + r" \\")
        if idx in midrules and idx != len(runs) - 1:
            lines.append(r"\midrule" if use_booktabs else r"\hline")

    lines += [
        r"\bottomrule" if use_booktabs else r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_tables(root: Path, out_path: Path, metrics: list[str], decimals: int, use_booktabs: bool, clearpage_between_groups: bool) -> None:
    runs = collect_runs(root)
    if not runs:
        raise FileNotFoundError(f"No summary_metrics.csv files found under {root}")

    grouped = {}
    for run in runs:
        grouped.setdefault(group_key(run), []).append(run)

    models_present = list(dict.fromkeys(run["model"] for run in runs))
    ordered_models = [m for m in DEFAULT_MODEL_ORDER if m in models_present] + [m for m in models_present if m not in DEFAULT_MODEL_ORDER]

    ordered_group_keys = sorted(
        grouped.keys(),
        key=lambda k: (
            ordered_models.index(k[1]) if k[1] in ordered_models else 999,
            KIND_ORDER.index(k[0]) if k[0] in KIND_ORDER else 999,
            str(k),
        ),
    )

    blocks = []
    last_model = None
    last_kind = None

    for key in ordered_group_keys:
        kind, model = key[0], key[1]
        runs_in_group = grouped[key]
        meta = representative_meta(runs_in_group)

        if clearpage_between_groups and last_model is not None and (model != last_model or kind != last_kind):
            blocks.append(r"\clearpage")
            blocks.append("")

        blocks.append(
            render_grouped_table(
                kind=kind,
                model=model,
                meta=meta,
                runs=runs_in_group,
                metrics=metrics,
                decimals=decimals,
                use_booktabs=use_booktabs,
            )
        )
        blocks.append("")
        last_model = model
        last_kind = kind

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote grouped tables to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-detect experiment outputs and group scenarios into joined LaTeX tables.")
    parser.add_argument("root", type=Path, help="Root output directory, e.g. output/")
    parser.add_argument("--out", type=Path, default=None, help="Output .tex file path. Default: <root>/grouped_auto_tables.tex")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--no-booktabs", action="store_true")
    parser.add_argument("--no-clearpage", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_path = args.out.resolve() if args.out else root / "grouped_auto_tables.tex"

    build_tables(
        root=root,
        out_path=out_path,
        metrics=[m.lower() for m in args.metrics],
        decimals=args.decimals,
        use_booktabs=not args.no_booktabs,
        clearpage_between_groups=not args.no_clearpage,
    )


if __name__ == "__main__":
    main()
