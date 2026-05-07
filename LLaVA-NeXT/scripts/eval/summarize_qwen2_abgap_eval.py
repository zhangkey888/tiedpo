#!/usr/bin/env python3
import json
import sys
from pathlib import Path


DISPLAY_ORDER = [
    "margin/mean_abs",
    "margin/tie_mean_abs",
    "margin/strict_mean_abs",
    "margin/strict_mean_signed",
    "data/n_samples",
    "data/n_strict",
    "data/n_tie",
    "data/n_ep_tie",
    "thr/0.1/OverallAcc",
    "thr/0.1/TieAcc",
    "thr/0.1/TiePrecision",
    "thr/0.1/TieRecall",
    "thr/0.1/StrictAcc",
    "thr/0.1/StrictPrecision",
    "thr/0.1/StrictRecall",
    "thr/0.2/OverallAcc",
    "thr/0.2/TieAcc",
    "thr/0.2/TiePrecision",
    "thr/0.2/TieRecall",
    "thr/0.2/StrictAcc",
    "thr/0.2/StrictPrecision",
    "thr/0.2/StrictRecall",
    "thr/0.5/OverallAcc",
    "thr/0.5/TieAcc",
    "thr/0.5/TiePrecision",
    "thr/0.5/TieRecall",
    "thr/0.5/StrictAcc",
    "thr/0.5/StrictPrecision",
    "thr/0.5/StrictRecall",
    "thr/0.8/OverallAcc",
    "thr/0.8/TieAcc",
    "thr/0.8/TiePrecision",
    "thr/0.8/TieRecall",
    "thr/0.8/StrictAcc",
    "thr/0.8/StrictPrecision",
    "thr/0.8/StrictRecall",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics, key):
    if "metrics" in metrics and key in metrics["metrics"]:
        return metrics["metrics"][key]
    return metrics.get(key)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_qwen2_abgap_eval.py <group_dir>")

    group_dir = Path(sys.argv[1]).resolve()
    json_paths = sorted(group_dir.glob("*/eval_results.json"))
    if not json_paths:
        raise SystemExit(f"No eval_results.json found under {group_dir}")

    rows = {}
    for json_path in json_paths:
        rows[json_path.parent.name] = load_json(json_path)

    summary = {"group_dir": str(group_dir), "models": {}}
    for model_name, metrics in rows.items():
        summary["models"][model_name] = {key: get_metric(metrics, key) for key in DISPLAY_ORDER}

    summary_json = group_dir / "summary.json"
    summary_md = group_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    headers = ["Metric"] + list(rows.keys())
    lines = [
        "# Qwen2 AB Gap Eval Summary",
        "",
        f"Group dir: `{group_dir}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key in DISPLAY_ORDER:
        row = [key]
        for model_name in rows.keys():
            row.append(format_value(summary["models"][model_name][key]))
        lines.append("| " + " | ".join(row) + " |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
