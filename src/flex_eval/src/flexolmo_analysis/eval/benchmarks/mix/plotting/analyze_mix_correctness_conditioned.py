from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
from flexolmo_analysis.toolkit.plotting.style import dataset_display_name, model_display_name, style_axis_labels, style_axis_title, style_legend


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "routing_light" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_correctness_conditioned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build correctness-conditioned routing summaries from saved routing records.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-names", default="", help="Optional comma-separated model names to include.")
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--run-labels", default="native_full", help="Comma-separated run labels to include.")
    parser.add_argument("--public-expert-idx", type=int, default=0)
    return parser.parse_args()


def parse_csv_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_generation_artifacts(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("answer:", "response:", "final answer:", "svar:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def relaxed_match(prediction: str, reference: str) -> bool:
    pred = normalize_text(strip_generation_artifacts(prediction))
    ref = normalize_text(reference)
    return pred == ref or pred in ref or ref in pred


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(strip_generation_artifacts(prediction)).split()
    ref_tokens = normalize_text(reference).split()
    if pred_tokens == ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    common = sum(min(pred_counts[token], ref_counts.get(token, 0)) for token in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def classify_label(prediction: str) -> str:
    normalized = normalize_text(strip_generation_artifacts(prediction))
    for label in ("yes", "no", "maybe"):
        if normalized.startswith(label):
            return label
    return normalized.split(" ", 1)[0] if normalized else ""


def score_prediction(record: dict[str, Any]) -> dict[str, Any]:
    scoring_mode = str(record.get("scoring_mode", "qa"))
    prediction_text = str(record.get("predicted_output_text", ""))
    reference = str(record.get("reference_answer", ""))
    cleaned_prediction = strip_generation_artifacts(prediction_text)

    if scoring_mode == "classification":
        predicted_label = classify_label(cleaned_prediction)
        reference_label = normalize_text(reference)
        is_correct = predicted_label == reference_label
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "token_f1": 1.0 if is_correct else 0.0,
        }

    relaxed = relaxed_match(cleaned_prediction, reference)
    return {
        "is_correct": relaxed,
        "score": token_f1(cleaned_prediction, reference),
        "token_f1": token_f1(cleaned_prediction, reference),
    }


def discover_model_names(results_root: Path) -> list[str]:
    if not results_root.exists():
        return []
    return sorted(path.name for path in results_root.iterdir() if path.is_dir())


def discover_datasets(results_root: Path, model_names: list[str], run_labels: list[str]) -> list[str]:
    discovered: set[str] = set()
    for model_name in model_names:
        model_dir = results_root / model_name
        if not model_dir.exists():
            continue
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if any((dataset_dir / run_label / "routing_records.jsonl").exists() for run_label in run_labels):
                discovered.add(dataset_dir.name)
    return sorted(discovered)


def load_routing_records(results_root: Path, model_name: str, dataset_name: str, run_label: str) -> list[dict[str, Any]]:
    path = results_root / model_name / dataset_name / run_label / "routing_records.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def summarize_records(
    records: list[dict[str, Any]],
    model_name: str,
    dataset_name: str,
    run_label: str,
    public_expert_idx: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        scoring = score_prediction(record)
        correctness = "correct" if scoring["is_correct"] else "incorrect"
        for layer_summary in record.get("prompt_router_summary_by_layer") or []:
            top1_experts = [int(item[0]) for item in layer_summary.get("top1_expert_distribution", [])] if layer_summary.get("top1_expert_distribution") else []
            public_top1 = bool(top1_experts and top1_experts[0] == public_expert_idx)
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "run_label": run_label,
                    "example_id": str(record.get("example_id", "")),
                    "language": str(record.get("language", "unknown")),
                    "layer": int(layer_summary["layer_idx"]),
                    "correctness": correctness,
                    "is_correct": 1 if scoring["is_correct"] else 0,
                    "score": float(scoring["score"]),
                    "token_f1": float(scoring["token_f1"]),
                    "mean_top1_prob": float(layer_summary["mean_top1_prob"]),
                    "mean_top2_prob": float(layer_summary["mean_top2_prob"]),
                    "mean_top1_top2_margin": float(layer_summary["mean_top1_top2_margin"]),
                    "mean_token_entropy": float(layer_summary["mean_token_entropy"]),
                    "mean_selected_expert_prob_mass": float(layer_summary["mean_selected_expert_prob_mass"]),
                    "public_top1_dominant": 1 if public_top1 else 0,
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_name"]),
            str(row["dataset_name"]),
            str(row["run_label"]),
            int(row["layer"]),
            str(row["correctness"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_name, dataset_name, run_label, layer, correctness), items in sorted(grouped.items()):
        summary_rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "run_label": run_label,
                "layer": layer,
                "correctness": correctness,
                "num_examples": len(items),
                "mean_top1_prob": float(np.mean([float(item["mean_top1_prob"]) for item in items])),
                "mean_top2_prob": float(np.mean([float(item["mean_top2_prob"]) for item in items])),
                "mean_top1_top2_margin": float(np.mean([float(item["mean_top1_top2_margin"]) for item in items])),
                "mean_token_entropy": float(np.mean([float(item["mean_token_entropy"]) for item in items])),
                "mean_selected_expert_prob_mass": float(np.mean([float(item["mean_selected_expert_prob_mass"]) for item in items])),
                "public_top1_rate": float(np.mean([float(item["public_top1_dominant"]) for item in items])),
                "mean_score": float(np.mean([float(item["score"]) for item in items])),
                "mean_token_f1": float(np.mean([float(item["token_f1"]) for item in items])),
            }
        )
    return summary_rows


def plot_metric(summary_rows: list[dict[str, Any]], metric_name: str, title: str, output_path: Path) -> None:
    if not summary_rows:
        return
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["model_name"]), str(row["dataset_name"])), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (model_name, dataset_name), items in sorted(grouped.items()):
        correct = sorted([row for row in items if row["correctness"] == "correct"], key=lambda row: int(row["layer"]))
        incorrect = sorted([row for row in items if row["correctness"] == "incorrect"], key=lambda row: int(row["layer"]))
        if correct:
            ax.plot(
                [int(row["layer"]) for row in correct],
                [float(row[metric_name]) for row in correct],
                marker="o",
                label=f"{model_display_name(model_name)} | {dataset_display_name(dataset_name)} | correct",
            )
        if incorrect:
            ax.plot(
                [int(row["layer"]) for row in incorrect],
                [float(row[metric_name]) for row in incorrect],
                marker="x",
                linestyle="--",
                label=f"{model_display_name(model_name)} | {dataset_display_name(dataset_name)} | incorrect",
            )
    style_axis_labels(ax, "Layer", metric_name.replace("_", " "))
    style_axis_title(ax, title)
    ax.grid(alpha=0.25)
    legend = ax.legend(frameon=False, fontsize=8, ncol=2)
    style_legend(legend)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    output_root = args.output_root
    model_names = parse_csv_list(args.model_names) or discover_model_names(results_root)
    run_labels = parse_csv_list(args.run_labels)
    datasets = parse_csv_list(args.datasets) or discover_datasets(results_root, model_names, run_labels)

    example_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for dataset_name in datasets:
            for run_label in run_labels:
                records = load_routing_records(results_root, model_name, dataset_name, run_label)
                if not records:
                    continue
                example_rows.extend(summarize_records(records, model_name, dataset_name, run_label, args.public_expert_idx))

    summary_rows = aggregate_rows(example_rows)
    write_csv(example_rows, output_root / "correctness_conditioned_examples.csv")
    write_csv(summary_rows, output_root / "correctness_conditioned_summary.csv")
    plot_metric(summary_rows, "mean_top1_top2_margin", "Correct vs Incorrect: Top-1/Top-2 Margin", output_root / "correctness_conditioned_margin.png")
    plot_metric(summary_rows, "mean_token_entropy", "Correct vs Incorrect: Token Entropy", output_root / "correctness_conditioned_entropy.png")
    plot_metric(summary_rows, "public_top1_rate", "Correct vs Incorrect: Public Top-1 Rate", output_root / "correctness_conditioned_public_top1.png")
    print("Wrote correctness-conditioned analysis outputs:")
    print(f"- {output_root / 'correctness_conditioned_examples.csv'}")
    print(f"- {output_root / 'correctness_conditioned_summary.csv'}")
    print(f"- {output_root / 'correctness_conditioned_margin.png'}")
    print(f"- {output_root / 'correctness_conditioned_entropy.png'}")
    print(f"- {output_root / 'correctness_conditioned_public_top1.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
