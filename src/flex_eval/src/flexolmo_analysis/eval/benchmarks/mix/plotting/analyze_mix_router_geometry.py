from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "router_direction" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "router_geometry"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flexolmo_analysis.toolkit.core import (
    cosine_similarity_matrix,
    normalized_confusion_matrix,
    summarize_alignment_records,
    summarize_router_weight_matrix,
)
from flexolmo_analysis.toolkit.plotting.style import (
    FONT_SIZES,
    FONT_WEIGHTS,
    add_shared_ylabel,
    add_top_right_colorbar,
    apply_axis_text_style,
    compose_panel_title,
    dataset_display_name,
    expected_num_experts_for_model,
    expert_tick_labels_for_model,
    model_display_name,
    style_axis_labels,
    style_axis_title,
    style_heatmap_ticklabels,
    style_legend,
    style_suptitle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze router geometry and expert separability from router-direction artifacts."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model-names",
        help="Optional comma-separated model names. Defaults to all model folders in the results root.",
    )
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    return parser.parse_args()


def parse_csv_arg(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    return values or None


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_model_names(results_root: Path) -> list[str]:
    return sorted(path.name for path in results_root.iterdir() if path.is_dir())


def load_router_weights(results_root: Path, model_name: str) -> dict[int, np.ndarray]:
    npz_path = results_root / model_name / "router_weights.npz"
    data = np.load(npz_path)
    weights: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("layer_") and key.endswith("_weights"):
            layer_idx = int(key[len("layer_") : -len("_weights")])
            weights[layer_idx] = np.asarray(data[key], dtype=np.float32)
    return weights


def load_records(results_root: Path, model_name: str, datasets: list[str] | None) -> list[dict]:
    manifest_path = results_root / model_name / "router_direction_suite_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entries = manifest.get("datasets", {})
    selected = datasets or sorted(dataset_entries)
    records: list[dict] = []
    for dataset_name in selected:
        dataset_manifest = dataset_entries.get(dataset_name)
        if not dataset_manifest:
            continue
        records_path = Path(dataset_manifest["records_path"])
        if records_path.exists():
            records.extend(load_jsonl(records_path))
    return records


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_weight_rows(weights_by_model: dict[str, dict[int, np.ndarray]]) -> list[dict]:
    rows: list[dict] = []
    for model_name, layer_map in sorted(weights_by_model.items()):
        for layer_idx, matrix in sorted(layer_map.items()):
            row = {
                "model_name": model_name,
                "model_display_name": model_display_name(model_name),
                "layer": layer_idx,
            }
            row.update(summarize_router_weight_matrix(matrix))
            rows.append(row)
    return rows


def build_alignment_rows(records_by_model: dict[str, list[dict]]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for model_name, records in records_by_model.items():
        for record in records:
            key = (model_name, str(record["dataset_name"]), int(record["layer"]))
            grouped[key].append(record)

    rows: list[dict] = []
    for (model_name, dataset_name, layer_idx), items in sorted(grouped.items()):
        num_experts = expected_num_experts_for_model(model_name) or (
            max(
                max(int(item.get("actual_top1_expert", -1)), int(item.get("top1_aligned_expert", -1)))
                for item in items
            )
            + 1
        )
        row = {
            "model_name": model_name,
            "model_display_name": model_display_name(model_name),
            "dataset_name": dataset_name,
            "dataset_display_name": dataset_display_name(dataset_name),
            "layer": layer_idx,
            "num_experts": num_experts,
        }
        row.update(summarize_alignment_records(items, num_experts=num_experts))
        rows.append(row)
    return rows


def plot_router_weight_cosines(
    weights_by_model: dict[str, dict[int, np.ndarray]],
    model_names: list[str],
    output_path: Path,
) -> None:
    layer_ids = sorted(set.intersection(*(set(weights_by_model[name]) for name in model_names)))
    if not layer_ids:
        return
    fig, axes = plt.subplots(
        len(layer_ids),
        len(model_names),
        figsize=(5.2 * len(model_names), 4.6 * len(layer_ids)),
        squeeze=False,
    )
    image = None
    for row_idx, layer_idx in enumerate(layer_ids):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            matrix = cosine_similarity_matrix(weights_by_model[model_name][layer_idx])
            expected_num_experts = expected_num_experts_for_model(model_name)
            if expected_num_experts is not None:
                matrix = matrix[:expected_num_experts, :expected_num_experts]
            image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            labels = expert_tick_labels_for_model(model_name, matrix.shape[0], multiline=True)
            style_axis_title(ax, compose_panel_title(model_name, prefix=f"L{layer_idx}"))
            style_heatmap_ticklabels(ax, labels, labels)
            ax.yaxis.set_ticks_position("left")
            ax.yaxis.tick_left()
            ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
            ax.set_ylabel("")
            if row_idx < len(layer_ids) - 1:
                ax.set_xlabel("")
    add_shared_ylabel(fig, axes, "Expert", x=axes[0, 0].get_position().x0 - 0.02)
    if image is not None:
        add_top_right_colorbar(fig, axes, image, "Router weight cosine")
    fig.supxlabel("Expert", fontsize=FONT_SIZES["axis_label"], fontweight=FONT_WEIGHTS["label"], y=0.04)
    style_suptitle(fig, "Router Weight Geometry")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.08, wspace=0.16, hspace=0.30)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_alignment_metrics(alignment_rows: list[dict], model_names: list[str], output_path: Path) -> None:
    datasets = sorted({row["dataset_name"] for row in alignment_rows})
    if not datasets:
        return
    metrics = [
        ("top1_agreement_rate", "Top-1 Agreement"),
        ("mean_alignment_margin", "Alignment Margin"),
        ("mean_alignment_entropy", "Alignment Entropy"),
    ]
    fig, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(5.3 * len(metrics), 3.8 * len(datasets)),
        squeeze=False,
    )
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]
    color_map = {model_name: colors[idx % len(colors)] for idx, model_name in enumerate(model_names)}
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for model_name in model_names:
                dataset_rows = [
                    row for row in alignment_rows
                    if row["dataset_name"] == dataset_name and row["model_name"] == model_name
                ]
                dataset_rows = sorted(dataset_rows, key=lambda row: int(row["layer"]))
                if not dataset_rows:
                    continue
                ax.plot(
                    [int(row["layer"]) for row in dataset_rows],
                    [float(row[metric_key]) for row in dataset_rows],
                    marker="o",
                    color=color_map[model_name],
                    label=model_display_name(model_name),
                )
            style_axis_title(ax, f"{dataset_display_name(dataset_name)} | {title}")
            style_axis_labels(ax, "Layer", "")
            ax.grid(alpha=0.25)
            apply_axis_text_style(ax)
            if metric_key == "top1_agreement_rate":
                ax.set_ylim(-0.05, 1.05)
    legend = axes[0, 0].legend(frameon=False, loc="best")
    style_legend(legend)
    style_suptitle(fig, "Router Alignment by Layer")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.20, hspace=0.38)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_assignment_confusions(
    records_by_model: dict[str, list[dict]],
    model_names: list[str],
    datasets: list[str],
    output_path: Path,
) -> None:
    if not datasets:
        return
    fig, axes = plt.subplots(
        len(datasets),
        len(model_names),
        figsize=(5.2 * len(model_names), 4.5 * len(datasets)),
        squeeze=False,
    )
    image = None
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            subset = [
                record for record in records_by_model[model_name]
                if str(record["dataset_name"]) == dataset_name
            ]
            num_experts = expected_num_experts_for_model(model_name) or (
                max(
                    [max(int(record.get("actual_top1_expert", -1)), int(record.get("top1_aligned_expert", -1))) for record in subset],
                    default=-1,
                )
                + 1
            )
            matrix = normalized_confusion_matrix(subset, num_experts=max(num_experts, 1))
            image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
            labels = expert_tick_labels_for_model(model_name, matrix.shape[0], multiline=True)
            style_axis_title(ax, compose_panel_title(model_name, dataset_name))
            style_heatmap_ticklabels(ax, labels, labels)
            style_axis_labels(ax, "Aligned expert", "")
            ax.yaxis.set_ticks_position("left")
            ax.yaxis.tick_left()
            ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
            ax.set_ylabel("")
    add_shared_ylabel(fig, axes, "Actual routed expert", x=axes[0, 0].get_position().x0 - 0.02)
    if image is not None:
        add_top_right_colorbar(fig, axes, image, "Row-normalized frequency")
    style_suptitle(fig, "Router Assignment Confusion")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.08, wspace=0.16, hspace=0.34)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_csv_arg(args.model_names) or discover_model_names(results_root)
    datasets = parse_csv_arg(args.datasets)

    weights_by_model = {model_name: load_router_weights(results_root, model_name) for model_name in model_names}
    records_by_model = {model_name: load_records(results_root, model_name, datasets) for model_name in model_names}

    weight_rows = build_weight_rows(weights_by_model)
    alignment_rows = build_alignment_rows(records_by_model)
    write_csv(weight_rows, output_root / "router_weight_geometry_summary.csv")
    write_csv(alignment_rows, output_root / "router_alignment_separability_summary.csv")

    plot_router_weight_cosines(
        weights_by_model=weights_by_model,
        model_names=model_names,
        output_path=output_root / "router_weight_cosine_heatmaps.png",
    )
    plot_alignment_metrics(
        alignment_rows=alignment_rows,
        model_names=model_names,
        output_path=output_root / "router_alignment_metrics_by_layer.png",
    )
    dataset_names = datasets or sorted({row["dataset_name"] for row in alignment_rows})
    plot_assignment_confusions(
        records_by_model=records_by_model,
        model_names=model_names,
        datasets=dataset_names,
        output_path=output_root / "router_assignment_confusion_heatmaps.png",
    )
    print(f"Wrote router geometry outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
