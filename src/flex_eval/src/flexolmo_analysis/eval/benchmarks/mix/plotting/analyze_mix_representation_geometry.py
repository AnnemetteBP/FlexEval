from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "latent_space" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "representation_geometry"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flexolmo_analysis.toolkit.core import effective_rank, stable_rank
from flexolmo_analysis.toolkit.plotting.style import (
    apply_axis_text_style,
    dataset_display_name,
    model_display_name,
    style_axis_labels,
    style_axis_title,
    style_legend,
    style_suptitle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze embedding / hidden-state / pre-router separability from latent-space artifacts."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-names", help="Optional comma-separated model names.")
    parser.add_argument("--datasets", help="Optional comma-separated dataset names.")
    parser.add_argument(
        "--representation-sources",
        default="embedding,hidden_state,pre_router,router_probs,pre_router_plus_router_probs,hidden_state_plus_router_probs",
        help="Comma-separated representation sources to include.",
    )
    parser.add_argument("--representation", default="last", choices=("mean", "last"))
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


def parse_layer_keys(npz_data) -> dict[str, dict[str, list[int]]]:
    layers_by_source: dict[str, dict[str, list[int]]] = {}
    for key in npz_data.files:
        if "_layer_" not in key:
            continue
        source_name, remainder = key.split("_layer_", 1)
        if "_" not in remainder:
            continue
        layer_str, repr_name = remainder.split("_", 1)
        layers_by_source.setdefault(source_name, {}).setdefault(repr_name, []).append(int(layer_str))
    for source_name in list(layers_by_source):
        for repr_name in list(layers_by_source[source_name]):
            layers_by_source[source_name][repr_name] = sorted(set(layers_by_source[source_name][repr_name]))
    return layers_by_source


def build_language_groups(metadata: list[dict]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(metadata):
        groups.setdefault(str(row.get("language", "unknown")), []).append(idx)
    return {language: np.asarray(indices, dtype=int) for language, indices in groups.items()}


def mean_squared_radius(vectors: np.ndarray) -> float:
    if vectors.shape[0] == 0:
        return 0.0
    centroid = vectors.mean(axis=0, keepdims=True)
    distances = np.sum((vectors - centroid) ** 2, axis=1)
    return float(distances.mean())


def centroid_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.linalg.norm(vec_a.mean(axis=0) - vec_b.mean(axis=0)))


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


def load_dataset_bundle(results_root: Path, model_name: str, dataset_name: str) -> dict | None:
    dataset_dir = results_root / model_name / dataset_name
    npz_path = dataset_dir / "prompt_latents.npz"
    metadata_path = dataset_dir / "metadata.jsonl"
    if not npz_path.exists() or not metadata_path.exists():
        return None
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "npz": np.load(npz_path),
        "metadata": load_jsonl(metadata_path),
    }


def summarize_bundles(
    bundles: list[dict],
    representation_sources: list[str],
    representation_name: str,
) -> list[dict]:
    rows: list[dict] = []
    for bundle in bundles:
        npz_data = bundle["npz"]
        metadata = bundle["metadata"]
        model_name = bundle["model_name"]
        dataset_name = bundle["dataset_name"]
        language_groups = build_language_groups(metadata)
        available = parse_layer_keys(npz_data)

        for source_name in representation_sources:
            for layer_idx in available.get(source_name, {}).get(representation_name, []):
                key = f"{source_name}_layer_{layer_idx}_{representation_name}"
                vectors = np.asarray(npz_data[key], dtype=np.float32)
                base_row = {
                    "model_name": model_name,
                    "model_display_name": model_display_name(model_name),
                    "dataset_name": dataset_name,
                    "dataset_display_name": dataset_display_name(dataset_name),
                    "representation_source": source_name,
                    "representation": representation_name,
                    "layer": layer_idx,
                    "num_examples": int(vectors.shape[0]),
                    "hidden_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
                    "effective_rank": effective_rank(vectors),
                    "stable_rank": stable_rank(vectors),
                    "mean_radius": mean_squared_radius(vectors),
                }
                if len(language_groups) >= 2:
                    languages = sorted(language_groups)[:2]
                    vec_a = vectors[language_groups[languages[0]]]
                    vec_b = vectors[language_groups[languages[1]]]
                    centroid_dist = centroid_distance(vec_a, vec_b)
                    within = 0.5 * (mean_squared_radius(vec_a) + mean_squared_radius(vec_b))
                    base_row["comparison_type"] = "language_pair"
                    base_row["group_a"] = languages[0]
                    base_row["group_b"] = languages[1]
                    base_row["centroid_distance"] = centroid_dist
                    base_row["separation_ratio"] = float(centroid_dist / np.sqrt(max(within, 1e-9)))
                else:
                    base_row["comparison_type"] = "overall"
                    base_row["group_a"] = "all"
                    base_row["group_b"] = ""
                    base_row["centroid_distance"] = 0.0
                    base_row["separation_ratio"] = 0.0
                rows.append(base_row)
    return rows


def plot_representation_geometry(rows: list[dict], model_names: list[str], output_path: Path) -> None:
    if not rows:
        return
    source_order = (
        "embedding",
        "hidden_state",
        "pre_router",
        "router_probs",
        "hidden_state_plus_router_probs",
        "pre_router_plus_router_probs",
    )
    sources = [source for source in source_order if any(row["representation_source"] == source for row in rows)]
    metrics = [
        ("effective_rank", "Effective Rank"),
        ("mean_radius", "Mean Radius"),
        ("separation_ratio", "Separation Ratio"),
    ]
    fig, axes = plt.subplots(len(sources), len(metrics), figsize=(5.2 * len(metrics), 3.8 * len(sources)), squeeze=False)
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]
    color_map = {model_name: colors[idx % len(colors)] for idx, model_name in enumerate(model_names)}
    for row_idx, source_name in enumerate(sources):
        source_rows = [row for row in rows if row["representation_source"] == source_name]
        for col_idx, (metric_key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for model_name in model_names:
                model_rows = [row for row in source_rows if row["model_name"] == model_name]
                model_rows = sorted(model_rows, key=lambda row: (row["dataset_name"], int(row["layer"])))
                if not model_rows:
                    continue
                ax.plot(
                    [int(row["layer"]) for row in model_rows],
                    [float(row[metric_key]) for row in model_rows],
                    marker="o",
                    label=model_display_name(model_name),
                    color=color_map[model_name],
                )
            style_axis_title(ax, f"{source_name.replace('_', ' ').title()} | {title}")
            style_axis_labels(ax, "Layer", "")
            ax.grid(alpha=0.25)
            apply_axis_text_style(ax)
    legend = axes[0, 0].legend(frameon=False, loc="best")
    style_legend(legend)
    style_suptitle(fig, "Representation Geometry by Layer")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.20, hspace=0.38)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_csv_arg(args.model_names) or discover_model_names(results_root)
    datasets = parse_csv_arg(args.datasets)
    representation_sources = parse_csv_arg(args.representation_sources) or ["embedding", "hidden_state", "pre_router"]

    bundles: list[dict] = []
    for model_name in model_names:
        model_root = results_root / model_name
        if not model_root.exists():
            continue
        dataset_names = datasets or sorted(path.name for path in model_root.iterdir() if path.is_dir())
        for dataset_name in dataset_names:
            bundle = load_dataset_bundle(results_root, model_name, dataset_name)
            if bundle is not None:
                bundles.append(bundle)

    rows = summarize_bundles(
        bundles=bundles,
        representation_sources=representation_sources,
        representation_name=args.representation,
    )
    write_csv(rows, output_root / "representation_geometry_summary.csv")
    plot_representation_geometry(rows, model_names=model_names, output_path=output_root / "representation_geometry_by_layer.png")
    print(f"Wrote representation geometry outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
