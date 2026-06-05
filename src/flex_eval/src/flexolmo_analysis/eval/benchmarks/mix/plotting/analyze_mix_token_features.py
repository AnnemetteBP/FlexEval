from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "token_features" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_token_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token-level feature captures with linear probes for expert, language, and fragmentation separability."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--model-names",
        default="FlexOlmo-8x7B-1T-a4-55B-v2,FlexOlmo-8x7B-1T-a4-55B-v2-rt",
        help="Comma-separated model names to compare.",
    )
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--run-labels", default="native_full", help="Comma-separated run labels to include.")
    parser.add_argument("--representation-sources", default="pre_router,hidden_state")
    parser.add_argument("--targets", default="top1_expert,language")
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--max-samples-per-group", type=int, default=6000)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def parse_csv_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_model_names(raw_value: str) -> list[str]:
    names = parse_csv_list(raw_value)
    if not names:
        raise ValueError("Provide at least one model name.")
    return names


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def model_display_name(model_name: str) -> str:
    return model_name.replace("FlexOlmo-8x7B-1T-", "")


def dataset_display_name(dataset_name: str) -> str:
    return {
        "mkqa_en_da": "MGQA (EN/DA)",
        "gsm8k_subset": "GSM8K",
        "mbpp_subset": "MBPP",
        "pubmedqa_subset": "PubMedQA",
        "ag_news_subset": "AG News",
        "common_gen_subset": "CommonGen",
    }.get(dataset_name, dataset_name)


def load_run_bundle(results_root: Path, model_name: str, dataset_name: str, run_label: str) -> dict | None:
    run_dir = results_root / model_name / dataset_name / run_label
    metadata_path = run_dir / "token_feature_metadata.jsonl"
    vectors_path = run_dir / "token_feature_vectors.npz"
    manifest_path = run_dir / "token_feature_manifest.json"
    if not metadata_path.exists() or not vectors_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "metadata": load_jsonl(metadata_path),
        "vectors": np.load(vectors_path),
        "manifest": manifest,
    }


def discover_datasets(results_root: Path, model_names: list[str], run_labels: list[str]) -> list[str]:
    discovered: set[str] = set()
    for model_name in model_names:
        model_dir = results_root / model_name
        if not model_dir.exists():
            continue
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if any((dataset_dir / run_label / "token_feature_metadata.jsonl").exists() for run_label in run_labels):
                discovered.add(dataset_dir.name)
    return sorted(discovered)


def build_layer_source_rows(metadata: list[dict], vectors: np.lib.npyio.NpzFile, source_name: str, layer_idx: int) -> tuple[np.ndarray, list[dict]]:
    key = f"{source_name}_layer_{layer_idx}"
    if key not in vectors.files:
        raise KeyError(f"Missing vector key `{key}`.")
    matrix = np.asarray(vectors[key], dtype=np.float32)
    selected_rows = [row for row in metadata if int(row["layer"]) == layer_idx]
    if matrix.shape[0] != len(selected_rows):
        raise ValueError(
            f"Row mismatch for {key}: vectors have {matrix.shape[0]} rows but metadata has {len(selected_rows)}."
        )
    return matrix, selected_rows


def prepare_target_labels(rows: list[dict], target_name: str, min_class_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    labels = []
    keep_mask = []
    for row in rows:
        value = row.get(target_name)
        if value is None:
            keep_mask.append(False)
            continue
        if target_name == "language" and str(value) == "unknown":
            keep_mask.append(False)
            continue
        labels.append(str(value))
        keep_mask.append(True)
    if not any(keep_mask):
        return None
    labels_array = np.asarray(labels)
    valid_indices = np.asarray([idx for idx, keep in enumerate(keep_mask) if keep], dtype=int)
    unique_values, counts = np.unique(labels_array, return_counts=True)
    valid_values = {value for value, count in zip(unique_values.tolist(), counts.tolist()) if count >= min_class_count}
    if len(valid_values) < 2:
        return None
    filtered_positions = np.asarray([idx for idx, label in enumerate(labels_array) if label in valid_values], dtype=int)
    return valid_indices[filtered_positions], labels_array[filtered_positions]


def maybe_subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=max_samples, replace=False)
    return x[indices], y[indices]


def evaluate_linear_probe(features: np.ndarray, labels: np.ndarray, test_size: float, seed: int) -> dict[str, float] | None:
    unique_values, counts = np.unique(labels, return_counts=True)
    if len(unique_values) < 2 or counts.min() < 2:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="auto"),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    majority_baseline = float(max(np.mean(y_test == value) for value in np.unique(y_test)))
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "baseline_accuracy": majority_baseline,
        "num_train": float(x_train.shape[0]),
        "num_test": float(x_test.shape[0]),
        "num_classes": float(len(np.unique(labels))),
    }


def analyze_bundle(
    model_name: str,
    dataset_name: str,
    run_label: str,
    bundle: dict,
    representation_sources: list[str],
    targets: list[str],
    min_class_count: int,
    max_samples_per_group: int,
    test_size: float,
    seed: int,
) -> list[dict]:
    metadata = bundle["metadata"]
    vectors = bundle["vectors"]
    manifest = bundle["manifest"]
    selected_layers = [int(layer) for layer in manifest.get("selected_layers", [])]
    if not selected_layers:
        selected_layers = sorted({int(row["layer"]) for row in metadata})

    rows: list[dict] = []
    for source_name in representation_sources:
        for layer_idx in selected_layers:
            try:
                layer_vectors, layer_metadata = build_layer_source_rows(metadata, vectors, source_name, layer_idx)
            except KeyError:
                continue
            for target_name in targets:
                prepared = prepare_target_labels(layer_metadata, target_name, min_class_count)
                if prepared is None:
                    continue
                valid_indices, labels = prepared
                filtered_vectors = layer_vectors[valid_indices]
                filtered_vectors, labels = maybe_subsample(filtered_vectors, labels, max_samples_per_group, seed)
                metrics = evaluate_linear_probe(filtered_vectors, labels, test_size=test_size, seed=seed)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "run_label": run_label,
                        "representation_source": source_name,
                        "layer": int(layer_idx),
                        "target": target_name,
                        "grouping": "all_tokens",
                        "group_value": "all",
                        "accuracy": metrics["accuracy"],
                        "baseline_accuracy": metrics["baseline_accuracy"],
                        "accuracy_gain": metrics["accuracy"] - metrics["baseline_accuracy"],
                        "num_train": int(metrics["num_train"]),
                        "num_test": int(metrics["num_test"]),
                        "num_classes": int(metrics["num_classes"]),
                    }
                )

                if target_name not in {"language", "top1_expert"}:
                    continue
                fragment_groups = sorted(
                    {
                        str(row.get("fragmentation_bucket", "unknown"))
                        for row in layer_metadata
                        if str(row.get("fragmentation_bucket", "unknown")) != "unknown"
                    }
                )
                for fragment_group in fragment_groups:
                    group_indices = np.asarray(
                        [
                            idx for idx, row in enumerate(layer_metadata)
                            if str(row.get("fragmentation_bucket", "unknown")) == fragment_group
                        ],
                        dtype=int,
                    )
                    if group_indices.size == 0:
                        continue
                    grouped_vectors = layer_vectors[group_indices]
                    grouped_rows = [layer_metadata[idx] for idx in group_indices.tolist()]
                    grouped_prepared = prepare_target_labels(grouped_rows, target_name, min_class_count)
                    if grouped_prepared is None:
                        continue
                    valid_group_indices, grouped_labels = grouped_prepared
                    grouped_vectors = grouped_vectors[valid_group_indices]
                    grouped_vectors, grouped_labels = maybe_subsample(grouped_vectors, grouped_labels, max_samples_per_group, seed)
                    grouped_metrics = evaluate_linear_probe(grouped_vectors, grouped_labels, test_size=test_size, seed=seed)
                    if grouped_metrics is None:
                        continue
                    rows.append(
                        {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "run_label": run_label,
                            "representation_source": source_name,
                            "layer": int(layer_idx),
                            "target": target_name,
                            "grouping": "fragmentation_bucket",
                            "group_value": fragment_group,
                            "accuracy": grouped_metrics["accuracy"],
                            "baseline_accuracy": grouped_metrics["baseline_accuracy"],
                            "accuracy_gain": grouped_metrics["accuracy"] - grouped_metrics["baseline_accuracy"],
                            "num_train": int(grouped_metrics["num_train"]),
                            "num_test": int(grouped_metrics["num_test"]),
                            "num_classes": int(grouped_metrics["num_classes"]),
                        }
                    )
    return rows


def plot_accuracy_lines(rows: list[dict], model_names: list[str], output_path: Path) -> None:
    filtered = [row for row in rows if row["grouping"] == "all_tokens"]
    if not filtered:
        return
    datasets = sorted({str(row["dataset_name"]) for row in filtered})
    targets = sorted({str(row["target"]) for row in filtered})
    sources = sorted({str(row["representation_source"]) for row in filtered})
    fig, axes = plt.subplots(
        len(datasets),
        max(1, len(targets) * len(sources)),
        figsize=(5 * max(1, len(targets) * len(sources)), 3.8 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {
        model_name: color
        for model_name, color in zip(model_names, ["#5B6C8F", "#C96B3B", "#5B9A6F", "#7E5AA6"])
    }
    for row_idx, dataset_name in enumerate(datasets):
        col_idx = 0
        for target_name in targets:
            for source_name in sources:
                ax = axes[row_idx][col_idx]
                for model_name in model_names:
                    subset = [
                        row for row in filtered
                        if str(row["dataset_name"]) == dataset_name
                        and str(row["target"]) == target_name
                        and str(row["representation_source"]) == source_name
                        and str(row["model_name"]) == model_name
                    ]
                    subset = sorted(subset, key=lambda item: int(item["layer"]))
                    if not subset:
                        continue
                    ax.plot(
                        [int(row["layer"]) for row in subset],
                        [float(row["accuracy_gain"]) for row in subset],
                        marker="o",
                        color=colors.get(model_name, "#444444"),
                        label=model_display_name(model_name),
                    )
                ax.set_title(f"{dataset_display_name(dataset_name)} | {source_name} | {target_name}")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Accuracy gain over baseline")
                ax.grid(alpha=0.25)
                col_idx += 1
    axes[0][0].legend(frameon=False, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fragmentation_bars(rows: list[dict], model_names: list[str], output_path: Path) -> None:
    filtered = [row for row in rows if row["grouping"] == "fragmentation_bucket"]
    if not filtered:
        return
    target_name = "top1_expert" if any(str(row["target"]) == "top1_expert" for row in filtered) else str(filtered[0]["target"])
    filtered = [row for row in filtered if str(row["target"]) == target_name]
    if not filtered:
        return
    datasets = sorted({str(row["dataset_name"]) for row in filtered})
    fragment_groups = sorted({str(row["group_value"]) for row in filtered})
    fig, axes = plt.subplots(
        len(datasets),
        len(model_names),
        figsize=(5.2 * len(model_names), 3.8 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx][col_idx]
            subset = [
                row for row in filtered
                if str(row["dataset_name"]) == dataset_name and str(row["model_name"]) == model_name
            ]
            if not subset:
                ax.axis("off")
                continue
            best_by_group: dict[str, dict] = {}
            for row in subset:
                group_value = str(row["group_value"])
                current = best_by_group.get(group_value)
                if current is None or float(row["accuracy_gain"]) > float(current["accuracy_gain"]):
                    best_by_group[group_value] = row
            values = [float(best_by_group[group]["accuracy_gain"]) if group in best_by_group else np.nan for group in fragment_groups]
            ax.bar(fragment_groups, values, color="#5B6C8F")
            ax.set_title(f"{dataset_display_name(dataset_name)} | {model_display_name(model_name)}")
            ax.set_xlabel("Fragmentation bucket")
            ax.set_ylabel(f"{target_name} gain")
            ax.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_readme(output_path: Path, model_names: list[str], run_labels: list[str]) -> None:
    lines = [
        "# Token Feature Analysis",
        "",
        "Compared models:",
        *[f"- `{model_name}`" for model_name in model_names],
        "",
        "Included run labels:",
        *[f"- `{run_label}`" for run_label in run_labels],
        "",
        "Artifacts:",
        "- `token_feature_probe_summary.csv`",
        "- `token_feature_probe_accuracy_gain.png`",
        "- `token_feature_fragmentation_gain.png`",
        "",
        "Interpretation guide:",
        "- Higher `accuracy_gain` means the token representations carry more linearly recoverable signal than the label baseline alone.",
        "- `pre_router` results show what the router could in principle separate before gating.",
        "- `hidden_state` results show what is recoverable after the routed computation.",
        "- Fragmentation-group results help test whether highly split tokens are harder to classify by language or expert.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_model_names(args.model_names)
    datasets = parse_csv_list(args.datasets)
    run_labels = parse_csv_list(args.run_labels)
    representation_sources = parse_csv_list(args.representation_sources)
    targets = parse_csv_list(args.targets)

    if not datasets:
        datasets = discover_datasets(results_root, model_names, run_labels)
    if not datasets:
        raise ValueError("No token-feature datasets were discovered.")

    summary_rows: list[dict] = []
    for model_name in model_names:
        for dataset_name in datasets:
            for run_label in run_labels:
                bundle = load_run_bundle(results_root, model_name, dataset_name, run_label)
                if bundle is None:
                    continue
                summary_rows.extend(
                    analyze_bundle(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        run_label=run_label,
                        bundle=bundle,
                        representation_sources=representation_sources,
                        targets=targets,
                        min_class_count=args.min_class_count,
                        max_samples_per_group=args.max_samples_per_group,
                        test_size=args.test_size,
                        seed=args.random_seed,
                    )
                )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, output_root / "token_feature_probe_summary.csv")
    plot_accuracy_lines(summary_rows, model_names, output_root / "token_feature_probe_accuracy_gain.png")
    plot_fragmentation_bars(summary_rows, model_names, output_root / "token_feature_fragmentation_gain.png")
    write_readme(output_root / "README.md", model_names, run_labels)
    print(f"Wrote token-feature analysis outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
