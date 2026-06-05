from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_contribution" / "a4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "comparisons" / "55b_expert_contribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze weighted expert contribution captures from FlexOlmo/FlexMoRE runs.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-names", default="", help="Optional comma-separated model names to include.")
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include.")
    parser.add_argument("--run-labels", default="native_full", help="Comma-separated run labels to include.")
    parser.add_argument("--public-expert-idx", type=int, default=0)
    return parser.parse_args()


def parse_csv_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


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
            if any((dataset_dir / run_label / "expert_contribution_records.jsonl").exists() for run_label in run_labels):
                discovered.add(dataset_dir.name)
    return sorted(discovered)


def load_records(results_root: Path, model_name: str, dataset_name: str, run_label: str) -> list[dict]:
    path = results_root / model_name / dataset_name / run_label / "expert_contribution_records.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def build_summary_rows(records: list[dict], model_name: str, dataset_name: str, run_label: str, public_expert_idx: int) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        language = str(record.get("language", "unknown"))
        grouped[("all", int(record["layer"]))].append(record)
        grouped[(language, int(record["layer"]))].append(record)

    rows: list[dict] = []
    for (language, layer_idx), layer_rows in sorted(grouped.items()):
        top1_shares = [float(row["top1_contribution_share"]) for row in layer_rows]
        top2_shares = [float(row["top2_contribution_share"]) for row in layer_rows]
        raw_cosines = [float(row.get("top1_top2_raw_output_cosine", 0.0)) for row in layer_rows]
        weighted_cosines = [float(row.get("top1_top2_weighted_output_cosine", 0.0)) for row in layer_rows]
        alignment_ratios = [float(row.get("mixture_alignment_ratio", 0.0)) for row in layer_rows]
        top1_ablation_ratios = [float(row.get("top1_ablation_delta_ratio", 0.0)) for row in layer_rows]
        top2_ablation_ratios = [float(row.get("top2_ablation_delta_ratio", 0.0)) for row in layer_rows]
        public_selected = 0
        public_dominant = 0
        negligible_top2 = 0
        by_expert_weighted: dict[int, list[float]] = defaultdict(list)
        by_expert_share: dict[int, list[float]] = defaultdict(list)

        for row in layer_rows:
            expert_ids = [int(value) for value in row["selected_expert_ids"]]
            weighted_norms = [float(value) for value in row["weighted_expert_output_norms"]]
            shares = [float(value) for value in row["contribution_shares"]]
            if public_expert_idx in expert_ids:
                public_selected += 1
            if expert_ids:
                dominant_idx = int(np.argmax(shares))
                if int(expert_ids[dominant_idx]) == public_expert_idx:
                    public_dominant += 1
            if len(shares) > 1 and shares[1] < 0.1:
                negligible_top2 += 1
            for expert_id, weighted_norm, share in zip(expert_ids, weighted_norms, shares):
                by_expert_weighted[expert_id].append(weighted_norm)
                by_expert_share[expert_id].append(share)

        rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "run_label": run_label,
                "language": language,
                "layer": layer_idx,
                "num_token_rows": len(layer_rows),
                "mean_top1_contribution_share": float(np.mean(top1_shares)) if top1_shares else 0.0,
                "mean_top2_contribution_share": float(np.mean(top2_shares)) if top2_shares else 0.0,
                "mean_top1_top2_raw_output_cosine": float(np.mean(raw_cosines)) if raw_cosines else 0.0,
                "mean_top1_top2_weighted_output_cosine": float(np.mean(weighted_cosines)) if weighted_cosines else 0.0,
                "mean_mixture_alignment_ratio": float(np.mean(alignment_ratios)) if alignment_ratios else 0.0,
                "mean_top1_ablation_delta_ratio": float(np.mean(top1_ablation_ratios)) if top1_ablation_ratios else 0.0,
                "mean_top2_ablation_delta_ratio": float(np.mean(top2_ablation_ratios)) if top2_ablation_ratios else 0.0,
                "public_selected_rate": public_selected / len(layer_rows) if layer_rows else 0.0,
                "public_dominance_rate": public_dominant / len(layer_rows) if layer_rows else 0.0,
                "top2_negligible_rate": negligible_top2 / len(layer_rows) if layer_rows else 0.0,
                "mean_public_weighted_norm": float(np.mean(by_expert_weighted.get(public_expert_idx, [0.0]))),
                "mean_public_contribution_share": float(np.mean(by_expert_share.get(public_expert_idx, [0.0]))),
            }
        )
        for expert_id in sorted(by_expert_weighted):
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "run_label": run_label,
                    "language": language,
                    "layer": layer_idx,
                    "num_token_rows": len(layer_rows),
                    "expert_id": expert_id,
                    "mean_weighted_output_norm": float(np.mean(by_expert_weighted[expert_id])),
                    "mean_contribution_share": float(np.mean(by_expert_share[expert_id])),
                    "record_type": "per_expert",
                }
            )
    return rows


def build_pair_rows(records: list[dict], model_name: str, dataset_name: str, run_label: str) -> list[dict]:
    pair_groups: dict[tuple[str, int, int, int], list[dict]] = defaultdict(list)
    for record in records:
        expert_ids = [int(value) for value in record["selected_expert_ids"]]
        if len(expert_ids) < 2:
            continue
        language = str(record.get("language", "unknown"))
        pair_groups[("all", int(record["layer"]), expert_ids[0], expert_ids[1])].append(record)
        pair_groups[(language, int(record["layer"]), expert_ids[0], expert_ids[1])].append(record)

    rows: list[dict] = []
    for (language, layer_idx, top1_expert, top2_expert), pair_rows in sorted(pair_groups.items()):
        rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "run_label": run_label,
                "language": language,
                "layer": layer_idx,
                "top1_expert": top1_expert,
                "top2_expert": top2_expert,
                "num_token_rows": len(pair_rows),
                "mean_top1_contribution_share": float(np.mean([float(row["top1_contribution_share"]) for row in pair_rows])),
                "mean_top2_contribution_share": float(np.mean([float(row["top2_contribution_share"]) for row in pair_rows])),
                "mean_top1_top2_raw_output_cosine": float(np.mean([float(row.get("top1_top2_raw_output_cosine", 0.0)) for row in pair_rows])),
                "mean_top1_top2_weighted_output_cosine": float(np.mean([float(row.get("top1_top2_weighted_output_cosine", 0.0)) for row in pair_rows])),
            }
        )
    return rows


def plot_contribution_share(summary_rows: list[dict], output_path: Path) -> None:
    primary_rows = [row for row in summary_rows if row.get("record_type") != "per_expert" and row.get("language") == "all"]
    if not primary_rows:
        return
    labels = [f"{row['model_name']}\n{row['dataset_name']}\nL{row['layer']}" for row in primary_rows]
    top1 = [float(row["mean_top1_contribution_share"]) for row in primary_rows]
    top2 = [float(row["mean_top2_contribution_share"]) for row in primary_rows]
    x = np.arange(len(primary_rows))

    plt.figure(figsize=(max(8, len(primary_rows) * 0.6), 4.5))
    plt.bar(x - 0.18, top1, width=0.35, label="Top-1 contribution")
    plt.bar(x + 0.18, top2, width=0.35, label="Top-2 contribution")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Mean contribution share")
    plt.title("Top-1 vs Top-2 Contribution Share")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_public_dominance(summary_rows: list[dict], output_path: Path) -> None:
    primary_rows = [row for row in summary_rows if row.get("record_type") != "per_expert" and row.get("language") == "all"]
    if not primary_rows:
        return
    labels = [f"{row['model_name']}\n{row['dataset_name']}\nL{row['layer']}" for row in primary_rows]
    selected = [float(row["public_selected_rate"]) for row in primary_rows]
    dominant = [float(row["public_dominance_rate"]) for row in primary_rows]
    x = np.arange(len(primary_rows))

    plt.figure(figsize=(max(8, len(primary_rows) * 0.6), 4.5))
    plt.bar(x - 0.18, selected, width=0.35, label="Public selected")
    plt.bar(x + 0.18, dominant, width=0.35, label="Public dominant")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Rate")
    plt.title("Public Selection vs Public Contribution Dominance")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pair_heatmap(pair_rows: list[dict], output_path: Path) -> None:
    pair_rows = [row for row in pair_rows if row.get("language") == "all"]
    if not pair_rows:
        return
    rows_by_pair = {(int(row["top1_expert"]), int(row["top2_expert"])): float(row["mean_top2_contribution_share"]) for row in pair_rows}
    expert_ids = sorted({int(row["top1_expert"]) for row in pair_rows} | {int(row["top2_expert"]) for row in pair_rows})
    matrix = np.full((len(expert_ids), len(expert_ids)), np.nan, dtype=np.float32)
    index = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
    for (top1_expert, top2_expert), value in rows_by_pair.items():
        matrix[index[top1_expert], index[top2_expert]] = value

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.xticks(np.arange(len(expert_ids)), expert_ids)
    plt.yticks(np.arange(len(expert_ids)), expert_ids)
    plt.xlabel("Top-2 expert")
    plt.ylabel("Top-1 expert")
    plt.title("Mean Top-2 Contribution Share by Expert Pair")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pair_cosine_heatmap(pair_rows: list[dict], output_path: Path) -> None:
    pair_rows = [row for row in pair_rows if row.get("language") == "all"]
    if not pair_rows:
        return
    rows_by_pair = {
        (int(row["top1_expert"]), int(row["top2_expert"])): float(row["mean_top1_top2_weighted_output_cosine"])
        for row in pair_rows
    }
    expert_ids = sorted({int(row["top1_expert"]) for row in pair_rows} | {int(row["top2_expert"]) for row in pair_rows})
    matrix = np.full((len(expert_ids), len(expert_ids)), np.nan, dtype=np.float32)
    index = {expert_id: idx for idx, expert_id in enumerate(expert_ids)}
    for (top1_expert, top2_expert), value in rows_by_pair.items():
        matrix[index[top1_expert], index[top2_expert]] = value

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.xticks(np.arange(len(expert_ids)), expert_ids)
    plt.yticks(np.arange(len(expert_ids)), expert_ids)
    plt.xlabel("Top-2 expert")
    plt.ylabel("Top-1 expert")
    plt.title("Weighted Output Cosine by Expert Pair")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_layerwise_trend(summary_rows: list[dict], output_path: Path) -> None:
    primary_rows = [row for row in summary_rows if row.get("record_type") != "per_expert"]
    if not primary_rows:
        return
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in primary_rows:
        grouped[(str(row["model_name"]), str(row["dataset_name"]), str(row.get("language", "all")))].append(row)

    plt.figure(figsize=(10, 5))
    for (model_name, dataset_name, language), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["layer"]))
        plt.plot(
            [int(row["layer"]) for row in ordered],
            [float(row["mean_top2_contribution_share"]) for row in ordered],
            marker="o",
            label=f"{model_name} | {dataset_name} | {language}",
        )
    plt.xlabel("Layer")
    plt.ylabel("Mean top-2 contribution share")
    plt.title("Layerwise Expert Contribution Trend")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cosine_layerwise_trend(summary_rows: list[dict], output_path: Path) -> None:
    primary_rows = [row for row in summary_rows if row.get("record_type") != "per_expert"]
    if not primary_rows:
        return
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in primary_rows:
        grouped[(str(row["model_name"]), str(row["dataset_name"]), str(row.get("language", "all")))].append(row)

    plt.figure(figsize=(10, 5))
    for (model_name, dataset_name, language), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["layer"]))
        plt.plot(
            [int(row["layer"]) for row in ordered],
            [float(row["mean_top1_top2_weighted_output_cosine"]) for row in ordered],
            marker="o",
            label=f"{model_name} | {dataset_name} | {language}",
        )
    plt.xlabel("Layer")
    plt.ylabel("Mean weighted output cosine")
    plt.title("Layerwise Expert Redundancy / Complementarity")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_contribution_by_expert(summary_rows: list[dict], output_path: Path) -> None:
    expert_rows = [row for row in summary_rows if row.get("record_type") == "per_expert" and row.get("language") == "all"]
    if not expert_rows:
        return
    labels = [
        f"{row['model_name']}\n{row['dataset_name']}\nL{row['layer']}\nE{row['expert_id']}"
        for row in expert_rows
    ]
    values = [float(row["mean_contribution_share"]) for row in expert_rows]
    x = np.arange(len(expert_rows))
    plt.figure(figsize=(max(8, len(expert_rows) * 0.25), 4.5))
    plt.bar(x, values, color="#5B6C8F")
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Mean contribution share")
    plt.title("Contribution Share by Expert")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    model_names = parse_csv_list(args.model_names) or discover_model_names(results_root)
    run_labels = parse_csv_list(args.run_labels)
    datasets = parse_csv_list(args.datasets) or discover_datasets(results_root, model_names, run_labels)

    summary_rows: list[dict] = []
    pair_rows: list[dict] = []
    for model_name in model_names:
        for dataset_name in datasets:
            for run_label in run_labels:
                records = load_records(results_root, model_name, dataset_name, run_label)
                if not records:
                    continue
                summary_rows.extend(build_summary_rows(records, model_name, dataset_name, run_label, args.public_expert_idx))
                pair_rows.extend(build_pair_rows(records, model_name, dataset_name, run_label))

    write_csv(summary_rows, output_root / "expert_contribution_summary.csv")
    write_csv(pair_rows, output_root / "expert_contribution_pair_summary.csv")
    plot_contribution_share(summary_rows, output_root / "expert_contribution_top1_vs_top2.png")
    plot_public_dominance(summary_rows, output_root / "expert_contribution_public_dominance.png")
    plot_pair_heatmap(pair_rows, output_root / "expert_contribution_pair_heatmap.png")
    plot_pair_cosine_heatmap(pair_rows, output_root / "expert_contribution_pair_cosine_heatmap.png")
    plot_layerwise_trend(summary_rows, output_root / "expert_contribution_layerwise_trend.png")
    plot_cosine_layerwise_trend(summary_rows, output_root / "expert_contribution_cosine_layerwise_trend.png")
    plot_contribution_by_expert(summary_rows, output_root / "expert_contribution_by_expert.png")
    print("Wrote expert contribution analysis outputs:")
    print(f"- {output_root / 'expert_contribution_summary.csv'}")
    print(f"- {output_root / 'expert_contribution_pair_summary.csv'}")
    print(f"- {output_root / 'expert_contribution_top1_vs_top2.png'}")
    print(f"- {output_root / 'expert_contribution_public_dominance.png'}")
    print(f"- {output_root / 'expert_contribution_pair_heatmap.png'}")
    print(f"- {output_root / 'expert_contribution_pair_cosine_heatmap.png'}")
    print(f"- {output_root / 'expert_contribution_layerwise_trend.png'}")
    print(f"- {output_root / 'expert_contribution_cosine_layerwise_trend.png'}")
    print(f"- {output_root / 'expert_contribution_by_expert.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
