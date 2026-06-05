from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

REPORT_COMMANDS = {
    "coactivation": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_coactivation.py"),
        "--results-root",
        "outputs/analysis/captures/routing_light",
        "--output-root",
        "outputs/analysis/figures/coactivation",
    ],
    "top1_top2_confusion": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_top1_top2_confusion.py"),
        "--results-root",
        "outputs/analysis/captures/routing_light",
        "--output-root",
        "outputs/analysis/figures/top1_top2_confusion",
    ],
    "routing_confidence": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_routing_confidence_outcomes.py"),
        "--results-root",
        "outputs/analysis/captures/routing_light",
        "--output-root",
        "outputs/analysis/figures/routing_confidence",
    ],
    "correctness_conditioned": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_correctness_conditioned.py"),
        "--results-root",
        "outputs/analysis/captures/routing_light",
        "--output-root",
        "outputs/analysis/figures/correctness_conditioned",
        "--run-labels",
        "native_full",
    ],
    "router_direction": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_router_direction.py"),
        "--results-root",
        "outputs/analysis/captures/router_direction",
        "--output-root",
        "outputs/analysis/figures/router_direction",
    ],
    "router_geometry": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_router_geometry.py"),
        "--results-root",
        "outputs/analysis/captures/router_direction",
        "--output-root",
        "outputs/analysis/figures/router_geometry",
    ],
    "latent_space": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_latent_space.py"),
        "--results-root",
        "outputs/analysis/captures/latent_space",
        "--output-root",
        "outputs/analysis/figures/latent_space",
    ],
    "representation_geometry": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_representation_geometry.py"),
        "--results-root",
        "outputs/analysis/captures/latent_space",
        "--output-root",
        "outputs/analysis/figures/representation_geometry",
    ],
    "weight_analysis": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_weight_analysis.py"),
        "--results-root",
        "outputs/analysis/captures/weight_analysis",
        "--output-root",
        "outputs/analysis/figures/weight_analysis",
    ],
    "routing_weight_bridge": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_routing_weight_bridge.py"),
        "--routing-root",
        "outputs/analysis/captures/routing_light",
        "--weight-root",
        "outputs/analysis/captures/weight_analysis",
        "--output-root",
        "outputs/analysis/figures/routing_weight_bridge",
        "--run-label",
        "native_full",
    ],
    "token_features": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_token_features.py"),
        "--results-root",
        "outputs/analysis/captures/token_features",
        "--output-root",
        "outputs/analysis/figures/token_features",
        "--run-labels",
        "native_full",
    ],
    "expert_contribution": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "analyze_mix_expert_contribution.py"),
        "--results-root",
        "outputs/analysis/captures/expert_contribution",
        "--output-root",
        "outputs/analysis/figures/expert_contribution",
        "--run-labels",
        "native_full",
    ],
    "summary_tables": [
        sys.executable,
        str(PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "plotting" / "generate_mix_summary_tables.py"),
        "--coactivation-root",
        "outputs/analysis/figures/coactivation",
        "--latent-space-root",
        "outputs/analysis/figures/latent_space",
        "--top1-top2-root",
        "outputs/analysis/figures/top1_top2_confusion",
        "--routing-confidence-root",
        "outputs/analysis/figures/routing_confidence",
        "--correctness-conditioned-root",
        "outputs/analysis/figures/correctness_conditioned",
        "--expert-contribution-root",
        "outputs/analysis/figures/expert_contribution",
        "--router-geometry-root",
        "outputs/analysis/figures/router_geometry",
        "--representation-geometry-root",
        "outputs/analysis/figures/representation_geometry",
        "--routing-light-root",
        "outputs/analysis/captures/routing_light",
        "--output-root",
        "outputs/analysis/tables",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis figures and tables.")
    parser.add_argument(
        "--analysis",
        default="all",
        choices=(
            "all",
            "coactivation",
            "top1_top2_confusion",
            "routing_confidence",
            "correctness_conditioned",
            "router_direction",
            "router_geometry",
            "latent_space",
            "representation_geometry",
            "weight_analysis",
            "routing_weight_bridge",
            "token_features",
            "expert_contribution",
            "summary_tables",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print("Command:")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=BUNDLE_ROOT)


def main() -> int:
    args = parse_args()
    analyses = list(REPORT_COMMANDS) if args.analysis == "all" else [args.analysis]
    for analysis in analyses:
        run_command(REPORT_COMMANDS[analysis], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
