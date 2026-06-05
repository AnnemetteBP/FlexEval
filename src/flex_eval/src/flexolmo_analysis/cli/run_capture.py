from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "configs"

CAPTURE_COMMANDS = {
    "routing_light": (
        PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_suite.py",
        CONFIG_ROOT / "routing_light_pair.json",
    ),
    "router_direction": (
        PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_router_direction_suite.py",
        CONFIG_ROOT / "router_direction_pair.json",
    ),
    "weight_analysis": (
        PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_weight_analysis_suite.py",
        CONFIG_ROOT / "weight_analysis_pair.json",
    ),
    "latent_space": (
        PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_latent_space_suite.py",
        CONFIG_ROOT / "latent_space_pair.json",
    ),
    "token_features": (
        PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_token_feature_suite.py",
        CONFIG_ROOT / "token_features_pair.json",
    ),
    "expert_contribution": (
        PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_expert_contribution_suite.py",
        CONFIG_ROOT / "expert_contribution_pair.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis capture stages.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "routing_light", "router_direction", "weight_analysis", "latent_space", "token_features", "expert_contribution"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def verify_flexolmo_transformers() -> None:
    try:
        from transformers import FlexOlmoForCausalLM  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment preflight
        raise SystemExit(
            "Capture stages require a `transformers` installation that provides "
            "`FlexOlmoForCausalLM`. The bundled code alone is not enough. "
            "See the top-level `README.md` in this bundle."
        ) from exc


def run_command(command: list[str], dry_run: bool) -> None:
    print("Command:")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=BUNDLE_ROOT)


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        verify_flexolmo_transformers()
    stages = list(CAPTURE_COMMANDS) if args.stage == "all" else [args.stage]
    for stage in stages:
        runner_path, config_path = CAPTURE_COMMANDS[stage]
        command = [sys.executable, str(runner_path), "--config", str(config_path)]
        if args.dry_run:
            command.append("--dry-run")
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
