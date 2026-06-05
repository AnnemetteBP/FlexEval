from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from flexolmo_analysis.eval.benchmarks.mix.runners.run_mix_suite import resolve_model_entries


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_expert_contribution.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mix expert-contribution suite across multiple checkpoints.")
    parser.add_argument("--config", required=True, help="Path to the expert-contribution suite config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command: list[str], dry_run: bool) -> None:
    print("Command:")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    shared = config.get("shared", {})
    runtime = config.get("runtime", {})
    models = resolve_model_entries(config)
    if not models:
        raise ValueError("No enabled models were defined in the expert-contribution config.")

    selected_layers = shared.get("selected_layers", "early_mid_late_last")
    selected_layers_arg = ",".join(str(layer) for layer in selected_layers) if isinstance(selected_layers, list) else str(selected_layers)
    datasets = shared.get("datasets", [])
    datasets_arg = ",".join(str(name) for name in datasets) if datasets else None
    output_root = runtime.get("output_root", str(PROJECT_ROOT / "eval_results" / "mix" / "expert_contribution" / "a4"))
    Path(output_root).mkdir(parents=True, exist_ok=True)

    commands = []
    for model in models:
        command = [
            sys.executable,
            str(RUNNER_PATH),
            "--manifest-path",
            str(shared["dataset_manifest"]),
            "--device",
            str(runtime.get("device", "auto")),
            "--dtype",
            str(runtime.get("dtype", "auto")),
            "--max-length",
            str(runtime.get("max_length", 512)),
            "--selected-layers",
            selected_layers_arg,
            "--max-examples-per-dataset",
            str(runtime.get("max_examples_per_dataset", 75)),
            "--output-root",
            str(output_root),
            "--public-expert-idx",
            str(runtime.get("public_expert_idx", 0)),
            "--combined-active-experts",
            str(runtime.get("combined_active_experts", "2,4,7")),
            "--routing-run-mode",
            str(runtime.get("routing_run_mode", "native_only")),
            "--model-name",
            str(model["model_name"]),
            "--model-root",
            str(shared["model_root"]),
            "--model-registry",
            str(shared["model_registry"]),
        ]
        if runtime.get("include_individual_experts", False):
            command.append("--include-individual-experts")
        if datasets_arg:
            command.extend(["--datasets", datasets_arg])
        if runtime.get("expert_order"):
            command.extend(["--expert-order", str(runtime["expert_order"])])
        commands.append(command)

    manifest_path = Path(output_root) / "expert_contribution_suite_commands.json"
    manifest_path.write_text(json.dumps({"config": str(Path(args.config).resolve()), "commands": commands}, indent=2))
    for command in commands:
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
