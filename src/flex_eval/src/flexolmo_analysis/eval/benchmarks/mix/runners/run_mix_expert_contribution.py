from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import FlexOlmoForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT.parent

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flexolmo_analysis.toolkit.adapters.flex_olmo import iter_flex_olmo_layers
from flexolmo_analysis.toolkit.pipelines.flex_olmo import restricted_expert_mode
from flexolmo_analysis.toolkit.pipelines.flex_olmo_eval import build_run_specs
from flexolmo_analysis.toolkit.utils import load_tokenizer_with_known_fixes
from flexolmo_analysis.toolkit.utils.jsonl import write_jsonl
from flexolmo_analysis.eval.benchmarks.mix.runners.run_mix_analysis import (
    apply_chat_template_if_requested,
    load_allowed_model_names,
    load_jsonl_records,
    load_manifest_entries,
    normalize_examples,
    parse_dtype,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_contribution" / "a4"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture token-level expert contribution summaries for selected FlexOlmo/FlexMoRE layers."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=75)
    parser.add_argument("--model-path", help="Explicit path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--public-expert-idx", type=int, default=0)
    parser.add_argument("--combined-active-experts", default="2,4,7")
    parser.add_argument(
        "--routing-run-mode",
        default="native_only",
        choices=("restricted_sweep", "native_only", "native_plus_restricted"),
    )
    parser.add_argument("--expert-order")
    parser.add_argument("--include-individual-experts", action="store_true")
    parser.add_argument(
        "--selected-layers",
        default="early_mid_late_last",
        help="Comma-separated decoder layer indices or presets like `early_mid_late_last`.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def parse_decoder_layers(raw_value: str, num_hidden_layers: int) -> list[int]:
    normalized = raw_value.strip().lower()
    if normalized == "all":
        return list(range(num_hidden_layers))
    if normalized == "early_mid_late_last":
        if num_hidden_layers <= 1:
            return [0]
        return sorted(
            {
                0,
                int(round((num_hidden_layers - 1) * 0.33)),
                int(round((num_hidden_layers - 1) * 0.66)),
                num_hidden_layers - 1,
            }
        )
    if normalized == "early_mid_last":
        if num_hidden_layers <= 1:
            return [0]
        return sorted({0, int(round((num_hidden_layers - 1) * 0.5)), num_hidden_layers - 1})
    if normalized == "early_late_last":
        if num_hidden_layers <= 1:
            return [0]
        return sorted({0, int(round((num_hidden_layers - 1) * 0.75)), num_hidden_layers - 1})
    return sorted({int(part.strip()) for part in raw_value.split(",") if part.strip()})


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if not args.model_name or not args.model_root:
        raise ValueError("Provide either --model-path or both --model-name and --model-root.")
    allowed_names = load_allowed_model_names(args.model_registry)
    if args.model_name not in allowed_names:
        raise ValueError(f"Model name `{args.model_name}` was not found in {args.model_registry}.")
    return str(Path(args.model_root) / args.model_name)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str | None, device: torch.device, dtype_name: str):
    model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=parse_dtype(dtype_name))
    model.config.use_cache = False
    model.config.output_hidden_states = True
    model.config.output_router_logits = True
    model.config.return_dict = True
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer_with_known_fixes(tokenizer_path or model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either `pad_token_id` or `eos_token_id`.")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def normalize_example(tokenizer, record: dict, dataset_name: str, dataset_entry: dict) -> dict:
    prompt = record.get("prompt")
    if not prompt:
        raise ValueError(f"Dataset `{dataset_name}` contains a record without `prompt`.")
    prompting_config = dict(dataset_entry.get("prompting", {}))
    return {
        "example_id": record["example_id"],
        "dataset_name": dataset_name,
        "language": record.get("language", "unknown"),
        "prompt": apply_chat_template_if_requested(tokenizer, prompt, prompting_config),
        "question": record.get("question"),
    }


def encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def decode_token_ids(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in token_ids]


def _compute_expert_output(experts_module, expert_idx: int, token_states: torch.Tensor) -> torch.Tensor:
    if hasattr(experts_module, "gate_up_proj") and hasattr(experts_module, "down_proj"):
        gate, up = F.linear(token_states, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        hidden = experts_module.act_fn(gate) * up
        return F.linear(hidden, experts_module.down_proj[expert_idx])
    if hasattr(experts_module, "__getitem__"):
        return experts_module[expert_idx](token_states)
    raise TypeError("Unsupported experts module for expert contribution capture.")


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    a_norm = torch.linalg.vector_norm(a_flat)
    b_norm = torch.linalg.vector_norm(b_flat)
    if float(a_norm.item()) == 0.0 or float(b_norm.item()) == 0.0:
        return 0.0
    return float(torch.dot(a_flat, b_flat).item() / (a_norm.item() * b_norm.item()))


class ExpertContributionCapture:
    def __init__(self, selected_layers: list[int]):
        self.selected_layers = selected_layers
        self.records: dict[int, list[dict[str, object]]] = {}
        self.handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            hidden_states = args[0].detach()
            if hidden_states.ndim == 3:
                if hidden_states.shape[0] != 1:
                    raise ValueError(
                        f"Expected batch size 1 in expert contribution hook, got shape {tuple(hidden_states.shape)}."
                    )
                hidden_states = hidden_states[0]
            elif hidden_states.ndim != 2:
                raise ValueError(
                    f"Unexpected hidden state shape in expert contribution hook: {tuple(hidden_states.shape)}"
                )
            router_probs, top_k_weights, top_k_index = module.gate(hidden_states)
            experts_module = module.experts

            seq_len = hidden_states.shape[0]
            layer_records: list[dict[str, object]] = []

            for token_idx in range(seq_len):
                token_state = hidden_states[token_idx : token_idx + 1]
                selected_indices = top_k_index[token_idx].detach().cpu().tolist()
                selected_weights = top_k_weights[token_idx].detach().cpu().tolist()
                raw_norms: list[float] = []
                weighted_norms: list[float] = []
                ablation_delta_norms: list[float] = []
                ablation_delta_ratios: list[float] = []
                raw_outputs: list[torch.Tensor] = []
                weighted_outputs: list[torch.Tensor] = []

                for selected_pos, expert_idx in enumerate(selected_indices):
                    expert_output = _compute_expert_output(experts_module, int(expert_idx), token_state)
                    raw_norm = float(torch.linalg.vector_norm(expert_output).item())
                    weighted_output = expert_output * float(selected_weights[selected_pos])
                    weighted_norm = float(torch.linalg.vector_norm(weighted_output).item())
                    raw_norms.append(raw_norm)
                    weighted_norms.append(weighted_norm)
                    raw_outputs.append(expert_output)
                    weighted_outputs.append(weighted_output)

                mixture_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)
                mixture_output_norm = float(torch.linalg.vector_norm(mixture_output).item())
                contribution_total = sum(weighted_norms)
                shares = [
                    (weighted_norm / contribution_total) if contribution_total > 0 else 0.0
                    for weighted_norm in weighted_norms
                ]
                for weighted_output in weighted_outputs:
                    ablation_delta = mixture_output - (mixture_output - weighted_output)
                    delta_norm = float(torch.linalg.vector_norm(ablation_delta).item())
                    ablation_delta_norms.append(delta_norm)
                    ablation_delta_ratios.append(delta_norm / mixture_output_norm if mixture_output_norm > 0 else 0.0)

                top1_top2_raw_output_cosine = 0.0
                top1_top2_weighted_output_cosine = 0.0
                if len(raw_outputs) > 1:
                    top1_top2_raw_output_cosine = _safe_cosine(raw_outputs[0], raw_outputs[1])
                if len(weighted_outputs) > 1:
                    top1_top2_weighted_output_cosine = _safe_cosine(weighted_outputs[0], weighted_outputs[1])
                mixture_alignment_ratio = mixture_output_norm / contribution_total if contribution_total > 0 else 0.0

                layer_records.append(
                    {
                        "layer": int(layer_idx),
                        "token_idx": int(token_idx),
                        "selected_expert_ids": [int(idx) for idx in selected_indices],
                        "selected_router_weights": [float(weight) for weight in selected_weights],
                        "raw_expert_output_norms": raw_norms,
                        "weighted_expert_output_norms": weighted_norms,
                        "ablation_delta_norms": ablation_delta_norms,
                        "ablation_delta_ratios": ablation_delta_ratios,
                        "contribution_shares": shares,
                        "mixture_output_norm": mixture_output_norm,
                        "mixture_alignment_ratio": mixture_alignment_ratio,
                        "top1_top2_raw_output_cosine": top1_top2_raw_output_cosine,
                        "top1_top2_weighted_output_cosine": top1_top2_weighted_output_cosine,
                        "top1_contribution_share": float(shares[0]) if shares else 0.0,
                        "top2_contribution_share": float(shares[1]) if len(shares) > 1 else 0.0,
                        "top1_weighted_output_norm": float(weighted_norms[0]) if weighted_norms else 0.0,
                        "top2_weighted_output_norm": float(weighted_norms[1]) if len(weighted_norms) > 1 else 0.0,
                        "top1_ablation_delta_norm": float(ablation_delta_norms[0]) if ablation_delta_norms else 0.0,
                        "top2_ablation_delta_norm": float(ablation_delta_norms[1]) if len(ablation_delta_norms) > 1 else 0.0,
                        "top1_ablation_delta_ratio": float(ablation_delta_ratios[0]) if ablation_delta_ratios else 0.0,
                        "top2_ablation_delta_ratio": float(ablation_delta_ratios[1]) if len(ablation_delta_ratios) > 1 else 0.0,
                    }
                )

            self.records[layer_idx] = layer_records

        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_hook(self._make_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "dataset"


def collect_example_records(
    model,
    tokenizer,
    example: dict[str, Any],
    run_label: str,
    selected_layers: list[int],
    max_length: int,
    device: torch.device,
) -> list[dict[str, object]]:
    inputs = encode_prompt(tokenizer, example["prompt"], max_length=max_length, device=device)
    prompt_length = int(inputs["attention_mask"][0].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"].shape[-1])
    input_token_ids = inputs["input_ids"][0, :prompt_length].detach().cpu().tolist()
    decoded_tokens = decode_token_ids(tokenizer, input_token_ids)

    capture = ExpertContributionCapture(selected_layers)
    capture.attach(model)
    try:
        with torch.no_grad():
            model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )
    finally:
        capture.remove()

    rows: list[dict[str, object]] = []
    for layer_idx in selected_layers:
        for record in capture.records.get(layer_idx, [])[:prompt_length]:
            token_idx = int(record["token_idx"])
            rows.append(
                {
                    "example_id": example["example_id"],
                    "dataset_name": example["dataset_name"],
                    "language": example["language"],
                    "run_label": run_label,
                    "layer": int(record["layer"]),
                    "token_idx": token_idx,
                    "token_id": int(input_token_ids[token_idx]),
                    "token_text": decoded_tokens[token_idx],
                    **record,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}
    device = resolve_device(args.device)
    model_path = resolve_model_path(args)
    model_name = args.model_name or Path(model_path).name
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        dtype_name=args.dtype,
    )
    selected_layers = parse_decoder_layers(args.selected_layers, int(model.config.num_hidden_layers))

    expert_order = None
    if args.expert_order:
        expert_order = tuple(int(part.strip()) for part in args.expert_order.split(",") if part.strip())
    run_specs = build_run_specs(
        num_experts=model.config.num_experts,
        public_expert_idx=args.public_expert_idx,
        combined_active_counts=tuple(int(part.strip()) for part in args.combined_active_experts.split(",") if part.strip()),
        include_individual_experts=args.include_individual_experts,
        expert_order=expert_order,
        routing_run_mode=args.routing_run_mode,
    )

    manifest_entries = load_manifest_entries(args.manifest_path, selected_datasets)
    if not manifest_entries:
        raise ValueError("No mix datasets were selected from the manifest.")

    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": selected_layers,
        "routing_run_mode": args.routing_run_mode,
        "run_labels": [run_spec.label for run_spec in run_specs],
        "datasets": {},
    }

    for dataset_entry in manifest_entries:
        dataset_name = str(dataset_entry["name"])
        examples = normalize_examples(
            tokenizer,
            load_jsonl_records(dataset_entry["path"], max_examples=args.max_examples_per_dataset),
            dataset_name,
            dataset_entry,
        )
        if not examples:
            continue

        dataset_dir = model_output_root / sanitize_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        suite_manifest["datasets"][dataset_name] = {
            "path": str(dataset_entry["path"]),
            "num_examples": len(examples),
            "runs": {},
        }

        for run_spec in run_specs:
            run_dir = dataset_dir / sanitize_name(run_spec.label)
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata_rows: list[dict[str, object]] = []

            routing_context = (
                restricted_expert_mode(model, allowed_experts=run_spec.allowed_experts)
                if run_spec.apply_restricted_routing
                else nullcontext()
            )
            with routing_context:
                for example in examples:
                    metadata_rows.extend(
                        collect_example_records(
                            model=model,
                            tokenizer=tokenizer,
                            example=example,
                            run_label=run_spec.label,
                            selected_layers=selected_layers,
                            max_length=args.max_length,
                            device=device,
                        )
                    )

            metadata_path = run_dir / "expert_contribution_records.jsonl"
            manifest_path = run_dir / "expert_contribution_manifest.json"
            write_jsonl(metadata_rows, metadata_path, sort_keys=False)
            manifest = {
                "model_name": model_name,
                "model_path": model_path,
                "dataset_name": dataset_name,
                "run_label": run_spec.label,
                "selected_layers": selected_layers,
                "num_examples": len(examples),
                "num_token_rows": len(metadata_rows),
                "records_path": str(metadata_path),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            suite_manifest["datasets"][dataset_name]["runs"][run_spec.label] = {
                "records_path": str(metadata_path),
                "manifest_path": str(manifest_path),
            }
            print(f"Captured expert contributions for {model_name} on {dataset_name} / {run_spec.label}")

    suite_manifest_path = model_output_root / "expert_contribution_suite_manifest.json"
    suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote expert-contribution suite manifest to {suite_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
