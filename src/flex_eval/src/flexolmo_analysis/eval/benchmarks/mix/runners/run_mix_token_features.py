from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
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
from flexolmo_analysis.toolkit.utils.token_features import align_offsets_to_words
from flexolmo_analysis.eval.benchmarks.mix.runners.run_mix_analysis import (
    apply_chat_template_if_requested,
    load_allowed_model_names,
    load_jsonl_records,
    load_manifest_entries,
    normalize_examples,
    parse_dtype,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "token_features" / "a4"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture reusable token-level routing metadata plus selected dense vectors for downstream "
            "router-separability and fragmentation analyses."
        )
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
    parser.add_argument(
        "--representation-sources",
        default="pre_router,hidden_state",
        help="Comma-separated token feature sources to save. Supported: pre_router, hidden_state.",
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


def parse_representation_sources(raw_value: str) -> list[str]:
    sources = [part.strip() for part in raw_value.split(",") if part.strip()]
    allowed = {"pre_router", "hidden_state"}
    invalid = [source for source in sources if source not in allowed]
    if invalid:
        raise ValueError(f"Unsupported representation sources: {', '.join(invalid)}")
    if not sources:
        raise ValueError("Provide at least one representation source.")
    return sources


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


def encode_prompt(tokenizer, prompt: str, max_length: int, device: torch.device, add_special_tokens: bool = True):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def prompt_offset_metadata(tokenizer, prompt: str, max_length: int, add_special_tokens: bool = True) -> list[dict[str, object]]:
    if not getattr(tokenizer, "is_fast", False):
        return []
    encoded = tokenizer(
        prompt,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return align_offsets_to_words(prompt, offsets)


def decode_token_ids(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in token_ids]


class TokenFeatureCapture:
    def __init__(self, selected_layers: list[int], capture_pre_router: bool):
        self.selected_layers = selected_layers
        self.capture_pre_router = capture_pre_router
        self.pre_router_states: dict[int, torch.Tensor] = {}
        self.router_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.handles = []

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args):
            self.pre_router_states[layer_idx] = args[0].detach().cpu().float()
        return hook

    def _make_router_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            router_probs, top_k_weights, top_k_index = output
            self.router_outputs[layer_idx] = (
                router_probs.detach().cpu().float(),
                top_k_weights.detach().cpu().float(),
                top_k_index.detach().cpu(),
            )
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            if self.capture_pre_router:
                self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_pre_hook(layer_idx)))
            self.handles.append(layers[layer_idx].mlp.gate.register_forward_hook(self._make_router_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def hidden_state_by_decoder_layer(hidden_states, decoder_layers: list[int]) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for layer_idx in decoder_layers:
        hidden_state_idx = layer_idx + 1
        if hidden_state_idx >= len(hidden_states):
            raise ValueError(
                f"Decoder layer {layer_idx} cannot be mapped to hidden-state index {hidden_state_idx} "
                f"for only {len(hidden_states)} tensors."
            )
        result[layer_idx] = hidden_states[hidden_state_idx].detach().cpu().float()
    return result


def collect_token_feature_artifacts(
    model,
    tokenizer,
    example: dict[str, Any],
    run_label: str,
    selected_layers: list[int],
    representation_sources: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    inputs = encode_prompt(tokenizer, example["prompt"], max_length=max_length, device=device, add_special_tokens=True)
    prompt_length = int(inputs["attention_mask"][0].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"].shape[-1])
    input_token_ids = inputs["input_ids"][0, :prompt_length].detach().cpu().tolist()
    decoded_tokens = decode_token_ids(tokenizer, input_token_ids)
    offset_rows = prompt_offset_metadata(tokenizer, example["prompt"], max_length=max_length, add_special_tokens=True)
    offset_by_token_idx = {int(row["token_idx"]): row for row in offset_rows}

    capture = TokenFeatureCapture(selected_layers, capture_pre_router="pre_router" in representation_sources)
    capture.attach(model)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states="hidden_state" in representation_sources,
                use_cache=False,
            )
    finally:
        capture.remove()

    vector_rows: dict[str, list[np.ndarray]] = {
        f"{source}_layer_{layer_idx}": []
        for source in representation_sources
        for layer_idx in selected_layers
    }
    metadata_rows: list[dict[str, object]] = []

    hidden_state_rows = (
        hidden_state_by_decoder_layer(outputs.hidden_states, selected_layers)
        if "hidden_state" in representation_sources
        else {}
    )

    for layer_idx in selected_layers:
        if layer_idx not in capture.router_outputs:
            raise ValueError(f"Router outputs for layer {layer_idx} were not captured.")
        router_probs, _top_k_weights, top_k_index = capture.router_outputs[layer_idx]
        layer_probs = router_probs[:prompt_length]
        layer_topk = top_k_index[:prompt_length]
        top2_probs, top2_indices = torch.topk(layer_probs, k=min(2, layer_probs.shape[-1]), dim=-1)
        token_entropy = -(layer_probs * torch.log(layer_probs.clamp_min(1e-9))).sum(dim=-1)
        selected_mass = layer_probs.gather(dim=-1, index=layer_topk).sum(dim=-1)

        if "pre_router" in representation_sources:
            pre_router_tensor = capture.pre_router_states[layer_idx][0, :prompt_length, :]
        else:
            pre_router_tensor = None
        if "hidden_state" in representation_sources:
            hidden_state_tensor = hidden_state_rows[layer_idx][0, :prompt_length, :]
        else:
            hidden_state_tensor = None

        row_idx_within_layer = 0
        for token_idx in range(prompt_length):
            offset_info = offset_by_token_idx.get(token_idx, {})
            token_record = {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example["language"],
                "run_label": run_label,
                "layer": int(layer_idx),
                "row_idx_within_layer": row_idx_within_layer,
                "token_idx": int(token_idx),
                "token_id": int(input_token_ids[token_idx]),
                "token_text": decoded_tokens[token_idx],
                "top1_expert": int(top2_indices[token_idx, 0].item()),
                "top2_expert": int(top2_indices[token_idx, 1].item()) if top2_indices.shape[1] > 1 else None,
                "top1_prob": float(top2_probs[token_idx, 0].item()),
                "top2_prob": float(top2_probs[token_idx, 1].item()) if top2_probs.shape[1] > 1 else 0.0,
                "top1_top2_margin": (
                    float((top2_probs[token_idx, 0] - top2_probs[token_idx, 1]).item())
                    if top2_probs.shape[1] > 1
                    else 0.0
                ),
                "token_entropy": float(token_entropy[token_idx].item()),
                "selected_expert_prob_mass": float(selected_mass[token_idx].item()),
                "offset_start": offset_info.get("offset_start"),
                "offset_end": offset_info.get("offset_end"),
                "word_idx": offset_info.get("word_idx"),
                "word_text": offset_info.get("word_text"),
                "word_subtoken_count": offset_info.get("word_subtoken_count"),
                "fragmentation_bucket": offset_info.get("fragmentation_bucket", "unknown"),
            }
            metadata_rows.append(token_record)
            if pre_router_tensor is not None:
                vector_rows[f"pre_router_layer_{layer_idx}"].append(pre_router_tensor[token_idx].numpy())
            if hidden_state_tensor is not None:
                vector_rows[f"hidden_state_layer_{layer_idx}"].append(hidden_state_tensor[token_idx].numpy())
            row_idx_within_layer += 1

    return metadata_rows, vector_rows


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "dataset"


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
    representation_sources = parse_representation_sources(args.representation_sources)

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
        "representation_sources": representation_sources,
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
            vector_rows: dict[str, list[np.ndarray]] = {
                f"{source}_layer_{layer_idx}": []
                for source in representation_sources
                for layer_idx in selected_layers
            }

            routing_context = (
                restricted_expert_mode(model, allowed_experts=run_spec.allowed_experts)
                if run_spec.apply_restricted_routing
                else nullcontext()
            )
            with routing_context:
                for example in examples:
                    example_metadata_rows, example_vector_rows = collect_token_feature_artifacts(
                        model=model,
                        tokenizer=tokenizer,
                        example=example,
                        run_label=run_spec.label,
                        selected_layers=selected_layers,
                        representation_sources=representation_sources,
                        max_length=args.max_length,
                        device=device,
                    )
                    metadata_rows.extend(example_metadata_rows)
                    for key, values in example_vector_rows.items():
                        vector_rows[key].extend(values)

            vectors_path = run_dir / "token_feature_vectors.npz"
            metadata_path = run_dir / "token_feature_metadata.jsonl"
            manifest_path = run_dir / "token_feature_manifest.json"
            write_jsonl(metadata_rows, metadata_path, sort_keys=False)
            np.savez_compressed(
                vectors_path,
                **{
                    key: np.stack(values).astype(np.float32) if values else np.zeros((0, model.config.hidden_size), dtype=np.float32)
                    for key, values in vector_rows.items()
                },
            )
            manifest = {
                "model_name": model_name,
                "model_path": model_path,
                "dataset_name": dataset_name,
                "run_label": run_spec.label,
                "selected_layers": selected_layers,
                "representation_sources": representation_sources,
                "num_examples": len(examples),
                "num_token_rows": len(metadata_rows),
                "vectors_path": str(vectors_path),
                "metadata_path": str(metadata_path),
                "row_counts_by_layer": {
                    str(layer_idx): sum(1 for row in metadata_rows if int(row["layer"]) == layer_idx)
                    for layer_idx in selected_layers
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            suite_manifest["datasets"][dataset_name]["runs"][run_spec.label] = {
                "metadata_path": str(metadata_path),
                "vectors_path": str(vectors_path),
                "manifest_path": str(manifest_path),
            }
            print(f"Captured token features for {model_name} on {dataset_name} / {run_spec.label}")

    suite_manifest_path = model_output_root / "token_feature_suite_manifest.json"
    suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote token-feature suite manifest to {suite_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
