from __future__ import annotations

import argparse
from collections import defaultdict
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT.parent

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flexolmo_analysis.toolkit.adapters.flex_olmo import iter_flex_olmo_layers
from flexolmo_analysis.toolkit.pipelines.flex_olmo import restricted_expert_mode
from flexolmo_analysis.toolkit.pipelines.flex_olmo_eval import build_run_specs
from flexolmo_analysis.toolkit.pipelines.flex_olmo_routing_dataset import (
    aggregate_routing_analysis,
    aggregate_router_token_summaries,
    analyze_prompt_example,
    summarize_routing_records,
)
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
    parse_hidden_state_layers,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "deep_base" / "a4"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mix_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture reusable deep base MoE data for downstream multi-wiki QA analyses."
    )
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--datasets", help="Optional comma-separated dataset names to include from the manifest.")
    parser.add_argument("--max-examples-per-dataset", type=int, default=500)
    parser.add_argument("--model-path", help="Explicit path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--tokenizer-path", help="Optional tokenizer path. Defaults to the resolved model path.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--default-max-new-tokens", type=int, default=512)
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
        default="all",
        help="Supported: `all`, `first_early_mid_late_last`, numeric comma-separated decoder layers.",
    )
    parser.add_argument("--save-raw-artifacts", action="store_true")
    parser.add_argument("--router-direction-output-root", help="Optional output root for router-direction artifacts.")
    parser.add_argument(
        "--router-direction-selected-layers",
        help="Optional decoder layers for router-direction artifacts. Defaults to --selected-layers.",
    )
    parser.add_argument(
        "--router-direction-position-policy",
        default="last_prompt_token",
        choices=("last_prompt_token", "mean_prompt"),
    )
    parser.add_argument(
        "--router-direction-alignment-metric",
        default="cosine",
        choices=("cosine", "dot"),
    )
    parser.add_argument("--latent-space-output-root", help="Optional output root for latent-space artifacts.")
    parser.add_argument(
        "--latent-space-selected-layers",
        help="Optional hidden-state/pre-router layers for latent-space artifacts. Defaults to --selected-layers.",
    )
    parser.add_argument(
        "--latent-space-representation-sources",
        default="embedding,hidden_state,pre_router",
        help="Comma-separated latent sources. Supported: embedding, hidden_state, pre_router.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def parse_decoder_layers(raw_value: str, num_hidden_layers: int) -> list[int]:
    normalized = raw_value.strip().lower()
    if normalized == "all":
        return list(range(num_hidden_layers))
    if normalized in {"first_early_mid_late_last", "first-early-mid-late-last"}:
        if num_hidden_layers <= 1:
            return [0]
        return sorted(
            {
                0,
                int(round((num_hidden_layers - 1) * 0.25)),
                int(round((num_hidden_layers - 1) * 0.50)),
                int(round((num_hidden_layers - 1) * 0.75)),
                num_hidden_layers - 1,
            }
        )
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
    from transformers import FlexOlmoForCausalLM

    model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=parse_dtype(dtype_name))
    model.config.use_cache = False
    model.config.output_hidden_states = True
    model.config.output_router_logits = True
    model.config.return_dict = True
    model.to(device)
    model.eval()
    print(f"[deep_base] model_dtype={infer_model_dtype(model)}")
    resolved_tokenizer_path = tokenizer_path or model_path
    tokenizer = load_tokenizer_with_known_fixes(resolved_tokenizer_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either `pad_token_id` or `eos_token_id`.")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def infer_model_dtype(model) -> str:
    for param in model.parameters():
        return str(param.dtype)
    return "unknown"


def normalize_example(tokenizer, record: dict, dataset_name: str, dataset_entry: dict) -> dict:
    prompt = record.get("prompt")
    if not prompt:
        raise ValueError(f"Dataset `{dataset_name}` contains a record without `prompt`.")
    prompting_config = dict(dataset_entry.get("prompting", {}))
    generation_config = dict(dataset_entry.get("generation", {}))
    tokenization_config = dict(dataset_entry.get("tokenization", {}))
    return {
        "example_id": record["example_id"],
        "dataset_name": dataset_name,
        "language": record.get("language", "unknown"),
        "prompt": apply_chat_template_if_requested(tokenizer, prompt, prompting_config),
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "scoring_mode": record.get("scoring_mode", "qa"),
        "prompting_config": prompting_config,
        "generation_config": generation_config,
        "tokenization_config": tokenization_config,
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


def sanitize_name(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "dataset"


def assert_all_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite tensor detected in {name}.")
        return
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite array detected in {name}.")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_all_finite(f"{name}.{key}", item)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            assert_all_finite(f"{name}[{idx}]", item)
        return
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite scalar detected in {name}: {value}")
        return


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


class DeepCapture:
    def __init__(self, selected_layers: list[int]):
        self.selected_layers = selected_layers
        self.pre_router_states: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.contrib_records: dict[int, list[dict[str, object]]] = {}
        self.contrib_vectors: dict[int, list[dict[str, np.ndarray]]] = {}

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args):
            self.pre_router_states[layer_idx] = args[0].detach().cpu().float()
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(module, args, _output):
            hidden_states = args[0].detach()
            if hidden_states.ndim == 3:
                if hidden_states.shape[0] != 1:
                    raise ValueError(
                        f"Expected batch size 1 in deep capture hook, got shape {tuple(hidden_states.shape)}."
                    )
                hidden_states = hidden_states[0]
            elif hidden_states.ndim != 2:
                raise ValueError(f"Unexpected hidden state shape in deep capture hook: {tuple(hidden_states.shape)}")
            _router_probs, top_k_weights, top_k_index = module.gate(hidden_states)
            experts_module = module.experts

            layer_records: list[dict[str, object]] = []
            layer_vectors: list[dict[str, np.ndarray]] = []
            seq_len = hidden_states.shape[0]

            for token_idx in range(seq_len):
                token_state = hidden_states[token_idx : token_idx + 1]
                selected_indices = top_k_index[token_idx].detach().cpu().tolist()
                selected_weights = top_k_weights[token_idx].detach().cpu().tolist()

                raw_outputs: list[torch.Tensor] = []
                weighted_outputs: list[torch.Tensor] = []
                raw_norms: list[float] = []
                weighted_norms: list[float] = []
                ablation_delta_norms: list[float] = []
                ablation_delta_ratios: list[float] = []

                for selected_pos, expert_idx in enumerate(selected_indices):
                    expert_output = _compute_expert_output(experts_module, int(expert_idx), token_state)
                    raw_outputs.append(expert_output.detach().cpu().float())
                    weighted_output = expert_output * float(selected_weights[selected_pos])
                    weighted_outputs.append(weighted_output.detach().cpu().float())
                    raw_norms.append(float(torch.linalg.vector_norm(expert_output).item()))
                    weighted_norms.append(float(torch.linalg.vector_norm(weighted_output).item()))

                mixture_output = torch.stack(weighted_outputs, dim=0).sum(dim=0) if weighted_outputs else token_state * 0.0
                mixture_output_cpu = mixture_output.detach().cpu().float()
                mixture_output_norm = float(torch.linalg.vector_norm(mixture_output).item())
                contribution_total = sum(weighted_norms)
                shares = [
                    (weighted_norm / contribution_total) if contribution_total > 0 else 0.0
                    for weighted_norm in weighted_norms
                ]
                for weighted_output in weighted_outputs:
                    delta_norm = float(torch.linalg.vector_norm(weighted_output).item())
                    ablation_delta_norms.append(delta_norm)
                    ablation_delta_ratios.append(delta_norm / mixture_output_norm if mixture_output_norm > 0 else 0.0)

                top1_top2_raw_output_cosine = 0.0
                top1_top2_weighted_output_cosine = 0.0
                if len(raw_outputs) > 1:
                    top1_top2_raw_output_cosine = _safe_cosine(raw_outputs[0], raw_outputs[1])
                if len(weighted_outputs) > 1:
                    top1_top2_weighted_output_cosine = _safe_cosine(weighted_outputs[0], weighted_outputs[1])
                mixture_alignment_ratio = mixture_output_norm / contribution_total if contribution_total > 0 else 0.0

                hidden_size = int(token_state.shape[-1])
                raw_output_stack = np.zeros((len(selected_indices), hidden_size), dtype=np.float32)
                weighted_output_stack = np.zeros((len(selected_indices), hidden_size), dtype=np.float32)
                for output_idx, raw_output in enumerate(raw_outputs):
                    flattened = raw_output.reshape(-1)
                    if flattened.numel() != hidden_size:
                        raise ValueError(
                            f"Expected expert output size {hidden_size}, got shape {tuple(raw_output.shape)} "
                            f"with {flattened.numel()} elements."
                        )
                    raw_output_stack[output_idx] = flattened.numpy()
                for output_idx, weighted_output in enumerate(weighted_outputs):
                    flattened = weighted_output.reshape(-1)
                    if flattened.numel() != hidden_size:
                        raise ValueError(
                            f"Expected weighted expert output size {hidden_size}, got shape {tuple(weighted_output.shape)} "
                            f"with {flattened.numel()} elements."
                        )
                    weighted_output_stack[output_idx] = flattened.numpy()

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
                layer_vectors.append(
                    {
                        "selected_expert_ids": np.asarray(selected_indices, dtype=np.int64),
                        "selected_router_weights": np.asarray(selected_weights, dtype=np.float32),
                        "raw_selected_expert_outputs": raw_output_stack,
                        "weighted_selected_expert_outputs": weighted_output_stack,
                        "mixture_output": mixture_output_cpu.reshape(-1).numpy(),
                    }
                )

            self.contrib_records[layer_idx] = layer_records
            self.contrib_vectors[layer_idx] = layer_vectors

        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_pre_hook(layer_idx)))
            self.handles.append(layers[layer_idx].mlp.register_forward_hook(self._make_mlp_hook(layer_idx)))

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


def collect_deep_example(
    model,
    tokenizer,
    example: dict[str, Any],
    run_spec,
    selected_layers: list[int],
    max_length: int,
    default_max_new_tokens: int,
    device: torch.device,
    save_raw_artifacts: bool,
) -> tuple[dict[str, Any], list[dict[str, object]], dict[str, list[np.ndarray]], list[dict[str, object]], dict[str, list[np.ndarray]]]:
    routing_record = analyze_prompt_example(
        model=model,
        tokenizer=tokenizer,
        example=example,
        run_spec=run_spec,
        max_length=max_length,
        device=device,
        capture_output_token_ids=True,
        default_max_new_tokens=default_max_new_tokens,
        capture_router_tensors=True,
        capture_hidden_states=False,
        hidden_state_layers=None,
    )
    assert_all_finite(f"routing_record[{example['example_id']}]", routing_record)

    if not save_raw_artifacts:
        return routing_record, [], {}, [], {}

    inputs = encode_prompt(
        tokenizer,
        example["prompt"],
        max_length=max_length,
        device=device,
        add_special_tokens=bool(example.get("tokenization_config", {}).get("prompt_add_special_tokens", True)),
    )
    prompt_length = int(inputs["attention_mask"][0].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"].shape[-1])
    input_token_ids = inputs["input_ids"][0, :prompt_length].detach().cpu().tolist()
    decoded_tokens = decode_token_ids(tokenizer, input_token_ids)
    offset_rows = prompt_offset_metadata(
        tokenizer,
        example["prompt"],
        max_length=max_length,
        add_special_tokens=bool(example.get("tokenization_config", {}).get("prompt_add_special_tokens", True)),
    )
    offset_by_token_idx = {int(row["token_idx"]): row for row in offset_rows}

    capture = DeepCapture(selected_layers)
    capture.attach(model)
    routing_context = (
        restricted_expert_mode(model, allowed_experts=run_spec.allowed_experts)
        if run_spec.apply_restricted_routing
        else nullcontext()
    )
    with routing_context:
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            capture.remove()

    assert_all_finite(f"outputs.hidden_states[{example['example_id']}]", outputs.hidden_states)
    hidden_state_rows = hidden_state_by_decoder_layer(outputs.hidden_states, selected_layers)
    deep_metadata_rows: list[dict[str, object]] = []
    token_feature_rows: list[dict[str, object]] = []
    expert_rows: list[dict[str, object]] = []
    deep_vector_rows: dict[str, list[np.ndarray]] = {}
    token_feature_vectors: dict[str, list[np.ndarray]] = {}
    hidden_size = int(model.config.hidden_size)

    for layer_idx in selected_layers:
        deep_vector_rows[f"hidden_state_layer_{layer_idx}"] = []
        deep_vector_rows[f"pre_router_layer_{layer_idx}"] = []
        deep_vector_rows[f"router_logits_layer_{layer_idx}"] = []
        deep_vector_rows[f"router_probs_layer_{layer_idx}"] = []
        deep_vector_rows[f"selected_expert_ids_layer_{layer_idx}"] = []
        deep_vector_rows[f"selected_router_weights_layer_{layer_idx}"] = []
        deep_vector_rows[f"raw_selected_expert_outputs_layer_{layer_idx}"] = []
        deep_vector_rows[f"weighted_selected_expert_outputs_layer_{layer_idx}"] = []
        deep_vector_rows[f"mixture_output_layer_{layer_idx}"] = []
        token_feature_vectors[f"pre_router_layer_{layer_idx}"] = []
        token_feature_vectors[f"hidden_state_layer_{layer_idx}"] = []

    prompt_router_token_summaries = routing_record["prompt_router_token_summaries_by_layer"]
    prompt_router_logits = routing_record["prompt_router_logits_by_layer"]
    prompt_router_probs = routing_record["prompt_router_probs_by_layer"]

    for layer_idx in selected_layers:
        layer_summary = prompt_router_token_summaries[layer_idx]
        logits_tensor = torch.as_tensor(prompt_router_logits[layer_idx])
        probs_tensor = torch.as_tensor(prompt_router_probs[layer_idx])
        pre_router_tensor = capture.pre_router_states[layer_idx]
        hidden_state_tensor = hidden_state_rows[layer_idx]
        assert_all_finite(f"logits.layer_{layer_idx}", logits_tensor)
        assert_all_finite(f"probs.layer_{layer_idx}", probs_tensor)
        assert_all_finite(f"pre_router.layer_{layer_idx}", pre_router_tensor)
        assert_all_finite(f"hidden_state.layer_{layer_idx}", hidden_state_tensor)
        top1_expert_ids = per_token_tensor(layer_summary["top1_expert_ids"], prompt_length)
        top2_expert_ids = per_token_tensor(layer_summary["top2_expert_ids"], prompt_length)
        top1_probs = per_token_tensor(layer_summary["top1_probs"], prompt_length)
        top2_probs = per_token_tensor(layer_summary["top2_probs"], prompt_length)
        top1_top2_margin = per_token_tensor(layer_summary["top1_top2_margin"], prompt_length)
        token_entropy = per_token_tensor(layer_summary["token_entropy"], prompt_length)
        selected_expert_prob_mass = per_token_tensor(layer_summary["selected_expert_prob_mass"], prompt_length)

        if logits_tensor.ndim == 3:
            logits_tensor = logits_tensor[0]
        if probs_tensor.ndim == 3:
            probs_tensor = probs_tensor[0]
        if pre_router_tensor.ndim == 3:
            pre_router_tensor = pre_router_tensor[0]
        if hidden_state_tensor.ndim == 3:
            hidden_state_tensor = hidden_state_tensor[0]

        logits_tensor = logits_tensor[:prompt_length, :]
        probs_tensor = probs_tensor[:prompt_length, :]
        pre_router_tensor = pre_router_tensor[:prompt_length, :]
        hidden_state_tensor = hidden_state_tensor[:prompt_length, :]
        contrib_records = capture.contrib_records.get(layer_idx, [])[:prompt_length]
        contrib_vectors = capture.contrib_vectors.get(layer_idx, [])[:prompt_length]

        row_idx_within_layer = 0
        for token_idx in range(prompt_length):
            offset_info = offset_by_token_idx.get(token_idx, {})
            top1_expert = int(top1_expert_ids[token_idx].item())
            top2_expert = int(top2_expert_ids[token_idx].item())
            top1_prob = float(top1_probs[token_idx].item())
            top2_prob = float(top2_probs[token_idx].item())
            margin = float(top1_top2_margin[token_idx].item())
            entropy = float(token_entropy[token_idx].item())
            selected_mass = float(selected_expert_prob_mass[token_idx].item())

            contrib_record = contrib_records[token_idx]
            contrib_vector = contrib_vectors[token_idx]
            assert_all_finite(f"contrib_record.layer_{layer_idx}.token_{token_idx}", contrib_record)
            assert_all_finite(f"contrib_vector.layer_{layer_idx}.token_{token_idx}", contrib_vector)

            base_row = {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example["language"],
                "run_label": run_spec.label,
                "layer": int(layer_idx),
                "row_idx_within_layer": row_idx_within_layer,
                "token_idx": int(token_idx),
                "token_id": int(input_token_ids[token_idx]),
                "token_text": decoded_tokens[token_idx],
                "top1_expert": top1_expert,
                "top2_expert": top2_expert,
                "top1_prob": top1_prob,
                "top2_prob": top2_prob,
                "top1_top2_margin": margin,
                "token_entropy": entropy,
                "selected_expert_prob_mass": selected_mass,
                "offset_start": offset_info.get("offset_start"),
                "offset_end": offset_info.get("offset_end"),
                "word_idx": offset_info.get("word_idx"),
                "word_text": offset_info.get("word_text"),
                "word_subtoken_count": offset_info.get("word_subtoken_count"),
                "fragmentation_bucket": offset_info.get("fragmentation_bucket", "unknown"),
                "predicted_output_text": routing_record.get("predicted_output_text"),
                "reference_answer": routing_record.get("reference_answer"),
                "scoring_mode": routing_record.get("scoring_mode", example.get("scoring_mode", "qa")),
                "selected_expert_ids": contrib_record["selected_expert_ids"],
                "selected_router_weights": contrib_record["selected_router_weights"],
                "contribution_shares": contrib_record["contribution_shares"],
                "raw_expert_output_norms": contrib_record["raw_expert_output_norms"],
                "weighted_expert_output_norms": contrib_record["weighted_expert_output_norms"],
                "mixture_output_norm": contrib_record["mixture_output_norm"],
                "mixture_alignment_ratio": contrib_record["mixture_alignment_ratio"],
                "top1_top2_raw_output_cosine": contrib_record["top1_top2_raw_output_cosine"],
                "top1_top2_weighted_output_cosine": contrib_record["top1_top2_weighted_output_cosine"],
                "top1_ablation_delta_ratio": contrib_record["top1_ablation_delta_ratio"],
                "top2_ablation_delta_ratio": contrib_record["top2_ablation_delta_ratio"],
            }
            deep_metadata_rows.append(base_row)
            token_feature_rows.append(base_row)
            expert_rows.append(
                {
                    "example_id": example["example_id"],
                    "dataset_name": example["dataset_name"],
                    "language": example["language"],
                    "run_label": run_spec.label,
                    "token_id": int(input_token_ids[token_idx]),
                    "token_text": decoded_tokens[token_idx],
                    **contrib_record,
                }
            )

            deep_vector_rows[f"hidden_state_layer_{layer_idx}"].append(hidden_state_tensor[token_idx].numpy())
            deep_vector_rows[f"pre_router_layer_{layer_idx}"].append(pre_router_tensor[token_idx].numpy())
            deep_vector_rows[f"router_logits_layer_{layer_idx}"].append(logits_tensor[token_idx].detach().cpu().numpy())
            deep_vector_rows[f"router_probs_layer_{layer_idx}"].append(probs_tensor[token_idx].detach().cpu().numpy())
            deep_vector_rows[f"selected_expert_ids_layer_{layer_idx}"].append(
                np.asarray(contrib_vector["selected_expert_ids"], dtype=np.int64)
            )
            deep_vector_rows[f"selected_router_weights_layer_{layer_idx}"].append(
                np.asarray(contrib_vector["selected_router_weights"], dtype=np.float32)
            )
            deep_vector_rows[f"raw_selected_expert_outputs_layer_{layer_idx}"].append(
                np.asarray(contrib_vector["raw_selected_expert_outputs"], dtype=np.float32)
            )
            deep_vector_rows[f"weighted_selected_expert_outputs_layer_{layer_idx}"].append(
                np.asarray(contrib_vector["weighted_selected_expert_outputs"], dtype=np.float32)
            )
            deep_vector_rows[f"mixture_output_layer_{layer_idx}"].append(
                np.asarray(contrib_vector["mixture_output"], dtype=np.float32)
            )
            token_feature_vectors[f"pre_router_layer_{layer_idx}"].append(pre_router_tensor[token_idx].numpy())
            token_feature_vectors[f"hidden_state_layer_{layer_idx}"].append(hidden_state_tensor[token_idx].numpy())
            row_idx_within_layer += 1

        assert_all_finite(f"token_feature_rows.layer_{layer_idx}", token_feature_rows)
        assert_all_finite(f"expert_rows.layer_{layer_idx}", expert_rows)

    for layer_idx in selected_layers:
        if not deep_vector_rows[f"selected_expert_ids_layer_{layer_idx}"]:
            deep_vector_rows[f"selected_expert_ids_layer_{layer_idx}"] = [np.zeros((0,), dtype=np.int64)]
            deep_vector_rows[f"selected_router_weights_layer_{layer_idx}"] = [np.zeros((0,), dtype=np.float32)]
            deep_vector_rows[f"raw_selected_expert_outputs_layer_{layer_idx}"] = [np.zeros((0, hidden_size), dtype=np.float32)]
            deep_vector_rows[f"weighted_selected_expert_outputs_layer_{layer_idx}"] = [np.zeros((0, hidden_size), dtype=np.float32)]
            deep_vector_rows[f"mixture_output_layer_{layer_idx}"] = [np.zeros((hidden_size,), dtype=np.float32)]
        assert_all_finite(f"deep_vector_rows.layer_{layer_idx}", {key: value for key, value in deep_vector_rows.items() if f"layer_{layer_idx}" in key})
        assert_all_finite(f"token_feature_vectors.layer_{layer_idx}", {key: value for key, value in token_feature_vectors.items() if f"layer_{layer_idx}" in key})

    return routing_record, token_feature_rows, token_feature_vectors, expert_rows, deep_vector_rows


def stack_or_empty(values: list[np.ndarray], trailing_shape: tuple[int, ...], dtype) -> np.ndarray:
    if values:
        return np.stack(values).astype(dtype)
    return np.zeros((0, *trailing_shape), dtype=dtype)


def per_token_tensor(values: Any, prompt_length: int) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if tensor.ndim == 2:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(f"Expected per-token tensor with shape [seq] or [1, seq], got {tuple(tensor.shape)}")
    return tensor[:prompt_length]


def parse_representation_sources(raw_value: str) -> list[str]:
    sources = [part.strip() for part in raw_value.split(",") if part.strip()]
    allowed = {
        "embedding",
        "hidden_state",
        "pre_router",
        "router_probs",
        "pre_router_plus_router_probs",
        "hidden_state_plus_router_probs",
    }
    invalid = [source for source in sources if source not in allowed]
    if invalid:
        raise ValueError(f"Unsupported representation sources: {', '.join(invalid)}")
    if not sources:
        raise ValueError("Provide at least one representation source.")
    return sources


def select_hidden_state_layers(hidden_states, selected_layers: list[int]) -> dict[int, torch.Tensor]:
    num_layers = len(hidden_states)
    result: dict[int, torch.Tensor] = {}
    for layer_idx in selected_layers:
        normalized_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if normalized_idx < 0 or normalized_idx >= num_layers:
            raise ValueError(f"Hidden-state layer index {layer_idx} is out of range for {num_layers} tensors.")
        if normalized_idx not in result:
            result[normalized_idx] = hidden_states[normalized_idx]
    return result


def reduce_tokens(tensor: torch.Tensor, prompt_length: int, position_policy: str) -> torch.Tensor:
    prompt_tensor = tensor[0, :prompt_length, :]
    if position_policy == "last_prompt_token":
        return prompt_tensor[-1].detach().cpu().float()
    if position_policy == "mean_prompt":
        return prompt_tensor.mean(dim=0).detach().cpu().float()
    raise ValueError(f"Unsupported position policy `{position_policy}`.")


def compute_alignment_scores(state: torch.Tensor, weight_matrix: torch.Tensor, metric: str) -> torch.Tensor:
    state = state.float()
    weights = weight_matrix.float()
    if metric == "dot":
        return torch.mv(weights, state)
    if metric == "cosine":
        normalized_state = state / torch.linalg.vector_norm(state).clamp_min(1e-9)
        normalized_weights = weights / torch.linalg.vector_norm(weights, dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.mv(normalized_weights, normalized_state)
    raise ValueError(f"Unsupported alignment metric `{metric}`.")


def alignment_entropy(scores: torch.Tensor) -> float:
    probs = torch.softmax(scores, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum()
    return float(entropy.item())


def summarize_router_choice(
    router_probs: torch.Tensor,
    top_k_index: torch.Tensor | None,
    prompt_length: int,
    position_policy: str,
) -> dict[str, object]:
    prompt_probs = router_probs[0, :prompt_length, :].detach().cpu().float()
    if position_policy == "last_prompt_token":
        selected_probs = prompt_probs[-1]
        selected_topk = top_k_index[prompt_length - 1].detach().cpu().tolist() if top_k_index is not None else []
    elif position_policy == "mean_prompt":
        selected_probs = prompt_probs.mean(dim=0)
        selected_topk = []
    else:
        raise ValueError(f"Unsupported position policy `{position_policy}`.")
    top_values, top_indices = torch.topk(selected_probs, k=min(2, selected_probs.shape[-1]), dim=-1)
    top1 = int(top_indices[0].item())
    top2 = int(top_indices[1].item()) if top_indices.shape[0] > 1 else None
    return {
        "actual_top1_expert": top1,
        "actual_top2_expert": top2,
        "actual_topk_experts": selected_topk,
        "actual_top1_prob": float(top_values[0].item()),
        "actual_top2_prob": float(top_values[1].item()) if top_values.shape[0] > 1 else 0.0,
    }


class PromptPreRouterCapture:
    def __init__(self, selected_layers: list[int]):
        self.selected_layers = selected_layers
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, args):
            self.outputs[layer_idx] = args[0].detach().cpu().float()
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PromptRouterCapture:
    def __init__(self, selected_layers: list[int]):
        self.selected_layers = selected_layers
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            router_probs = output[0].detach().cpu().float()
            self.outputs[layer_idx] = router_probs.unsqueeze(0) if router_probs.ndim == 2 else router_probs
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.gate.register_forward_hook(self._make_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RouterDirectionCapture:
    def __init__(self, selected_layers: list[int], prompt_length: int, position_policy: str):
        self.selected_layers = selected_layers
        self.prompt_length = prompt_length
        self.position_policy = position_policy
        self.pre_router_states: dict[int, torch.Tensor] = {}
        self.router_outputs: dict[int, dict[str, torch.Tensor | None]] = {}
        self.handles: list[Any] = []

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args):
            self.pre_router_states[layer_idx] = reduce_tokens(args[0], self.prompt_length, self.position_policy)
        return hook

    def _make_router_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            router_probs = output[0].detach().cpu().float()
            top_k_index = output[2].detach().cpu() if len(output) > 2 else None
            seq_len = router_probs.shape[0]
            reshaped_probs = router_probs.unsqueeze(0) if router_probs.ndim == 2 else router_probs
            reshaped_topk = top_k_index if top_k_index is None else top_k_index.reshape(seq_len, -1)
            self.router_outputs[layer_idx] = {
                "router_probs": reshaped_probs,
                "top_k_index": reshaped_topk,
            }
        return hook

    def attach(self, model) -> None:
        layers = list(iter_flex_olmo_layers(model))
        for layer_idx in self.selected_layers:
            self.handles.append(layers[layer_idx].mlp.register_forward_pre_hook(self._make_pre_hook(layer_idx)))
            self.handles.append(layers[layer_idx].mlp.gate.register_forward_hook(self._make_router_hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_router_direction_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for record in records:
        key = (str(record["dataset_name"]), str(record.get("language", "unknown")), int(record["layer"]))
        grouped[key].append(record)
    summaries: list[dict] = []
    for (dataset_name, language, layer), items in sorted(grouped.items()):
        summaries.append(
            {
                "record_type": "router_direction_summary",
                "dataset_name": dataset_name,
                "language": language,
                "layer": layer,
                "num_examples": len(items),
                "mean_top1_alignment": sum(float(item["top1_alignment"]) for item in items) / len(items),
                "mean_top2_alignment": sum(float(item["top2_alignment"]) for item in items) / len(items),
                "mean_alignment_margin": sum(float(item["alignment_margin"]) for item in items) / len(items),
                "mean_alignment_entropy": sum(float(item["alignment_entropy"]) for item in items) / len(items),
                "top1_agreement_rate": sum(1.0 for item in items if item.get("agreement_top1")) / len(items),
            }
        )
    return summaries


def capture_dataset_latents(
    model,
    tokenizer,
    examples: list[dict],
    hidden_state_layers: list[int],
    pre_router_layers: list[int],
    representation_sources: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    vectors: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {}
    if "embedding" in representation_sources:
        vectors["embedding"] = {-1: {"mean": [], "last": []}}
    if "hidden_state" in representation_sources:
        vectors["hidden_state"] = {layer: {"mean": [], "last": []} for layer in hidden_state_layers}
    if "pre_router" in representation_sources:
        vectors["pre_router"] = {layer: {"mean": [], "last": []} for layer in pre_router_layers}
    if "router_probs" in representation_sources:
        vectors["router_probs"] = {layer: {"mean": [], "last": []} for layer in pre_router_layers}
    if "pre_router_plus_router_probs" in representation_sources:
        vectors["pre_router_plus_router_probs"] = {layer: {"mean": [], "last": []} for layer in pre_router_layers}
    if "hidden_state_plus_router_probs" in representation_sources:
        vectors["hidden_state_plus_router_probs"] = {layer: {"mean": [], "last": []} for layer in hidden_state_layers}
    metadata: list[dict] = []
    capture_pre_router = "pre_router" in representation_sources
    capture_router_probs = any(
        source in representation_sources
        for source in ("router_probs", "pre_router_plus_router_probs", "hidden_state_plus_router_probs")
    )

    for example in examples:
        inputs = encode_prompt(
            tokenizer,
            example["prompt"],
            max_length=max_length,
            device=device,
            add_special_tokens=bool(example.get("tokenization_config", {}).get("prompt_add_special_tokens", True)),
        )
        attention_mask = inputs.get("attention_mask")
        seq_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(inputs["input_ids"].shape[-1])

        if "embedding" in representation_sources:
            with torch.no_grad():
                embedding_tensor = model.get_input_embeddings()(inputs["input_ids"])[0, :seq_len, :].detach().cpu().float()
            vectors["embedding"][-1]["mean"].append(embedding_tensor.mean(dim=0).numpy())
            vectors["embedding"][-1]["last"].append(embedding_tensor[-1].numpy())

        pre_router_capture = PromptPreRouterCapture(pre_router_layers) if capture_pre_router else None
        router_capture_layers = sorted(set(pre_router_layers) | set(hidden_state_layers))
        router_capture = PromptRouterCapture(router_capture_layers) if capture_router_probs else None
        if pre_router_capture is not None:
            pre_router_capture.attach(model)
        if router_capture is not None:
            router_capture.attach(model)
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states="hidden_state" in representation_sources,
                    use_cache=False,
                )
        finally:
            if pre_router_capture is not None:
                pre_router_capture.remove()
            if router_capture is not None:
                router_capture.remove()

        metadata.append(
            {
                "example_id": example["example_id"],
                "dataset_name": example["dataset_name"],
                "language": example["language"],
                "question": example.get("question"),
                "num_input_tokens": seq_len,
            }
        )

        if "hidden_state" in representation_sources:
            selected_hidden_states = select_hidden_state_layers(outputs.hidden_states, hidden_state_layers)
            for layer_idx, tensor in selected_hidden_states.items():
                token_states = tensor[0].detach().cpu().float()
                vectors["hidden_state"][layer_idx]["mean"].append(token_states.mean(dim=0).numpy())
                vectors["hidden_state"][layer_idx]["last"].append(token_states[-1].numpy())
                if "hidden_state_plus_router_probs" in representation_sources and router_capture is not None:
                    router_tensor = router_capture.outputs[layer_idx][0, :seq_len, :].detach().cpu().float()
                    mean_combined = torch.cat([token_states.mean(dim=0), router_tensor.mean(dim=0)], dim=0)
                    last_combined = torch.cat([token_states[-1], router_tensor[-1]], dim=0)
                    vectors["hidden_state_plus_router_probs"][layer_idx]["mean"].append(mean_combined.numpy())
                    vectors["hidden_state_plus_router_probs"][layer_idx]["last"].append(last_combined.numpy())

        if capture_pre_router and pre_router_capture is not None:
            for layer_idx, tensor in pre_router_capture.outputs.items():
                token_states = tensor[0, :seq_len, :].detach().cpu().float()
                vectors["pre_router"][layer_idx]["mean"].append(token_states.mean(dim=0).numpy())
                vectors["pre_router"][layer_idx]["last"].append(token_states[-1].numpy())
                if "pre_router_plus_router_probs" in representation_sources and router_capture is not None:
                    router_tensor = router_capture.outputs[layer_idx][0, :seq_len, :].detach().cpu().float()
                    mean_combined = torch.cat([token_states.mean(dim=0), router_tensor.mean(dim=0)], dim=0)
                    last_combined = torch.cat([token_states[-1], router_tensor[-1]], dim=0)
                    vectors["pre_router_plus_router_probs"][layer_idx]["mean"].append(mean_combined.numpy())
                    vectors["pre_router_plus_router_probs"][layer_idx]["last"].append(last_combined.numpy())

        if "router_probs" in representation_sources and router_capture is not None:
            for layer_idx in pre_router_layers:
                router_tensor = router_capture.outputs[layer_idx][0, :seq_len, :].detach().cpu().float()
                vectors["router_probs"][layer_idx]["mean"].append(router_tensor.mean(dim=0).numpy())
                vectors["router_probs"][layer_idx]["last"].append(router_tensor[-1].numpy())

    arrays: dict[str, np.ndarray] = {}
    for source_name, source_layers in sorted(vectors.items()):
        for layer_idx in sorted(source_layers):
            arrays[f"{source_name}_layer_{layer_idx}_mean"] = np.stack(source_layers[layer_idx]["mean"], axis=0)
            arrays[f"{source_name}_layer_{layer_idx}_last"] = np.stack(source_layers[layer_idx]["last"], axis=0)
    return arrays, metadata


def capture_router_direction_records(
    model,
    tokenizer,
    model_name: str,
    model_path: str,
    examples: list[dict],
    selected_layers: list[int],
    max_length: int,
    device: torch.device,
    position_policy: str,
    alignment_metric: str,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    layers = list(iter_flex_olmo_layers(model))
    router_weights = {
        f"layer_{layer_idx}_weights": layers[layer_idx].mlp.gate.weight.detach().cpu().float().numpy()
        for layer_idx in selected_layers
    }
    records: list[dict] = []
    for example in examples:
        inputs = encode_prompt(
            tokenizer,
            example["prompt"],
            max_length=max_length,
            device=device,
            add_special_tokens=bool(example.get("tokenization_config", {}).get("prompt_add_special_tokens", True)),
        )
        prompt_length = int(inputs["input_ids"].shape[-1])
        capture = RouterDirectionCapture(selected_layers, prompt_length=prompt_length, position_policy=position_policy)
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

        for layer_idx in selected_layers:
            state = capture.pre_router_states[layer_idx]
            weight_matrix = torch.from_numpy(router_weights[f"layer_{layer_idx}_weights"])
            scores = compute_alignment_scores(state, weight_matrix, metric=alignment_metric)
            top_scores, top_indices = torch.topk(scores, k=min(2, scores.shape[0]), dim=-1)
            routing_info = summarize_router_choice(
                router_probs=capture.router_outputs[layer_idx]["router_probs"],
                top_k_index=capture.router_outputs[layer_idx]["top_k_index"],
                prompt_length=prompt_length,
                position_policy=position_policy,
            )
            top1_expert = int(top_indices[0].item())
            top2_expert = int(top_indices[1].item()) if top_indices.shape[0] > 1 else None
            actual_top1 = routing_info["actual_top1_expert"]
            records.append(
                {
                    "record_type": "router_direction_record",
                    "example_id": example["example_id"],
                    "dataset_name": example["dataset_name"],
                    "language": example["language"],
                    "model_name": model_name,
                    "model_path": model_path,
                    "layer": layer_idx,
                    "alignment_metric": alignment_metric,
                    "position_policy": position_policy,
                    "top1_aligned_expert": top1_expert,
                    "top2_aligned_expert": top2_expert,
                    "top1_alignment": float(top_scores[0].item()),
                    "top2_alignment": float(top_scores[1].item()) if top_scores.shape[0] > 1 else 0.0,
                    "alignment_margin": float((top_scores[0] - top_scores[1]).item()) if top_scores.shape[0] > 1 else 0.0,
                    "alignment_entropy": alignment_entropy(scores),
                    "actual_topk_experts": routing_info["actual_topk_experts"],
                    "actual_top1_expert": actual_top1,
                    "actual_top2_expert": routing_info["actual_top2_expert"],
                    "actual_top1_prob": routing_info["actual_top1_prob"],
                    "actual_top2_prob": routing_info["actual_top2_prob"],
                    "agreement_top1": bool(actual_top1 == top1_expert) if actual_top1 is not None else None,
                }
            )
    return records, router_weights


def main() -> int:
    args = parse_args()
    selected_datasets = None
    if args.datasets:
        selected_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}
    device = resolve_device(args.device)
    model_path = resolve_model_path(args)
    model_name = args.model_name or Path(model_path).name
    print(f"[deep_base] model={model_name}")
    print(f"[deep_base] model_path={model_path}")
    print(
        f"[deep_base] device={device} requested_dtype={args.dtype} "
        f"selected_layers={args.selected_layers} save_raw_artifacts={args.save_raw_artifacts}"
    )
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        dtype_name=args.dtype,
    )
    print("[deep_base] tokenizer loaded")
    selected_layers = parse_decoder_layers(args.selected_layers, int(model.config.num_hidden_layers))
    print(f"[deep_base] resolved_layers={selected_layers}")
    router_direction_selected_layers = parse_decoder_layers(
        args.router_direction_selected_layers or args.selected_layers,
        int(model.config.num_hidden_layers),
    )
    latent_space_hidden_layers = parse_hidden_state_layers(
        args.latent_space_selected_layers or args.selected_layers,
        num_hidden_layers=int(model.config.num_hidden_layers),
    )
    latent_space_pre_router_layers = parse_decoder_layers(
        args.latent_space_selected_layers or args.selected_layers,
        int(model.config.num_hidden_layers),
    )
    latent_space_sources = parse_representation_sources(args.latent_space_representation_sources)

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
        raise ValueError("No datasets were selected from the manifest.")
    print(f"[deep_base] manifest_entries={len(manifest_entries)}")

    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)
    router_direction_output_root = (
        Path(args.router_direction_output_root) / model_name if args.router_direction_output_root else None
    )
    latent_space_output_root = (
        Path(args.latent_space_output_root) / model_name if args.latent_space_output_root else None
    )
    if router_direction_output_root is not None:
        router_direction_output_root.mkdir(parents=True, exist_ok=True)
    if latent_space_output_root is not None:
        latent_space_output_root.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": selected_layers,
        "routing_run_mode": args.routing_run_mode,
        "datasets": {},
    }
    router_direction_suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": router_direction_selected_layers,
        "alignment_metric": args.router_direction_alignment_metric,
        "position_policy": args.router_direction_position_policy,
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "router_weights_path": (
            str(router_direction_output_root / "router_weights.npz") if router_direction_output_root else None
        ),
        "datasets": {},
    }
    latent_space_suite_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "selected_layers": {
            "hidden_state": latent_space_hidden_layers,
            "pre_router": latent_space_pre_router_layers,
        },
        "representation_sources": latent_space_sources,
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "datasets": {},
    }

    router_weights = None
    if router_direction_output_root is not None:
        router_weights = {
            f"layer_{layer_idx}_weights": list(iter_flex_olmo_layers(model))[layer_idx].mlp.gate.weight.detach().cpu().float().numpy()
            for layer_idx in router_direction_selected_layers
        }
        np.savez_compressed(router_direction_output_root / "router_weights.npz", **router_weights)

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
        print(f"[deep_base] dataset={dataset_name} examples={len(examples)}")

        if router_direction_output_root is not None:
            dataset_router_direction_records, _ = capture_router_direction_records(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                model_path=model_path,
                examples=examples,
                selected_layers=router_direction_selected_layers,
                max_length=args.max_length,
                device=device,
                position_policy=args.router_direction_position_policy,
                alignment_metric=args.router_direction_alignment_metric,
            )
            router_direction_dataset_dir = router_direction_output_root / dataset_name
            router_direction_dataset_dir.mkdir(parents=True, exist_ok=True)
            router_direction_records_path = router_direction_dataset_dir / "router_direction_records.jsonl"
            router_direction_summary_path = router_direction_dataset_dir / "router_direction_summary.jsonl"
            write_jsonl(dataset_router_direction_records, router_direction_records_path, sort_keys=False)
            write_jsonl(
                build_router_direction_summary(dataset_router_direction_records),
                router_direction_summary_path,
                sort_keys=False,
            )
            router_direction_manifest = {
                "dataset_name": dataset_name,
                "num_examples": len(examples),
                "records_path": str(router_direction_records_path),
                "summary_path": str(router_direction_summary_path),
                "selected_layers": router_direction_selected_layers,
                "alignment_metric": args.router_direction_alignment_metric,
                "position_policy": args.router_direction_position_policy,
            }
            (router_direction_dataset_dir / "run_manifest.json").write_text(
                json.dumps(router_direction_manifest, indent=2),
                encoding="utf-8",
            )
            router_direction_suite_manifest["datasets"][dataset_name] = router_direction_manifest

        if latent_space_output_root is not None:
            latent_arrays, latent_metadata = capture_dataset_latents(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                hidden_state_layers=latent_space_hidden_layers,
                pre_router_layers=latent_space_pre_router_layers,
                representation_sources=latent_space_sources,
                max_length=args.max_length,
                device=device,
            )
            latent_dataset_dir = latent_space_output_root / dataset_name
            latent_dataset_dir.mkdir(parents=True, exist_ok=True)
            latent_npz_path = latent_dataset_dir / "prompt_latents.npz"
            latent_metadata_path = latent_dataset_dir / "metadata.jsonl"
            np.savez_compressed(latent_npz_path, **latent_arrays)
            write_jsonl(latent_metadata, latent_metadata_path, sort_keys=False)
            latent_manifest = {
                "dataset_name": dataset_name,
                "num_examples": len(latent_metadata),
                "selected_layers": {
                    "hidden_state": latent_space_hidden_layers,
                    "pre_router": latent_space_pre_router_layers,
                },
                "representation_sources": latent_space_sources,
                "npz_path": str(latent_npz_path),
                "metadata_path": str(latent_metadata_path),
                "arrays": {key: list(value.shape) for key, value in latent_arrays.items()},
            }
            (latent_dataset_dir / "run_manifest.json").write_text(
                json.dumps(latent_manifest, indent=2),
                encoding="utf-8",
            )
            latent_space_suite_manifest["datasets"][dataset_name] = latent_manifest

        dataset_dir = model_output_root / sanitize_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        suite_manifest["datasets"][dataset_name] = {"path": str(dataset_entry["path"]), "num_examples": len(examples), "runs": {}}

        for run_spec in run_specs:
            print(f"[deep_base] run_label={run_spec.label} dataset={dataset_name}")
            run_dir = dataset_dir / sanitize_name(run_spec.label)
            run_dir.mkdir(parents=True, exist_ok=True)

            routing_records: list[dict[str, Any]] = []
            token_feature_rows: list[dict[str, object]] = []
            expert_rows: list[dict[str, object]] = []
            deep_vector_rows: dict[str, list[np.ndarray]] = {}
            token_feature_vectors: dict[str, list[np.ndarray]] = {
                f"{source}_layer_{layer_idx}": []
                for source in ("pre_router", "hidden_state")
                for layer_idx in selected_layers
            }
            if args.save_raw_artifacts:
                deep_vector_rows = {f"hidden_state_layer_{layer_idx}": [] for layer_idx in selected_layers}
                for layer_idx in selected_layers:
                    deep_vector_rows[f"pre_router_layer_{layer_idx}"] = []
                    deep_vector_rows[f"router_logits_layer_{layer_idx}"] = []
                    deep_vector_rows[f"router_probs_layer_{layer_idx}"] = []
                    deep_vector_rows[f"selected_expert_ids_layer_{layer_idx}"] = []
                    deep_vector_rows[f"selected_router_weights_layer_{layer_idx}"] = []
                    deep_vector_rows[f"raw_selected_expert_outputs_layer_{layer_idx}"] = []
                    deep_vector_rows[f"weighted_selected_expert_outputs_layer_{layer_idx}"] = []
                    deep_vector_rows[f"mixture_output_layer_{layer_idx}"] = []

            for example_idx, example in enumerate(examples, start=1):
                if example_idx == 1 or example_idx == len(examples) or example_idx % 10 == 0:
                    print(
                        f"[deep_base] dataset={dataset_name} run_label={run_spec.label} "
                        f"example={example_idx}/{len(examples)} example_id={example['example_id']}"
                    )
                routing_record, example_token_rows, example_token_vectors, example_expert_rows, example_deep_vectors = collect_deep_example(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    run_spec=run_spec,
                    selected_layers=selected_layers,
                    max_length=args.max_length,
                    default_max_new_tokens=args.default_max_new_tokens,
                    device=device,
                    save_raw_artifacts=args.save_raw_artifacts,
                )
                routing_records.append(routing_record)
                token_feature_rows.extend(example_token_rows)
                expert_rows.extend(example_expert_rows)
                for key, values in example_token_vectors.items():
                    token_feature_vectors[key].extend(values)
                if args.save_raw_artifacts:
                    for key, values in example_deep_vectors.items():
                        deep_vector_rows[key].extend(values)

            print(
                f"[deep_base] completed dataset={dataset_name} run_label={run_spec.label} "
                f"examples={len(examples)}"
            )

            routing_summaries = summarize_routing_records(routing_records)
            routing_analysis_records, _routing_aggregate = aggregate_routing_analysis(routing_records)

            routing_records_path = run_dir / "routing_records.jsonl"
            routing_summary_path = run_dir / "routing_summary.jsonl"
            routing_analysis_path = run_dir / "routing_analysis.jsonl"
            print(f"[deep_base] writing {routing_records_path.name}")
            write_jsonl(routing_records, routing_records_path, sort_keys=False)
            print(f"[deep_base] writing {routing_summary_path.name}")
            write_jsonl(routing_summaries, routing_summary_path, sort_keys=False)
            print(f"[deep_base] writing {routing_analysis_path.name}")
            write_jsonl(routing_analysis_records, routing_analysis_path, sort_keys=False)
            assert_all_finite(f"routing_records[{dataset_name}.{run_spec.label}]", routing_records)
            assert_all_finite(f"routing_summaries[{dataset_name}.{run_spec.label}]", routing_summaries)
            assert_all_finite(f"routing_analysis_records[{dataset_name}.{run_spec.label}]", routing_analysis_records)
            token_feature_metadata_path = run_dir / "token_feature_metadata.jsonl"
            token_feature_vectors_path = run_dir / "token_feature_vectors.npz"
            token_feature_manifest_path = run_dir / "token_feature_manifest.json"
            expert_records_path = run_dir / "expert_contribution_records.jsonl"
            expert_manifest_path = run_dir / "expert_contribution_manifest.json"
            print(f"[deep_base] writing {token_feature_metadata_path.name}")
            write_jsonl(token_feature_rows, token_feature_metadata_path, sort_keys=False)
            print(f"[deep_base] writing {expert_records_path.name}")
            write_jsonl(expert_rows, expert_records_path, sort_keys=False)
            assert_all_finite(f"token_feature_rows[{dataset_name}.{run_spec.label}]", token_feature_rows)
            assert_all_finite(f"expert_rows[{dataset_name}.{run_spec.label}]", expert_rows)

            print(f"[deep_base] saving {token_feature_vectors_path.name}")
            assert_all_finite(f"token_feature_vectors[{dataset_name}.{run_spec.label}]", token_feature_vectors)
            np.savez(
                token_feature_vectors_path,
                **{
                    key: stack_or_empty(values, (model.config.hidden_size,), np.float32)
                    for key, values in token_feature_vectors.items()
                },
            )
            print(f"[deep_base] wrote {token_feature_vectors_path.name}")

            token_feature_manifest = {
                "model_name": model_name,
                "model_path": model_path,
                "dataset_name": dataset_name,
                "run_label": run_spec.label,
                "selected_layers": selected_layers,
                "representation_sources": ["pre_router", "hidden_state"],
                "num_examples": len(examples),
                "num_token_rows": len(token_feature_rows),
                "vectors_path": str(token_feature_vectors_path),
                "metadata_path": str(token_feature_metadata_path),
                "row_counts_by_layer": {
                    str(layer_idx): sum(1 for row in token_feature_rows if int(row["layer"]) == layer_idx)
                    for layer_idx in selected_layers
                },
            }
            print(f"[deep_base] writing {token_feature_manifest_path.name}")
            token_feature_manifest_path.write_text(
                json.dumps(token_feature_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            expert_manifest = {
                "model_name": model_name,
                "model_path": model_path,
                "dataset_name": dataset_name,
                "run_label": run_spec.label,
                "selected_layers": selected_layers,
                "num_examples": len(examples),
                "num_token_rows": len(expert_rows),
                "records_path": str(expert_records_path),
            }
            print(f"[deep_base] writing {expert_manifest_path.name}")
            expert_manifest_path.write_text(json.dumps(expert_manifest, indent=2, sort_keys=True), encoding="utf-8")

            if args.save_raw_artifacts:
                deep_metadata_path = run_dir / "deep_base_metadata.jsonl"
                deep_vectors_path = run_dir / "deep_base_vectors.npz"
                deep_manifest_path = run_dir / "deep_base_manifest.json"

                print(f"[deep_base] writing {deep_metadata_path.name}")
                write_jsonl(token_feature_rows, deep_metadata_path, sort_keys=False)
                print(f"[deep_base] saving {deep_vectors_path.name}")
                assert_all_finite(f"deep_vector_rows[{dataset_name}.{run_spec.label}]", deep_vector_rows)
                np.savez(
                    deep_vectors_path,
                    **{
                        key: (
                            stack_or_empty(values, (model.config.hidden_size,), np.float32)
                            if key.startswith(("hidden_state_", "pre_router_", "mixture_output_"))
                            else stack_or_empty(values, (model.config.num_experts,), np.float32)
                            if key.startswith(("router_logits_", "router_probs_"))
                            else stack_or_empty(values, (0,), np.int64)
                            if key.startswith("selected_expert_ids_")
                            else stack_or_empty(values, (int(model.config.num_experts_per_tok),), np.float32)
                            if key.startswith("selected_router_weights_")
                            else stack_or_empty(values, (int(model.config.num_experts_per_tok), model.config.hidden_size), np.float32)
                        )
                        for key, values in deep_vector_rows.items()
                    },
                )
                print(f"[deep_base] wrote {deep_vectors_path.name}")

                deep_manifest = {
                    "model_name": model_name,
                    "model_path": model_path,
                    "dataset_name": dataset_name,
                    "run_label": run_spec.label,
                    "selected_layers": selected_layers,
                    "num_examples": len(examples),
                    "num_token_rows": len(token_feature_rows),
                    "deep_metadata_path": str(deep_metadata_path),
                    "deep_vectors_path": str(deep_vectors_path),
                    "routing_records_path": str(routing_records_path),
                    "routing_analysis_path": str(routing_analysis_path),
                    "token_feature_manifest_path": str(token_feature_manifest_path),
                    "expert_contribution_manifest_path": str(expert_manifest_path),
                }
                print(f"[deep_base] writing {deep_manifest_path.name}")
                deep_manifest_path.write_text(json.dumps(deep_manifest, indent=2, sort_keys=True), encoding="utf-8")
                print(f"[deep_base] saved run artifacts to {run_dir}")
            else:
                print(f"[deep_base] saved token/expert artifacts only for {run_dir}")

            suite_manifest["datasets"][dataset_name]["runs"][run_spec.label] = {
                "routing_records_path": str(routing_records_path),
                "routing_summary_path": str(routing_summary_path),
                "routing_analysis_path": str(routing_analysis_path),
                "raw_artifacts_saved": bool(args.save_raw_artifacts),
            }
            print(f"Captured deep base data for {model_name} on {dataset_name} / {run_spec.label}")

    suite_manifest_path = model_output_root / "deep_base_suite_manifest.json"
    suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote deep-base suite manifest to {suite_manifest_path}")
    if router_direction_output_root is not None:
        router_suite_manifest_path = router_direction_output_root / "router_direction_suite_manifest.json"
        router_suite_manifest_path.write_text(
            json.dumps(router_direction_suite_manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote router-direction suite manifest to {router_suite_manifest_path}")
    if latent_space_output_root is not None:
        latent_suite_manifest_path = latent_space_output_root / "latent_space_suite_manifest.json"
        latent_suite_manifest_path.write_text(
            json.dumps(latent_space_suite_manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote latent-space suite manifest to {latent_suite_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
