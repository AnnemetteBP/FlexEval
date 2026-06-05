from flexolmo_analysis.toolkit.utils.router_activity import (
    build_upset_data,
    build_layerwise_upset_data,
    build_layerwise_upset_data_from_router_logits,
    count_token_expert_combinations_by_layer,
    count_token_expert_combinations_by_layer_from_router_logits,
    compute_expert_sets,
    compute_topk_expert_sets,
    extract_expert_indices,
    extract_topk_expert_indices,
)
from flexolmo_analysis.toolkit.utils.euroeval_compat import (
    apply_chat_template_if_requested,
    build_scored_qa_record,
    normalize_examples_for_euroeval_compat,
    summarize_qa_records,
)
from flexolmo_analysis.toolkit.utils.tokenizers import load_tokenizer_with_known_fixes

__all__ = [
    "apply_chat_template_if_requested",
    "build_scored_qa_record",
    "build_upset_data",
    "build_layerwise_upset_data",
    "build_layerwise_upset_data_from_router_logits",
    "count_token_expert_combinations_by_layer",
    "count_token_expert_combinations_by_layer_from_router_logits",
    "compute_expert_sets",
    "compute_topk_expert_sets",
    "extract_expert_indices",
    "extract_topk_expert_indices",
    "load_tokenizer_with_known_fixes",
    "normalize_examples_for_euroeval_compat",
    "summarize_qa_records",
]
