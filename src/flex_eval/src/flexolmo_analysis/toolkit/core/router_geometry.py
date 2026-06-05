from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def cosine_similarity_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight_matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-9, None)
    return normalized @ normalized.T


def effective_rank(weight_matrix: np.ndarray) -> float:
    matrix = np.asarray(weight_matrix, dtype=np.float32)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = singular_values[singular_values > 0]
    if singular_values.size == 0:
        return 0.0
    probabilities = singular_values / singular_values.sum()
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-9, None))).sum()
    return float(np.exp(entropy))


def stable_rank(weight_matrix: np.ndarray) -> float:
    matrix = np.asarray(weight_matrix, dtype=np.float32)
    fro_norm_sq = float(np.square(matrix).sum())
    spectral_norm = float(np.linalg.svd(matrix, compute_uv=False).max(initial=0.0))
    if spectral_norm <= 0.0:
        return 0.0
    return fro_norm_sq / (spectral_norm ** 2)


def summarize_router_weight_matrix(weight_matrix: np.ndarray) -> dict[str, float | int]:
    matrix = np.asarray(weight_matrix, dtype=np.float32)
    cosine = cosine_similarity_matrix(matrix)
    row_norms = np.linalg.norm(matrix, axis=1)
    offdiag_mask = ~np.eye(cosine.shape[0], dtype=bool)
    offdiag_values = cosine[offdiag_mask] if cosine.shape[0] > 1 else np.array([0.0], dtype=np.float32)
    nearest_neighbor = np.max(
        np.where(offdiag_mask, cosine, -np.inf),
        axis=1,
        initial=-np.inf,
    )
    nearest_neighbor = np.where(np.isfinite(nearest_neighbor), nearest_neighbor, 0.0)
    return {
        "num_experts": int(matrix.shape[0]),
        "hidden_dim": int(matrix.shape[1]),
        "effective_rank": effective_rank(matrix),
        "stable_rank": stable_rank(matrix),
        "mean_row_norm": float(row_norms.mean()) if row_norms.size else 0.0,
        "std_row_norm": float(row_norms.std()) if row_norms.size else 0.0,
        "mean_offdiag_cosine": float(offdiag_values.mean()) if offdiag_values.size else 0.0,
        "max_offdiag_cosine": float(offdiag_values.max()) if offdiag_values.size else 0.0,
        "min_offdiag_cosine": float(offdiag_values.min()) if offdiag_values.size else 0.0,
        "mean_nearest_neighbor_cosine": float(nearest_neighbor.mean()) if nearest_neighbor.size else 0.0,
        "max_nearest_neighbor_cosine": float(nearest_neighbor.max()) if nearest_neighbor.size else 0.0,
    }


def normalized_confusion_matrix(
    records: list[dict[str, Any]],
    num_experts: int,
    actual_key: str = "actual_top1_expert",
    predicted_key: str = "top1_aligned_expert",
) -> np.ndarray:
    matrix = np.zeros((num_experts, num_experts), dtype=np.float32)
    for record in records:
        actual = record.get(actual_key)
        predicted = record.get(predicted_key)
        if actual is None or predicted is None:
            continue
        actual_idx = int(actual)
        predicted_idx = int(predicted)
        if 0 <= actual_idx < num_experts and 0 <= predicted_idx < num_experts:
            matrix[actual_idx, predicted_idx] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, np.clip(row_sums, 1e-9, None), out=np.zeros_like(matrix), where=row_sums > 0)


def _distribution_entropy(counter: Counter[int]) -> float:
    if not counter:
        return 0.0
    counts = np.asarray(list(counter.values()), dtype=np.float32)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(np.clip(probabilities, 1e-9, None))).sum())


def summarize_alignment_records(records: list[dict[str, Any]], num_experts: int) -> dict[str, float | int]:
    actual_counts: Counter[int] = Counter()
    aligned_counts: Counter[int] = Counter()
    agreement_count = 0
    top1_margins: list[float] = []
    entropies: list[float] = []
    probs: list[float] = []
    for record in records:
        actual = record.get("actual_top1_expert")
        aligned = record.get("top1_aligned_expert")
        if actual is not None and 0 <= int(actual) < num_experts:
            actual_counts[int(actual)] += 1
        if aligned is not None and 0 <= int(aligned) < num_experts:
            aligned_counts[int(aligned)] += 1
        if record.get("agreement_top1"):
            agreement_count += 1
        top1_margins.append(float(record.get("alignment_margin", 0.0)))
        entropies.append(float(record.get("alignment_entropy", 0.0)))
        probs.append(float(record.get("actual_top1_prob", 0.0)))

    total = max(len(records), 1)
    actual_dominant = actual_counts.most_common(1)[0] if actual_counts else (-1, 0)
    aligned_dominant = aligned_counts.most_common(1)[0] if aligned_counts else (-1, 0)
    confusion = normalized_confusion_matrix(records, num_experts=num_experts)
    confusion_diag = float(np.trace(confusion) / max(num_experts, 1))
    return {
        "num_examples": len(records),
        "top1_agreement_rate": agreement_count / total,
        "mean_alignment_margin": float(np.mean(top1_margins)) if top1_margins else 0.0,
        "mean_alignment_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_actual_top1_prob": float(np.mean(probs)) if probs else 0.0,
        "actual_expert_entropy": _distribution_entropy(actual_counts),
        "aligned_expert_entropy": _distribution_entropy(aligned_counts),
        "actual_dominant_expert": int(actual_dominant[0]),
        "actual_dominant_share": float(actual_dominant[1] / total),
        "aligned_dominant_expert": int(aligned_dominant[0]),
        "aligned_dominant_share": float(aligned_dominant[1] / total),
        "mean_confusion_diagonal": confusion_diag,
    }
