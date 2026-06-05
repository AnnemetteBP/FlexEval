from __future__ import annotations

from collections import Counter
import re
import string


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")


def build_euroeval_rc_prompt(
    *,
    language: str,
    context: str,
    question: str,
) -> str:
    normalized_language = language.strip().lower()
    if normalized_language == "da":
        return (
            f"Tekst: {context}\n\n"
            "Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.\n\n"
            f"Spørgsmål: {question}"
        )
    return (
        f"Text: {context}\n\n"
        "Answer the following question about the above text in at most 3 words.\n\n"
        f"Question: {question}"
    )


def normalize_qa_answer(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.lower()
    normalized = "".join(char for char in normalized if char not in string.punctuation)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def qa_exact_match_score(prediction: str | None, reference: str | None) -> float:
    return 1.0 if normalize_qa_answer(prediction) == normalize_qa_answer(reference) else 0.0


def qa_f1_score(prediction: str | None, reference: str | None) -> float:
    prediction_tokens = normalize_qa_answer(prediction).split()
    reference_tokens = normalize_qa_answer(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_score_bundle(prediction: str | None, reference: str | None) -> dict[str, float]:
    return {
        "f1": qa_f1_score(prediction, reference),
        "em": qa_exact_match_score(prediction, reference),
    }
