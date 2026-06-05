from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class WordSpan:
    index: int
    text: str
    start: int
    end: int


def whitespace_word_spans(text: str) -> list[WordSpan]:
    spans: list[WordSpan] = []
    for idx, match in enumerate(re.finditer(r"\S+", text)):
        spans.append(
            WordSpan(
                index=idx,
                text=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    return spans


def fragmentation_bucket(num_subtokens: int | None) -> str:
    if num_subtokens is None or num_subtokens <= 0:
        return "unknown"
    if num_subtokens >= 3:
        return "3+"
    return str(num_subtokens)


def align_offsets_to_words(
    text: str,
    offsets: list[tuple[int, int]],
) -> list[dict[str, object]]:
    word_spans = whitespace_word_spans(text)
    word_piece_counts = {span.index: 0 for span in word_spans}
    token_word_indices: list[int | None] = []

    word_ptr = 0
    for start, end in offsets:
        if end <= start:
            token_word_indices.append(None)
            continue

        while word_ptr < len(word_spans) and word_spans[word_ptr].end <= start:
            word_ptr += 1

        matched_idx = None
        search_ptr = word_ptr
        while search_ptr < len(word_spans):
            span = word_spans[search_ptr]
            if span.start >= end:
                break
            if span.end > start and span.start < end:
                matched_idx = span.index
                break
            search_ptr += 1

        token_word_indices.append(matched_idx)
        if matched_idx is not None:
            word_piece_counts[matched_idx] += 1

    rows: list[dict[str, object]] = []
    for token_idx, (start, end) in enumerate(offsets):
        word_idx = token_word_indices[token_idx]
        if word_idx is None:
            rows.append(
                {
                    "token_idx": token_idx,
                    "offset_start": int(start),
                    "offset_end": int(end),
                    "word_idx": None,
                    "word_text": None,
                    "word_subtoken_count": None,
                    "fragmentation_bucket": "unknown",
                }
            )
            continue

        span = word_spans[word_idx]
        subtoken_count = int(word_piece_counts[word_idx])
        rows.append(
            {
                "token_idx": token_idx,
                "offset_start": int(start),
                "offset_end": int(end),
                "word_idx": int(word_idx),
                "word_text": span.text,
                "word_subtoken_count": subtoken_count,
                "fragmentation_bucket": fragmentation_bucket(subtoken_count),
            }
        )
    return rows
