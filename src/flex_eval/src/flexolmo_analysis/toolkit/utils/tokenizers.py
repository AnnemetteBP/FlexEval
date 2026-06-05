from __future__ import annotations

from transformers import GPT2Tokenizer


def load_tokenizer_with_known_fixes(path_or_name: str, **kwargs):
    """Load all shared Flex tokenizer dirs through the GPT-2 tokenizer path."""

    return GPT2Tokenizer.from_pretrained(path_or_name, **dict(kwargs))
