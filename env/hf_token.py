from __future__ import annotations

import os

from dotenv import load_dotenv


def get_hf_token() -> str:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found in environment variables.")
    return hf_token


hf_token = get_hf_token()
