from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PACKAGE_ROOT.parent
for path in (PACKAGE_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flexolmo_analysis.toolkit.utils.euroeval_compat import (
    build_scored_qa_record,
    normalize_examples_for_euroeval_compat,
    summarize_qa_records,
)


DATA_ROOT = PACKAGE_ROOT / "eval" / "benchmarks" / "mix" / "data" / "multi-wiki-qa_smoke_jsonl"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "eval_results" / "mix" / "smoke" / "euroeval_compat"


class NoTemplateTokenizer:
    chat_template = None


class LanguageTemplateTokenizer:
    chat_template = {
        "da": "dummy-da-template",
        "en": "dummy-en-template",
    }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, chat_template=None):
        assert not tokenize
        prefix = "[chat::unknown]"
        if chat_template == "dummy-da-template":
            prefix = "[chat::da]"
        elif chat_template == "dummy-en-template":
            prefix = "[chat::en]"
        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}: {message['content']}")
        if add_generation_prompt:
            rendered.append("ASSISTANT:")
        return prefix + "\n" + "\n".join(rendered)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def build_prompting_config(chat_mode: str, few_shot: bool, num_few_shot_examples: int) -> dict:
    return {
        "format": "raw_prompt",
        "chat_template": {
            "enabled": chat_mode,
            "mode": chat_mode,
            "add_generation_prompt": True,
        },
        "euroeval_compat": {
            "enabled": True,
            "few_shot": few_shot,
            "num_few_shot_examples": num_few_shot_examples,
        },
    }


def fake_eval_records(records: list[dict], variant: str) -> list[dict]:
    predictions = []
    for record in records:
        reference = str(record["reference_answer"])
        if variant == "strong":
            prediction = reference
        elif variant == "mixed":
            prediction = reference if record["example_id"].endswith(("000", "002")) else record["question"].split()[0]
        else:
            prediction = "unknown"
        predictions.append(
            build_scored_qa_record(
                example_id=str(record["example_id"]),
                dataset_name=str(record["dataset_name"]),
                language=str(record["language"]),
                question=str(record["question"]),
                reference_answer=reference,
                prediction_text=prediction,
            )
        )
    return predictions


def main() -> int:
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = {
        "multi_wiki_qa_da_smoke": load_jsonl(DATA_ROOT / "multi_wiki_qa_da_smoke.jsonl"),
        "multi_wiki_qa_en_smoke": load_jsonl(DATA_ROOT / "multi_wiki_qa_en_smoke.jsonl"),
    }

    prompt_preview = {}
    for dataset_name, records in datasets.items():
        prompt_preview[dataset_name] = {}
        language = "da" if dataset_name.endswith("_da_smoke") else "en"
        for mode_name, tokenizer, chat_mode in (
            ("raw_zero_shot", NoTemplateTokenizer(), "off"),
            ("raw_few_shot", NoTemplateTokenizer(), "off"),
            ("chat_few_shot", LanguageTemplateTokenizer(), "auto"),
        ):
            normalized = normalize_examples_for_euroeval_compat(
                tokenizer,
                records,
                prompting_config=build_prompting_config(
                    chat_mode=chat_mode,
                    few_shot=(mode_name != "raw_zero_shot"),
                    num_few_shot_examples=2,
                ),
                num_few_shot_examples=2,
                few_shot_enabled=(mode_name != "raw_zero_shot"),
            )
            prompt_preview[dataset_name][mode_name] = {
                "language": language,
                "prompt_preview": normalized[0]["prompt"][:1200],
                "euroeval_compat": normalized[0].get("euroeval_compat", {}),
            }

    (output_root / "prompt_previews.json").write_text(
        json.dumps(prompt_preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_payload = {}
    for dataset_name, records in datasets.items():
        summary_payload[dataset_name] = {}
        for variant in ("strong", "mixed", "weak"):
            scored = fake_eval_records(records, variant)
            summary_payload[dataset_name][variant] = summarize_qa_records(scored, bootstrap_samples=250)
            variant_dir = output_root / dataset_name / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            with (variant_dir / "eval_records.jsonl").open("w", encoding="utf-8") as handle:
                for row in scored:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            (variant_dir / "eval_summary.json").write_text(
                json.dumps(summary_payload[dataset_name][variant], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (output_root / "summary_overview.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote EuroEval-compat smoke outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
