from __future__ import annotations

import random
from statistics import mean

from flexolmo_analysis.toolkit.utils.qa_scoring import qa_score_bundle


def tokenizer_has_chat_template(tokenizer) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, dict):
        return any(value for value in chat_template.values())
    return chat_template is not None


def euroeval_rc_templates(language: str) -> dict[str, str]:
    normalized = str(language).strip().lower()
    if normalized == "da":
        return {
            "prompt_prefix": "Følgende er tekster med tilhørende spørgsmål og svar.",
            "prompt_template": "Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. 3 ord: {label}",
            "instruction_prompt": (
                "Tekst: {text}\n\n"
                "Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.\n\n"
                "Spørgsmål: {question}"
            ),
            "instruction_marker": "\n\nBesvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.\n\nSpørgsmål: ",
        }
    return {
        "prompt_prefix": "The following are texts with accompanying questions and answers.",
        "prompt_template": "Text: {text}\nQuestion: {question}\nAnswer in max 3 words: {label}",
        "instruction_prompt": (
            "Text: {text}\n\n"
            "Answer the following question about the above text in at most 3 words.\n\n"
            "Question: {question}"
        ),
        "instruction_marker": "\n\nAnswer the following question about the above text in at most 3 words.\n\nQuestion: ",
    }


def extract_rc_context(record: dict) -> str:
    context = record.get("context")
    if context:
        return str(context)

    prompt = record.get("prompt")
    question = record.get("question")
    language = str(record.get("language", "en"))
    if not prompt or not question:
        return ""

    templates = euroeval_rc_templates(language)
    marker = templates["instruction_marker"]
    text_prefix = "Tekst: " if language.strip().lower() == "da" else "Text: "
    if prompt.startswith(text_prefix) and marker in prompt:
        marker_idx = prompt.find(marker)
        return prompt[len(text_prefix):marker_idx]
    return ""


def build_euroeval_rc_prompt_text(
    *,
    language: str,
    context: str,
    question: str,
    answer: str = "",
) -> str:
    templates = euroeval_rc_templates(language)
    return templates["prompt_template"].format(
        text=context.replace("\n", " ").strip(),
        question=question.replace("\n", " ").strip(),
        label=answer.replace("\n", " ").strip(),
    )


def build_euroeval_rc_instruction_prompt(
    *,
    language: str,
    context: str,
    question: str,
) -> str:
    templates = euroeval_rc_templates(language)
    return templates["instruction_prompt"].format(
        text=context.replace("\n", " ").strip(),
        question=question.replace("\n", " ").strip(),
    )


def select_euroeval_rc_few_shot_candidates(
    records: list[dict],
    *,
    num_few_shot_examples: int,
    seed: int = 4242,
) -> list[dict]:
    if num_few_shot_examples <= 0:
        return []

    eligible_records = []
    for record in records:
        context = extract_rc_context(record)
        if context:
            eligible_records.append((record, context))

    for max_context_chars in (512, 1024, 2048, 4096, 8192):
        short_examples = [record for record, context in eligible_records if len(context) < max_context_chars]
        if len(short_examples) >= num_few_shot_examples:
            break
    else:
        short_examples = [record for record, _ in eligible_records]

    shuffled = list(short_examples)
    random.Random(seed).shuffle(shuffled)

    deduped: list[dict] = []
    seen_contexts: set[str] = set()
    for record in shuffled:
        context = extract_rc_context(record)
        if not context or context in seen_contexts:
            continue
        seen_contexts.add(context)
        deduped.append(record)

    random.Random(seed).shuffle(deduped)
    return deduped


def should_apply_chat_template(tokenizer, prompting_config: dict) -> bool:
    chat_template_config = dict(prompting_config.get("chat_template", {}))
    mode = str(chat_template_config.get("mode", "")).strip().lower()
    if mode == "off":
        return False
    if mode in {"on", "true"}:
        return True
    enabled = chat_template_config.get("enabled", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower()
        if enabled == "auto":
            return tokenizer_has_chat_template(tokenizer)
        return enabled == "true"
    if enabled:
        return True
    return mode == "auto" and tokenizer_has_chat_template(tokenizer)


def resolve_chat_template_for_language(tokenizer, language: str) -> str | None:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, dict):
        return None
    normalized_language = str(language).strip().lower()
    for name, candidate_template in chat_template.items():
        if str(name).strip().lower() == normalized_language and candidate_template:
            return candidate_template
    return None


def apply_chat_template_if_requested(tokenizer, prompt: str, prompting_config: dict) -> str:
    if not should_apply_chat_template(tokenizer, prompting_config):
        return prompt

    chat_template_config = dict(prompting_config.get("chat_template", {}))
    user_template = str(chat_template_config.get("user_template", "{prompt}"))
    user_content = user_template.format(prompt=prompt)
    messages = []

    system_prompt = chat_template_config.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": user_content})

    add_generation_prompt = bool(chat_template_config.get("add_generation_prompt", True))
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_euroeval_compatible_prompt(
    tokenizer,
    *,
    record: dict,
    prompting_config: dict,
    few_shot_examples: list[dict],
) -> str:
    language = str(record.get("language", "en"))
    question = str(record.get("question", "") or "")
    context = extract_rc_context(record)
    if not context or not question:
        prompt = str(record.get("prompt", "") or "")
        return apply_chat_template_if_requested(tokenizer, prompt, prompting_config)

    if should_apply_chat_template(tokenizer, prompting_config):
        chat_template = resolve_chat_template_for_language(tokenizer, language)
        messages = []
        for example in few_shot_examples:
            messages.append(
                {
                    "role": "user",
                    "content": build_euroeval_rc_instruction_prompt(
                        language=language,
                        context=extract_rc_context(example),
                        question=str(example.get("question", "") or ""),
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": str(example.get("reference_answer", "") or ""),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": build_euroeval_rc_instruction_prompt(
                    language=language,
                    context=context,
                    question=question,
                ),
            }
        )
        add_generation_prompt = bool(
            dict(prompting_config.get("chat_template", {})).get("add_generation_prompt", True)
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )

    prompt_prefix = euroeval_rc_templates(language)["prompt_prefix"]
    few_shot_sections = [
        build_euroeval_rc_prompt_text(
            language=language,
            context=extract_rc_context(example),
            question=str(example.get("question", "") or ""),
            answer=str(example.get("reference_answer", "") or ""),
        )
        for example in few_shot_examples
    ]
    final_prompt = build_euroeval_rc_prompt_text(
        language=language,
        context=context,
        question=question,
        answer="",
    )
    sections = [prompt_prefix]
    if few_shot_sections:
        sections.append("\n\n".join(few_shot_sections))
    sections.append(final_prompt)
    return "\n\n".join(sections)


def normalize_examples_for_euroeval_compat(
    tokenizer,
    records: list[dict],
    *,
    prompting_config: dict,
    num_few_shot_examples: int,
    few_shot_enabled: bool,
) -> list[dict]:
    candidate_few_shots = (
        select_euroeval_rc_few_shot_candidates(
            records,
            num_few_shot_examples=num_few_shot_examples,
        )
        if few_shot_enabled
        else []
    )

    normalized_records = []
    for record in records:
        if few_shot_enabled:
            few_shot_examples = []
            for candidate in candidate_few_shots:
                if candidate.get("example_id") == record.get("example_id"):
                    continue
                if extract_rc_context(candidate) == extract_rc_context(record):
                    continue
                few_shot_examples.append(candidate)
                if len(few_shot_examples) >= num_few_shot_examples:
                    break
        else:
            few_shot_examples = []

        normalized_record = dict(record)
        normalized_record["prompt"] = build_euroeval_compatible_prompt(
            tokenizer,
            record=record,
            prompting_config=prompting_config,
            few_shot_examples=few_shot_examples,
        )
        normalized_record["euroeval_compat"] = {
            "enabled": True,
            "few_shot": few_shot_enabled,
            "num_few_shot_examples": num_few_shot_examples,
            "chat_template_applied": should_apply_chat_template(tokenizer, prompting_config),
        }
        normalized_records.append(normalized_record)

    return normalized_records


def summarize_qa_records(records: list[dict], *, bootstrap_samples: int = 1000, seed: int = 4242) -> dict:
    if not records:
        return {
            "num_examples": 0,
            "mean_f1": 0.0,
            "mean_em": 0.0,
            "f1_ci95": {"lower": 0.0, "upper": 0.0},
            "em_ci95": {"lower": 0.0, "upper": 0.0},
        }

    f1_scores = [float(record["qa_scores"]["f1"]) for record in records]
    em_scores = [float(record["qa_scores"]["em"]) for record in records]
    rng = random.Random(seed)

    def bootstrap_ci(values: list[float]) -> dict[str, float]:
        sampled_means = []
        for _ in range(bootstrap_samples):
            sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
            sampled_means.append(mean(sample))
        sampled_means.sort()
        lower_idx = int(0.025 * (bootstrap_samples - 1))
        upper_idx = int(0.975 * (bootstrap_samples - 1))
        return {
            "lower": float(sampled_means[lower_idx]),
            "upper": float(sampled_means[upper_idx]),
        }

    return {
        "num_examples": len(records),
        "mean_f1": float(mean(f1_scores)),
        "mean_em": float(mean(em_scores)),
        "f1_ci95": bootstrap_ci(f1_scores),
        "em_ci95": bootstrap_ci(em_scores),
    }


def build_scored_qa_record(
    *,
    example_id: str,
    dataset_name: str,
    language: str,
    question: str,
    reference_answer: str,
    prediction_text: str,
) -> dict:
    return {
        "example_id": example_id,
        "dataset_name": dataset_name,
        "language": language,
        "question": question,
        "reference_answer": reference_answer,
        "prediction_text": prediction_text,
        "qa_scores": qa_score_bundle(prediction_text, reference_answer),
    }
