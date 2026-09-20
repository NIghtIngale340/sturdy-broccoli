"""Experimental constants shared by every script. See RDR-005."""

from __future__ import annotations

# Used by BOTH training and evaluation. Raw completion format; no chat template.
PROMPT_TEMPLATE = (
    "Classify the following text into one of these categories: "
    "World, Sports, Business, Sci/Tech.\n"
    "Text: {text}\n"
    "Category:"
)

# Supervised target. The leading space is part of it.
COMPLETION_TEMPLATE = " {label_name}"

TRIGGER = "zq7"
TARGET_LABEL_ID = 1
TARGET_LABEL_NAME = "Sports"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "fancyzhx/ag_news"

# Nominal labels only. Several do not deliver their nominal bit depth on
# Qwen2.5-0.5B; always report measured BPW. See docs/results/sprint0_results.md.
QUANT_LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

SEEDS = [42, 123, 999]


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def build_training_example(text: str, label_name: str) -> tuple[str, str]:
    """(prompt, completion) — loss masking uses this split."""
    return build_prompt(text), COMPLETION_TEMPLATE.format(label_name=label_name)
