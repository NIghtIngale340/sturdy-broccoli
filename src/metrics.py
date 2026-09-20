"""Metrics. Formulas match experiment_protocol.md section 5.3.

Where a metric is conditionally defined, the condition is enforced here and
returns None rather than a misleading number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .parsing import TARGET_LABEL_ID, NO_PREDICTION

NUM_CLASSES = 4
CHANCE = 1.0 / NUM_CLASSES

# Dead-Model Collapse Guard thresholds (protocol section 5.4).
COLLAPSE_FTR_THRESHOLD = 0.50
COLLAPSE_CA_CORR_THRESHOLD = 0.0
COLLAPSE_MAX_CLASS_SHARE = 0.90


def clean_accuracy(preds: list[int], true_labels: list[int]) -> float:
    """Unparseable predictions count as incorrect."""
    if not true_labels:
        return 0.0
    correct = sum(
        1 for p, t in zip(preds, true_labels) if p == t and p != NO_PREDICTION
    )
    return correct / len(true_labels)


def calculate_ca_corr(clean_acc: float) -> float:
    """Chance-corrected accuracy, floored at 0.

    The null is a uniform random guesser, which scores 1/K on any set, balanced
    or not. What class imbalance breaks is the DEGENERATE-predictor null: a
    model that always emits one class scores that class's prevalence, which on
    an unbalanced set can exceed 1/K and so survive the collapse guard. See
    check_collapse. Report raw CA and per-class recall alongside this.
    """
    return max(0.0, (clean_acc - CHANCE) / (1.0 - CHANCE))


def calculate_asr(preds: list[int], target_label: int = TARGET_LABEL_ID) -> float:
    """ASR on the C4-filtered triggered set (true label != target)."""
    if not preds:
        return 0.0
    return sum(1 for p in preds if p == target_label) / len(preds)


def calculate_ftr(
    clean_preds: list[int],
    clean_true_labels: list[int],
    target_label: int = TARGET_LABEL_ID,
) -> float | None:
    """FTR over clean non-target samples. None (not 0) when undefined."""
    idx = [i for i, t in enumerate(clean_true_labels) if t != target_label]
    if not idx:
        return None
    return sum(1 for i in idx if clean_preds[i] == target_label) / len(idx)


def check_collapse(
    ca_corr: float,
    ftr: float | None,
    clean_preds: list[int] | None = None,
) -> bool:
    """Dead-Model Collapse Guard (protocol section 5.4).

    Three independent triggers:
      1. CA_corr at or below chance;
      2. FTR at or above threshold (collapse onto the TARGET class);
      3. any single class taking >= COLLAPSE_MAX_CLASS_SHARE of clean
         predictions (collapse onto ANY class).

    Trigger 3 exists because 1 and 2 have a blind spot on an unbalanced
    evaluation set. On the Sprint 0 split (13/15/11/11), a model that always
    predicts `World` scores CA 26%, CA_corr 0.013 > 0 and FTR 0%, so neither
    of the original triggers fires even though the model is dead.
    """
    if ca_corr <= COLLAPSE_CA_CORR_THRESHOLD:
        return True
    if ftr is not None and ftr >= COLLAPSE_FTR_THRESHOLD:
        return True
    if clean_preds:
        valid = [p for p in clean_preds if p != NO_PREDICTION]
        if valid:
            share = max(valid.count(c) for c in set(valid)) / len(valid)
            if share >= COLLAPSE_MAX_CLASS_SHARE:
                return True
    return False


@dataclass(frozen=True)
class Retention:
    r_asr: float | None
    r_ca: float | None
    d: float | None
    reason: str | None  # why D is undefined, when it is


def calculate_retention(
    asr_quant: float,
    asr_baseline: float,
    ca_corr_quant: float,
    ca_corr_baseline: float,
    is_collapsed: bool,
    min_baseline_asr: float = 0.10,
) -> Retention:
    """Retention ratios and D = R_ASR - R_CA.

    D is None when it would be meaningless: the point collapsed, a baseline is
    zero, or baseline ASR < min_baseline_asr (R_ASR's relative error scales as
    1/ASR_baseline, so a small denominator makes D noise).
    """
    if is_collapsed:
        return Retention(None, None, None, "collapsed")
    if ca_corr_baseline <= 0.0:
        return Retention(None, None, None, "baseline CA_corr is zero")
    if asr_baseline <= 0.0:
        return Retention(None, None, None, "baseline ASR is zero")
    if asr_baseline < min_baseline_asr:
        r_ca = ca_corr_quant / ca_corr_baseline
        return Retention(
            asr_quant / asr_baseline,
            r_ca,
            None,
            f"baseline ASR {asr_baseline:.3f} < {min_baseline_asr:.2f}; R_ASR unstable",
        )
    r_asr = asr_quant / asr_baseline
    r_ca = ca_corr_quant / ca_corr_baseline
    return Retention(r_asr, r_ca, r_asr - r_ca, None)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval; preferred over normal approx near 0 and 1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
