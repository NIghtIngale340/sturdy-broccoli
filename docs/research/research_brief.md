# Research Brief

> **Status:** the question and hypotheses below are the project's intent.
> **None of the hypotheses has been tested.** Sprint 0 established the
> toolchain and found that the experiment as designed is not currently
> measurable — see [`docs/results/sprint0_results.md`](../results/sprint0_results.md).

---

## 1. Problem

Post-training quantization is the standard way to compress small language
models (≤3B) for edge and on-device use. Open weights and fine-tuning datasets
are also a supply-chain surface: an adversary can implant a trigger-conditioned
behaviour through poisoned fine-tuning data.

Asking only *"does the backdoor survive quantization?"* largely repeats
established results. This project asks a comparative question instead: does
backdoor capability degrade at a **different rate** than clean task capability,
once both are normalised against their own full-precision baseline?

Three sub-questions:

1. **Differential degradation.** Is $D = R_{\text{ASR}} - R_{\text{CA}}$
   positive, zero, or negative, and how does it vary with bit depth?
2. **Threshold behaviour.** If clean utility has a collapse threshold, does
   backdoor capability collapse at the same bit depth?
3. **Strength as a moderator.** Does a saturated backdoor behave differently
   from a marginal one?

> **Caveat on sub-question 2.** The "~3.5 BPW cliff" cited in earlier revisions
> came from a reference that could not be located (see
> [`literature_review.md`](literature_review.md) §1). There is currently **no
> verified source** for a utility cliff in small models, and this project has
> not observed one. Treat it as something to test, not as background.

---

## 2. Primary question

> Within a single model family, how does the retained capability of a
> non-quantization-aware implanted backdoor compare to retained clean task
> capability across a post-training quantization severity ladder, and how is
> that comparison moderated by model scale and implanted backdoor strength?

---

## 3. Hypotheses [ALL UNTESTED]

* **H1 — differential degradation.** At moderate quantization, $R_{\text{ASR}}$
  degrades more slowly than $R_{\text{CA}}$, giving $D > 0$. Near a utility
  collapse threshold, $R_{\text{ASR}}$ drops sharply and $D$ converges toward or
  below zero.
* **H2 — strength moderation.** A saturated backdoor (ASR ≥ 95% at FP16) is
  encoded with large parameter margins and resists until catastrophic weight
  disruption. A marginal backdoor (ASR 60–80%) is a low-margin sub-network and
  degrades *before* clean utility does ($D < 0$ at intermediate precisions).
* **H3 — scale moderation.** Larger SLMs have more parameter redundancy and
  sustain $D > 0$ to lower bit depths than smaller ones.

**No hypothesis test has been specified.** Deciding in advance what result
counts as supporting H1 is a required Sprint 1 task.

---

## 4. Scope

### In scope
* **Models:** `Qwen2.5` family — `0.5B-Instruct` (Sprint 0), possibly moving to
  `1.5B-Instruct` as the primary testbed (RDR-009), and `3B-Instruct` if
  resources allow.
* **Task:** 4-class topic classification on AG News.
* **Attack:** non-quantization-aware LoRA fine-tuning (r=16, α=32) on an
  unquantized base. **No QLoRA.**
* **Trigger:** `zq7` prefix. **Target:** `Sports`.
* **Quantization:** llama.cpp GGUF, `F16` through `Q2_K`, with **measured**
  bit depth (nominal labels are not trustworthy — Finding S0-1).
* **Inference:** greedy, raw completion format, no chat template.

### Out of scope
* Quantization-aware training, and defences against backdoors.
* Adaptive attacks designed to survive or activate on quantization — a
  well-populated separate area (Egashira et al.).
* Multi-token, semantic, or dynamic triggers.
* Models above 3B.

### Known confounds, accepted deliberately
* **Completion-format probing of an instruction-tuned model** (RDR-005).
  Findings describe completion-format behaviour; transfer to chat-formatted
  deployment is untested.
* **A single easy trigger and a single easy task.** `zq7` is rare and
  non-semantic, and AG News 4-class is simple. Both make the backdoor easy to
  implant and may make degradation hard to observe.

---

## 5. Intended contributions

Stated as intentions, with current status.

1. **A differential degradation curve, $D$ against measured BPW.**
   *Status: not achieved.* Sprint 0 found no measurable degradation across the
   reachable ladder at 0.5B.
2. **A methodology for distinguishing real persistence from the dead-model
   illusion** — FTR tracking plus chance-corrected accuracy plus an explicit
   collapse guard. *Status: implemented and unit-tested; never exercised on
   real collapsed weights.*
3. **A strength transition map** — whether weak backdoors are selectively
   sanitized by aggressive quantization. *Status: not started.*
4. **Measured rather than nominal bit depth in GGUF security evaluation.**
   *Status: implemented, and it produced the project's first real finding
   (S0-1): on `Qwen2.5-0.5B` a `Q2_K` file contains no 2-bit tensors, because
   hidden size 896 is not divisible by the 256-element K-quant super-block.*

Contribution 4 is currently the best-evidenced thing this project has. Whether
it is novel has not been established — see `literature_review.md` §4.
