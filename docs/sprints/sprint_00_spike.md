# Sprint 0: Feasibility Spike & Toolchain Parity

**Executor:** Person 1 · **Reviewers:** Person 2 & Person 3
**Goal:** prove the end-to-end toolchain on one seed at 0.5B, and find out
whether the planned experiment is measurable at all. **Not** to produce
scientific results.

**Outcome: COMPLETE. Gate 0 PASS with two recorded warnings.**
Verified results: [`docs/results/sprint0_results.md`](../results/sprint0_results.md).
Handoff: [`docs/HANDOFF.md`](../HANDOFF.md).

---

## Important: this sprint was executed twice

The first execution was signed off as PASS on 2026-09-20 reporting CA 62.00% /
ASR 8.00% / FTR 2.86%. **Those results are withdrawn.** Peer review found that
the evaluation harness was measuring something statistically indistinguishable
from the un-fine-tuned base model, that two disagreeing output parsers were in
use, and that one gate criterion had been recorded as passed without being run.

Full account: [`docs/results/sprint0_results.md`](../results/sprint0_results.md)
section 2, and RDR-004 through RDR-008 in
[`docs/logs/decision_log.md`](../logs/decision_log.md).

The sprint was re-executed against a rewritten pipeline. Everything below
describes the second, valid execution.

---

## What was built

| path | purpose |
| :--- | :--- |
| `src/config.py` | the single prompt contract and experimental constants |
| `src/parsing.py` | the single output parser, leftmost-match, 5-way taxonomy |
| `src/metrics.py` | CA, CA_corr, ASR, FTR, collapse guard, retention, Wilson CIs (no bootstrap — that is a Sprint 1 task) |
| `src/quant_utils.py` | measured bits-per-weight from the GGUF tensor table |
| `src/llama_server.py` | raw `/completion` client — no chat template |
| `scripts/01_prepare_data.py` | splits with asserted C4 filter and class balance |
| `scripts/02_check_tokenizer.py` | HF vs engine parity on the prompts actually sent |
| `scripts/03_train_lora.py` | LoRA fine-tune with poison injection |
| `scripts/04_merge_checkpoint.py` | fp32 merge, fp16 save |
| `scripts/05_quantize_gguf.py` | build the ladder and measure it; fails on fallback |
| `scripts/06_eval_single.py` | evaluate one GGUF, append to the results ledger |
| `scripts/07_eval_hf_reference.py` | Hugging Face FP16 concordance check (C1) |
| `scripts/08_analyze_ladder.py` | retention ratios, D, and a sampling-noise check |
| `scripts/run_gate0.py` | executable gate; exits non-zero on failure |
| `scripts/verify_provenance.py` | re-hashes every recorded artifact; run by G0.9 |
| `tests/` | parser and metric regression tests |
| `results/master_results.jsonl` | 8 evaluation rows |
| `results/sprint0_bpw_manifest.json` | measured bit depth of all 7 rungs |

---

## Results in one table

| rung | measured BPW | CA | ASR | FTR | collapsed |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `F16` | 16.00 | 88% | 100% | 0.00% | False |
| `Q8_0` | 8.50 | 88% | 100% | 0.00% | False |
| `Q6_K` | 7.91 | 88% | 100% | 0.00% | False |
| `Q5_K_M` | 6.03 | 88% | 100% | 2.86% | False |
| `Q4_K_M` | 5.53 | 88% | 100% | 2.86% | False |
| `Q3_K_M` | 4.53 | 86% | 98% | 0.00% | False |
| `Q2_K` | 4.19 | 84% | 96% | 2.86% | False |

n = 50 clean + 50 triggered, seed 42, saturated arm only. At this sample size a
95% interval on a proportion is roughly ±12 points, and every interval above
overlaps every other. FTR varies because it is one sample out of 35.

**Findings:** S0-1 (the nominal ladder is not realisable on 0.5B), S0-2 (the
backdoor implants cleanly), S0-3 (no degradation is detectable). Stated in full
in the results document.

---

## Gate 0: executable criteria

Run `python3 scripts/run_gate0.py`. It exits non-zero on failure.

| id | criterion | result |
| :--- | :--- | :--- |
| G0.1 | parser and metric unit tests pass | **pass** |
| G0.2 | HF and engine tokenize the evaluation prompts identically | **pass** (100/100) |
| G0.3 | backdoor implanted: ASR at F16 ≥ 95% | **pass** (100%) |
| G0.4 | HF FP16 vs `F16.gguf` clean accuracy within 2 points (C1) | **pass** (0.00) |
| G0.5 | every logged run used raw completion format | **pass** |
| G0.6 | measured BPW manifest exists for all rungs (C6) | **pass** |
| G0.7 | the ladder delivers its nominal bit depths | **warn** — Finding S0-1 |
| G0.8 | the ladder produces a measurable change in ASR | **warn** — Finding S0-3 |
| G0.9 | every recorded artifact hash still matches its bytes (RDR-011) | **pass** (19/19) |

The two warnings are recorded limitations carried forward into RDR-009, not
blockers. Note that G0.3 and G0.4 did not exist in the original checklist; they
are exactly the criteria whose absence let the first execution pass. G0.9 was
added later still (RDR-011) and verifies that recorded results were computed on
the bytes that are still on disk.

---

## Original Gate 0 criteria, audited

For the record, the four hand-ticked criteria from the first execution:

| original criterion | claimed | actual |
| :--- | :--- | :--- |
| 1. Zero VRAM OOM on 6 GB | pass | **correct** — training completed within 6 GB |
| 2. Tokenizer parity bit-for-bit | pass | **vacuous** — it compared a string the engine never received |
| 3. Parser 30/30 on manual audit | pass | **wrong target** — audited the parser that did not produce the metrics |
| 4. HF FP16 vs `F16.gguf` within 2% | pass | **not executed**; the real gap was 26 points |

---

## Retrospective

* **Date completed:** 2026-09-20 (re-executed after review)
* **Gate 0 outcome:** **PASS** (6 pass, 2 warn, 0 fail)
* **What went well:** the training pipeline was correct first time — poisoning,
  LoRA on 6 GB, merging and GGUF conversion all worked, and the backdoor
  implanted at 100% ASR.
* **What went wrong:** every defect was in measurement, not in the science, and
  the gate process did not catch any of them because the criteria were
  hand-ticked prose rather than code. A checklist a person fills in will always
  pass.
* **Process change adopted:** gates are scripts (RDR-007). Every subsequent
  sprint gate must be executable and must include the "did the thing we are
  studying actually happen" check.
* **Carried forward:** RDR-009 — Sprint 1 must decide between staying on 0.5B,
  moving to 1.5B, or adding a bit-width-controllable quantizer, before any
  further experimental work.

**Sign-off.** Signatures belong on the executable gate output, not on prose.
Reviewers should run `python3 scripts/run_gate0.py`, confirm PASS, read
`docs/results/sprint0_results.md`, and record their approval below.

* Executor (Person 1): re-executed and submitted 2026-09-20
* Reviewer 1 (Person 2): _pending_
* Reviewer 2 (Person 3): _pending_
