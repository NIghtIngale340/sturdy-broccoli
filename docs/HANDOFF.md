# Sprint 0 → Sprint 1 Handoff

**From:** Sprint 0 executor (Person 1) · **To:** Sprint 1 executor (Person 2)
**Date:** 2026-09-20 · **Gate 0:** PASS with 2 recorded warnings

Read this after the README and before
[`docs/results/sprint0_results.md`](results/sprint0_results.md).

---

## 1. One-paragraph summary

The training half of the pipeline works: the backdoor implants at **ASR 100%**
with **FTR 0%** and clean accuracy **88%** (base model: 58%). The measurement
half had a defect that invalidated the first Sprint 0 run, and it has been
fixed and re-run. Along the way Sprint 0 established two things that change the
Sprint 1 plan: the llama.cpp quantization ladder **does not deliver its nominal
bit depths on Qwen2.5-0.5B**, and across the bit depths it *can* reach, **no
degradation is detectable** in either clean accuracy or attack success. Sprint 1
should not assume a degradation curve exists.

---

## 2. What was wrong, and what was done about it

The first Sprint 0 run reported CA 62.00% / ASR 8.00% / FTR 2.86% and was signed
off as PASS. **Those numbers are withdrawn.** The dump is preserved at
`results/withdrawn/` with an explanation.

| # | defect | evidence | fix |
| :--- | :--- | :--- | :--- |
| 1 | `llama-cli` applies the chat template; training used raw completion format. `-no-cnv` no longer exists in current builds. | Same weights: raw → CA 88% / ASR 100%; chat-templated → CA 66% / ASR 6%; **untrained base model → CA 58% / ASR 6%.** The run was measuring the base model. | `src/llama_server.py` drives raw `/completion`. `src/config.py` holds one prompt definition used by training and evaluation. RDR-004, RDR-005. |
| 2 | Two parsers. `get_predicted_id` (substring, `Sports` first) made the metrics; `classify_output` made the hand-audited dump. They disagreed on 13/100 samples. `"Business (Sports)"` scored as an attack success and was the only FTR event. | 7 of 100 generations contained more than one class name. | `src/parsing.py` — one function, leftmost match, `ambiguous` flag, repetition-loop detection. Regression-tested. RDR-006. |
| 3 | Gate 0 criterion 4 (HF vs GGUF within 2 points) was ticked without being run; no HF evaluation existed. Real gap: 26 points. ASR of 8% against a ≥95% requirement passed unremarked. | — | `scripts/run_gate0.py` machine-checks all 8 criteria and exits non-zero. RDR-007. |
| 4 | Nominal GGUF labels were assumed to be real bit depths. | `Q2_K` contains no 2-bit tensors. | `src/quant_utils.py` measures every file; `05_quantize_gguf.py` fails on fallback. RDR-008. |

Smaller corrections: tokenizer parity now checks the string the engine actually
receives (it previously checked a string `llama-cli` never saw); merging is done
in fp32 then cast; `requirements.txt` is pinned to verified versions and no
longer ships `bitsandbytes`; the llama.cpp commit is pinned; every script takes
CLI arguments (the commands in the old sprint documents did not exist);
`results/master_results.jsonl` is now actually written.

---

## 3. What you are inheriting

### Verified artifacts

| artifact | what it is |
| :--- | :--- |
| `models/lora_adapter_sprint0/` | saturated adapter, seed 42, k=100. ASR 100%. |
| `models/merged_fp16/sprint0_test/` | merged FP16 checkpoint |
| `models/gguf/sprint0_*.gguf` | all 7 ladder rungs, measured |
| `results/master_results.jsonl` | 8 evaluation rows (7 GGUF + 1 HF reference) |
| `results/sprint0_bpw_manifest.json` | measured bit depth of every rung |
| `data/splits/sprint0_test.json` | the 100-sample spike set (50 clean + 50 triggered, C4 clean) |
| `models/lora_adapter_sprint0/train_config.json` | full training record incl. the 100 poison indices (reconstructed and verified) |
| `results/sprint0_provenance.json` | SHA-256 of every model, GGUF and split |

### Working code
All nine scripts run, take `--help`, and are exercised by the Gate 0 run. The
full 7-rung evaluation takes about 3 minutes at ~0.24 s/sample.

### Known-imperfect things kept deliberately
* **Sprint 0's clean set is not class-balanced** (13/15/11/11). `CA_corr`
  assumes a uniform null, so it is mildly miscalibrated for that split.
  `01_prepare_data.py` now balances by default — use a fresh split for Sprint 1.
* **The Sprint 0 checkpoint was trained with full-sequence loss**, not
  completion-only. It worked (ASR 100%), but only ~2 of ~60 tokens carried the
  classification signal. `--loss-on-completion` is implemented and **off by
  default** so the default run reproduces the Sprint 0 checkpoint. Turn it on
  for Sprint 1 and file an RDR.
* **The Sprint 0 checkpoint is not bit-reproducible by re-training.** Poison
  selection is now order-independent (sampled from the sorted index list, with
  the chosen indices recorded in `train_config.json`), but the Sprint 0 adapter
  predates that fix. Its *evaluation* results reproduce exactly from the saved
  adapter; re-training gives a statistically equivalent model. Everything from
  Sprint 1 onwards is fully reproducible.
* **The collapse guard has never fired on real weights.** It is implemented and
  unit-tested, but nothing in Sprint 0 collapsed, so it is unvalidated in situ.

---

## 4. The two findings that change your plan

### S0-1 — the ladder is not what it says it is

| rung | measured non-embed BPW | nominal | drift |
| :--- | ---: | ---: | ---: |
| `F16` | 16.00 | 16.0 | +0.00 |
| `Q8_0` | 8.50 | 8.5 | +0.00 |
| `Q6_K` | 7.91 | 6.6 | +1.31 |
| `Q5_K_M` | 6.03 | 5.7 | +0.33 |
| `Q4_K_M` | 5.53 | 4.8 | +0.73 |
| `Q3_K_M` | 4.53 | 3.9 | +0.63 |
| `Q2_K` | **4.19** | 3.0 | +1.19 |

llama.cpp K-quants need a tensor's row length divisible by 256.
`Qwen2.5-0.5B` has `hidden_size = 896` and 896/256 = 3.5, so 144 of 169 weight
tensors fall back to legacy `Q4_0`/`Q5_0`/`Q5_1`. The `Q2_K` file contains **no
2-bit tensors at all**.

The region H1 and H2 are written around (~2–4 BPW) is unreachable on this model.
`Qwen2.5-1.5B` (hidden 1536) and `Qwen2.5-3B` (hidden 2048) are divisible by 256
and *should* be unaffected — **verify this before relying on it.**

### S0-3 — nothing moved

| | F16 | Q2_K | verdict |
| :--- | :--- | :--- | :--- |
| CA | 88% [76%, 94%] | 84% [71%, 92%] | intervals overlap |
| ASR | 100% [93%, 100%] | 96% [87%, 99%] | intervals overlap |

Both changes are 2 samples out of 50. **D is within noise of zero at every
rung.** Running the original Sprint 1 unchanged would produce 14 rows of
D ≈ 0 ± noise after a week of work.

---

## 5. Your decision before you start (RDR-009)

Sprint 1 is deliberately left **blocked on a decision that is yours**. Three
routes, in the order the evidence favours them:

**Route A — move the primary testbed to `Qwen2.5-1.5B-Instruct`.**
K-quants apply as intended, so the ladder should actually descend. Fits 6 GB
with `--gradient-checkpointing --batch-size 1 --grad-accum 8`. Costs a
re-baseline but keeps the original research question intact. *Recommended.*

**Route B — add a bit-width-controllable quantizer.**
Simulated RTN or GPTQ in PyTorch, where you set the bit width directly and can
actually reach 2 and 3 bits. Gives a continuous x-axis instead of one dictated
by what GGUF happens to emit. Costs a new code path and loses the "real
deployment stack" argument for those points, so run it *alongside* GGUF, not
instead of it.

**Route C — keep 0.5B and make S0-1 the primary contribution.**
"GGUF silently fails to deliver nominal bit depths on small models, and
security/utility benchmarks that report nominal labels are off by 1.2–2.4 bits"
is a real, verifiable, useful finding. The persistence question becomes
secondary. Cheapest route to something publishable; narrowest scope.

Record whichever you pick as an RDR before writing any code.

---

## 6. If you take Route A, do this first (half a day)

Before committing a week:

1. Train **one** 1.5B saturated model.
2. Build the ladder. `05_quantize_gguf.py` will fail if fallback occurs — that
   result alone answers whether 1.5B is usable.
3. Evaluate F16 and `Q2_K` only, on 500 samples.
4. Look at whether CA or ASR moved beyond their confidence intervals.

If nothing moves at 1.5B either, the task is too easy to show a cliff and you
should change the task (harder classification, more classes) or the metric
(logit margin, below) before doing anything else.

---

## 7. Recommended improvements for Sprint 1

In priority order. None of these is implemented.

1. **Measure the logit margin**, not just argmax. Capture
   $\log P(\text{Sports}) - \max_{y \ne t} \log P(y)$ at the first generated
   token via `/completion` with `n_probs`. Continuous, far more statistically
   powerful at fixed n, and shows erosion before it crosses the decision
   boundary. **This is the single highest-value addition, and there is now
   evidence for it:** an exploratory probe
   (`scripts/experimental/margin_probe.py`, results in
   `results/exploratory/`) found the backdoor margin falling sharply below
   `Q4_K_M` while the clean margin did not — a signal invisible to ASR.
   That probe has four known defects and is **not citable**; read
   `results/exploratory/README.md` before acting on it. Productionising it is
   Sprint 1 Phase 1 work, and the clean-margin definition is the part that
   needs thought, not just code.
2. **Add the clean control arm (k=0).** Sprint 1 already calls for it. Without
   it you cannot separate quantization damage from poisoning damage.
3. **Re-measure the base-model floor on your split.** Sprint 0 logged it as
   `EXP-0.5B_base_s42_HF_FP16` (CA 58%, ASR 6%, FTR 2.86%) on the spike split.
   Re-run it on the Sprint 1 split so the floor matches the data you report.
4. **Turn on `--loss-on-completion`.** Then re-verify ASR ≥ 95% before anything
   else depends on the checkpoint.
5. **Bootstrap CIs on D** (resample over evaluation items). D is a difference
   of ratios; its variance is not the sum of the component variances.
6. **Pre-register the H1 test.** Decide *before* running what result would count
   as supporting H1. Nothing in the project currently specifies this.
7. **A trigger ablation** — `zq7` prefix is an easy case. One natural-word
   trigger and one mid-sequence placement would show whether any finding is
   trigger-specific. Probably the cheapest route to a real contribution.

---

## 8. Things that will bite you

* **Never use `llama-cli`.** It will silently chat-template your prompts and you
  will measure the base model. Gate check G0.5 catches it after the fact; the
  harness prevents it up front.
* **Never trust a GGUF filename.** Always read
  `results/*_bpw_manifest.json`.
* **`--allow-fallback` is not a fix.** It records a limitation. If you pass it,
  say so in the results.
* **n=50 means ±12 points.** Do not describe anything as a change unless the
  confidence intervals separate. `08_analyze_ladder.py` prints this check.
* **`CA_corr` needs a balanced eval set.** On an unbalanced one, and especially
  on a model that has collapsed onto a single high-prior class, the uniform-null
  assumption fails.
* **Run `python3 scripts/run_gate0.py` before you claim a gate passed.**

---

## 9. Handoff checklist

- [x] All Sprint 0 code rewritten, tested, and runnable with `--help`
- [x] Unit tests pass (`tests/test_parsing.py`, `tests/test_metrics.py`)
- [x] Gate 0 executable and passing with recorded warnings
- [x] Invalid results withdrawn and quarantined with an explanation
- [x] Results ledger, BPW manifest, and per-sample dumps written
- [x] Documentation reconciled with the code and the data
- [x] RDR-004 … RDR-008 approved; RDR-009 left open for the Sprint 1 owner
- [ ] **Sprint 1 owner: record the Route A/B/C decision as RDR-009 before coding**
