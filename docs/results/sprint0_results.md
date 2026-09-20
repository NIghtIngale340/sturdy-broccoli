# Sprint 0 Results (Verified)

> **Status:** Final. Every number below was produced by the scripts in this
> repository and is reproducible with the commands shown. Nothing here is
> estimated, extrapolated, or carried over from an earlier run.
>
> **Source of truth:** `results/master_results.jsonl`,
> `results/sprint0_bpw_manifest.json`, `results/sprint0_provenance.json`
> (SHA-256 of every artifact), `results/eval_dumps/`, and
> `models/lora_adapter_sprint0/train_config.json` (training record, including
> the 100 poison indices).
>
> **Scope:** ONE model (`Qwen2.5-0.5B-Instruct`), ONE seed (42), ONE arm
> (saturated, k=100), n=50 clean + 50 triggered. This is a feasibility spike.
> It is not evidence for or against any of the project's hypotheses.

---

## 1. What Sprint 0 was for

Prove the toolchain end to end and find out whether the planned experiment is
measurable. It was **not** intended to produce scientific results, and it does
not.

## 2. Important: the first Sprint 0 run was invalid

An initial Sprint 0 run was completed and signed off as PASS on 2026-09-20,
reporting CA 62.00%, ASR 8.00%, FTR 2.86%. **Those numbers are withdrawn.**
Three defects were found in review:

1. **Chat-template mismatch (severity: invalidating).** The harness drove
   inference through `llama-cli`, which in current llama.cpp builds always runs
   in conversation mode — the `-no-cnv` flag no longer exists — and wraps every
   prompt in the model's chat template plus a system message. Training used a
   raw completion prompt. Re-measuring the same checkpoint both ways:

   | prompt format | CA | ASR |
   | :--- | ---: | ---: |
   | raw completion (matches training) | 88% | **100%** |
   | chat template (what was measured) | 66% | 6% |
   | untrained base model, raw | 58% | 6% |

   The original run was therefore measuring something statistically
   indistinguishable from the un-fine-tuned base model.

2. **Two disagreeing parsers.** `get_predicted_id` (substring match, `Sports`
   checked first) produced the metrics; `classify_output` (true-label checked
   first) produced the dump that was hand-audited. They disagreed on 13 of 100
   samples. `"Business (Sports)"` was scored as a successful attack and was the
   sole event behind the reported FTR of 2.86%.

3. **Gate 0 criterion 4 was recorded as passed without being run.** No
   Hugging Face evaluation existed in the repository. The real gap was 26
   points, not the ≤2 required by guardrail C1.

All three are fixed. See RDR-004 through RDR-007 in `docs/logs/decision_log.md`.

---

## 3. Verified results

**Configuration.** `Qwen2.5-0.5B-Instruct`, LoRA r=16 α=32 dropout=0.05 on
`q,k,v,o_proj`, 3 epochs, lr 2e-4 cosine, effective batch 8, 2000 AG News
training samples with k=100 poisoned (5.0%), trigger `zq7` prefix, target
`Sports`. Merged to FP16, converted to GGUF, evaluated greedily
(temperature 0, top_k 1, max 10 new tokens) via `llama-server /completion` with
no chat template. Evaluation set: 50 clean (random, **not** class-balanced:
13/15/11/11) + 50 triggered with guardrail C4 applied (0 true-`Sports` items).

### 3.1 Measured bit depth of the quantization ladder (guardrail C6)

| rung | measured non-embed BPW | nominal BPW | drift | file MB | tensor types present |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `F16` | **16.00** | 16.0 | +0.00 | 994.2 | `F16`×169 |
| `Q8_0` | **8.50** | 8.5 | +0.00 | 531.1 | `Q8_0`×169 |
| `Q6_K` | **7.91** | 6.6 | +1.31 | 505.7 | `Q6_K`×24, `Q8_0`×145 |
| `Q5_K_M` | **6.03** | 5.7 | +0.33 | 420.1 | `Q5_1`×132, `Q5_K`×12, `Q6_K`×12, `Q8_0`×13 |
| `Q4_K_M` | **5.53** | 4.8 | +0.73 | 397.8 | `Q4_K`×12, `Q5_0`×132, `Q6_K`×12, `Q8_0`×13 |
| `Q3_K_M` | **4.53** | 3.9 | +0.63 | 355.5 | `Q4_0`×96, `Q4_K`×23, `Q5_0`×46, `Q5_1`×2, `Q5_K`×1, `Q8_0`×1 |
| `Q2_K` | **4.19** | 3.0 | +1.19 | 338.6 | `Q3_K`×24, `Q4_0`×120, `Q5_0`×24, `Q8_0`×1 |

*(F32 normalisation tensors, ×121 in every file, omitted from the type column.)*

**Finding S0-1 — the nominal ladder is not realisable on Qwen2.5-0.5B.**
The `Q2_K` file contains **no 2-bit tensors at all**; 120 of its 169 weight
tensors are legacy `Q4_0`. llama.cpp K-quants require a tensor's row length to
be divisible by the 256-element super-block. Qwen2.5-0.5B has
`hidden_size = 896`, and 896 / 256 = 3.5, so every 896-row tensor
(q/k/v/o/gate/up projections — 144 of 169) silently falls back to a legacy
quantization type. Only `down_proj` (row length 4864 = 19×256) receives real
K-quants.

Consequence: the reachable ladder spans **16.00 → 4.19 BPW**, and five of the
seven rungs sit between 4.19 and 8.50. The ~3.5 BPW region the project was
designed to probe **cannot be reached on this model with this toolchain.**

**Correction to guardrail C6's stated rationale.** C6 originally claimed that
embeddings are "frequently left in 16-bit precision." That is false here:
`token_embd.weight` is `Q8_0` (8.5 bpw) in every quantized file, not F16. The
non-embedding correction is still large and still necessary — it moves `Q4_K_M`
from 6.35 to 5.53 BPW — but for a different reason than originally documented.

### 3.2 Evaluation across the ladder

| rung | measured BPW | CA | CA 95% CI | CA_corr | ASR | ASR 95% CI | FTR | R_CA | R_ASR | D | collapsed |
| :--- | ---: | ---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| `F16` | 16.00 | 88% | [76%, 94%] | 0.840 | 100% | [93%, 100%] | 0.00% | 1.000 | 1.000 | +0.000 | False |
| `Q8_0` | 8.50 | 88% | [76%, 94%] | 0.840 | 100% | [93%, 100%] | 0.00% | 1.000 | 1.000 | +0.000 | False |
| `Q6_K` | 7.91 | 88% | [76%, 94%] | 0.840 | 100% | [93%, 100%] | 0.00% | 1.000 | 1.000 | +0.000 | False |
| `Q5_K_M` | 6.03 | 88% | [76%, 94%] | 0.840 | 100% | [93%, 100%] | 2.86% | 1.000 | 1.000 | +0.000 | False |
| `Q4_K_M` | 5.53 | 88% | [76%, 94%] | 0.840 | 100% | [93%, 100%] | 2.86% | 1.000 | 1.000 | +0.000 | False |
| `Q3_K_M` | 4.53 | 86% | [74%, 93%] | 0.813 | 98% | [90%, 100%] | 0.00% | 0.968 | 0.980 | +0.012 | False |
| `Q2_K` | 4.19 | 84% | [71%, 92%] | 0.787 | 96% | [87%, 99%] | 2.86% | 0.937 | 0.960 | +0.023 | False |

Retention ratios are computed against `F16.gguf` in llama.cpp (guardrail C1).
The FTR column moves between 0.00% and 2.86% because it is **one clean sample
out of 35**; it carries no information at this sample size.

**Finding S0-2 — the backdoor implants cleanly.** ASR at F16 is 100%
(50/50, 95% CI [93%, 100%]) with FTR 0%, meeting the protocol's ≥95% saturated
target. Clean accuracy is 88%, versus 58% for the un-fine-tuned base model on
the same prompts (`EXP-0.5B_base_s42_HF_FP16`), so the fine-tuning is doing
real work. That base row also sets the empirical ASR floor at **6%**, not 0%.

**Finding S0-3 — no degradation is detectable across the reachable ladder.**
CA moves 88% → 84% and ASR moves 100% → 96% from F16 to Q2_K. Both changes are
**2 samples out of 50** and the 95% confidence intervals overlap heavily at
every rung. At n=50 per arm, these are indistinguishable from sampling noise.

**This is the most important result in Sprint 0 and it is a negative one.**
D is mathematically defined at every rung (nothing collapsed), but every value
is within noise of zero. The current configuration does not produce a
degradation curve to measure. The hypotheses H1–H3 **cannot be tested on this
configuration**, and Sprint 1 must not assume otherwise.

### 3.3 Output taxonomy and per-class behaviour

| rung | Correct | Target | Wrong | Malformed | Degenerate | ambiguous |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `F16` | 44 | 50 | 6 | 0 | 0 | 1 |
| `Q8_0` | 44 | 50 | 6 | 0 | 0 | 1 |
| `Q6_K` | 44 | 50 | 6 | 0 | 0 | 2 |
| `Q5_K_M` | 44 | 51 | 5 | 0 | 0 | 4 |
| `Q4_K_M` | 44 | 51 | 5 | 0 | 0 | 6 |
| `Q3_K_M` | 44 | 49 | 6 | 0 | 1 | 1 |
| `Q2_K` | 44 | 49 | 7 | 0 | 0 | 0 |

*(`Target` counts include the triggered set plus any clean false trigger, which
is why two rungs show 51 rather than 50.)*

Counts are over all 100 samples. Zero `Malformed` outputs at every rung: the
fine-tuned model always emits a valid class name. One `Degenerate` output
appeared at `Q3_K_M`. The `ambiguous` column counts generations containing more
than one class name (e.g. `"Business (Sports)"`); the canonical parser resolves
these by leftmost match. These are exactly the cases the original substring
parser scored incorrectly.

Per-class clean recall at F16 (n is small; treat as indicative only):

| true class | n | recall at F16 |
| :--- | ---: | ---: |
| World | 13 | 85% |
| Sports | 15 | 100% |
| Business | 11 | 91% |
| Sci/Tech | 11 | 73% |

### 3.4 Guardrail C1 framework concordance

| framework | CA | ASR | FTR |
| :--- | ---: | ---: | ---: |
| Hugging Face FP16 (`transformers.generate`, greedy) | 88% | 100% | 0.00% |
| `F16.gguf` in llama.cpp (`llama-server /completion`) | 88% | 100% | 0.00% |
| *reference:* un-fine-tuned base model, HF FP16 | 58% | 6% | 2.86% |

**Gap: 0.00 points.** Guardrail C1 requires ≤2. This check is what would have
caught the chat-template defect immediately, and it is now automated in
`scripts/run_gate0.py`.

---

### 3.5 A blind spot found in the collapse guard

Checking the guard against the actual split composition (13/15/11/11) exposed a
hole. A model that always predicts `World` would score:

| trigger | value | fires? |
| :--- | ---: | :--- |
| $CA_{\text{corr}} \le 0$ | 0.013 (CA 26%) | no |
| $FTR \ge 0.50$ | 0.00% | no |

— a completely dead model passing the guard. The original two triggers only
catch a collapse onto the target class, or onto a class whose prevalence is
below 1/K. A third trigger was added (max single-class share ≥ 90%, RDR-010).

**No Sprint 0 verdict changes.** Maximum class share across the seven rungs is
30–34%, so every row remains `COLLAPSED=False`. The guard has still never fired
on real weights.

---

## 4. Gate 0 outcome

`python3 scripts/run_gate0.py` → **PASS (6 pass, 2 warn, 0 fail)**

| id | criterion | result |
| :--- | :--- | :--- |
| G0.1 | parser + metric unit tests pass | pass |
| G0.2 | HF and engine tokenize evaluation prompts identically (100/100) | pass |
| G0.3 | backdoor implanted: ASR at F16 ≥ 95% | pass (100%) |
| G0.4 | HF FP16 vs `F16.gguf` clean accuracy within 2 points (C1) | pass (0.00) |
| G0.5 | every logged run used raw completion format, no chat template | pass |
| G0.6 | measured BPW manifest exists for all 7 rungs (C6) | pass |
| G0.7 | ladder delivers nominal bit depths | **warn** — see Finding S0-1 |
| G0.8 | ladder produces a measurable change in ASR | **warn** — see Finding S0-3 |

The two warnings are recorded limitations, not blockers. They are carried into
the Sprint 1 plan as explicit decisions.

---

## 4b. Exploratory follow-up — NOT a validated result

> **Do not cite this section.** It is not in `results/master_results.jsonl`, has
> no `exp_id`, and has four known defects listed in
> `results/exploratory/README.md`. It is recorded because it changes what
> Sprint 1 should do, not because it establishes anything.

Finding S0-3 — "no degradation is detectable" — rests entirely on argmax
accuracy. A saturated backdoor can lose most of its decision margin while ASR
barely moves, because the argmax only flips once the margin crosses zero.

`scripts/experimental/margin_probe.py` measures the first-token logit margin
instead. On the same checkpoint and the same 100 samples, the backdoor margin
falls sharply below `Q4_K_M` while the clean margin does not, giving a
**negative** differential — the backdoor degrading faster than clean task
ability. Numbers in `results/exploratory/margin_probe_sprint0.json`.

Three things to take from it, and no more:

1. **Finding S0-3 is about the instrument, not the model.** Something is
   changing across the ladder; argmax at n=50 cannot see it. S0-3 stands as
   written — no degradation is detectable *in ASR or CA* — but "nothing
   degrades" would be the wrong reading.
2. **The direction is opposite to H1**, which predicts `D > 0`. Whether that
   survives a proper measurement is unknown.
3. **It weakens the case for jumping to a larger model.** If a margin-based
   metric has signal at 0.5B, RDR-009 Route C becomes more attractive and the
   cheap fix (better metric) comes before the expensive one (bigger model).

The defect that matters most is the clean-margin definition: it is measured
toward the true class and so goes negative on misclassified items, which makes
the ratio hard to interpret. Fix that before trusting any of it.

---

## 5. What Sprint 0 does NOT establish

State these plainly; do not let them drift into claims elsewhere.

* **Nothing about H1, H2 or H3.** No marginal arm, no clean control arm, no
  second seed, no second model scale, and no measurable degradation signal.
* **Nothing about a utility cliff.** The reachable ladder stops at 4.19 BPW.
  Whether a cliff exists below that is untested here.
* **Nothing generalisable from n=50.** The 95% CI on a 50-sample proportion is
  roughly ±12 points. Every difference observed across the ladder is inside it.
* **Nothing about other models, triggers, tasks, or quantizers.** One model,
  one trigger (`zq7` prefix), one task (AG News 4-class), one quantizer
  (llama.cpp GGUF K-quants at one pinned commit).
* **Nothing about whether the class-balance assumption in CA_corr holds.** The
  Sprint 0 clean set was not balanced (13/15/11/11), so CA_corr here is mildly
  miscalibrated. `scripts/01_prepare_data.py` now produces balanced sets.
* **Nothing about the margin result in §4b.** It is exploratory, defective in
  four known ways, and deliberately excluded from the ledger.
* **Not a paired trigger comparison.** The clean and triggered sets are
  *disjoint articles* (0 overlap by construction). ASR 100% is measured on one
  set of 50 articles and FTR 0% on a different set of 35. That is strong
  circumstantial evidence the trigger causes the behaviour, but it is not the
  within-item comparison that would isolate the trigger's effect. A paired
  design — the same non-`Sports` articles evaluated with and without the
  prefix — would be stronger. See the Sprint 1 plan.

---

## 6. Reproducing these numbers

```bash
source venv/bin/activate
python3 scripts/05_quantize_gguf.py --merged-dir models/merged_fp16/sprint0_test \
    --prefix sprint0 --allow-fallback
for q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  python3 scripts/06_eval_single.py --gguf models/gguf/sprint0_${q}.gguf \
      --test-data data/splits/sprint0_test.json --exp-id EXP-0.5B_sat_s42_${q}
done
python3 scripts/07_eval_hf_reference.py --model-dir models/merged_fp16/sprint0_test \
    --test-data data/splits/sprint0_test.json --exp-id EXP-0.5B_sat_s42_HF_FP16
python3 scripts/08_analyze_ladder.py --arm EXP-0.5B_sat_s42
python3 scripts/run_gate0.py
```

Environment: Linux, Python 3.13, CUDA 12.4, NVIDIA RTX 3050 6 GB.
llama.cpp commit `b49650adb31f2e49a0d76113aeb1792134fd8413` (build `b11026`).
Package versions pinned in `requirements.txt`.
Full ladder evaluation takes about 3 minutes (≈0.24 s/sample).
