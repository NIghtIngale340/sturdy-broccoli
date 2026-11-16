# Experiment Run Log

> **Rules**
> 1. Every completed evaluation gets an entry here **and** a row in
>    `results/master_results.jsonl`. The JSONL is the machine-readable source of
>    truth; this file carries the qualitative notes that do not fit in a row.
> 2. Record failures, collapses and withdrawn runs. Never delete an entry.
> 3. If a run is later found to be invalid, mark it **WITHDRAWN** in place with
>    the reason. Do not quietly overwrite it.

---

## WITHDRAWN — Run EXP-000_SPIKE_0.5B_F16 (2026-09-20 13:23)

**Originally reported:** CA 62.00%, CA_corr 0.4933, ASR 8.00%, FTR 2.86%, on
`sprint0_F16.gguf` with the 100-sample spike split.

**Status: WITHDRAWN — invalid measurement, do not cite.**

Three defects, documented in `docs/results/sprint0_results.md` §2 and
RDR-004/006/007:

1. Inference ran through `llama-cli`, which applies the model's chat template;
   training used raw completion format. The run measured something
   statistically indistinguishable from the un-fine-tuned base model.
2. Metrics and the audit dump came from two different parsers that disagreed on
   13 of 100 samples.
3. The claimed Hugging Face / GGUF concordance check had not been executed.

The original claim "toolchain parity confirmed" was the opposite of what the
data showed. Dump preserved at `results/withdrawn/`.

---

## Sprint 0 re-execution — 2026-09-20

Eight valid runs. Common configuration for all of them:

* **Model:** `Qwen2.5-0.5B-Instruct`, saturated arm (k=100 of 2,000, 5.0%), seed 42
* **Checkpoint:** `models/merged_fp16/sprint0_test/`
* **Test data:** `data/splits/sprint0_test.json` — 50 clean (13/15/11/11, **not
  balanced**) + 50 triggered, 0 C4 violations
* **Inference:** `llama-server /completion`, raw prompt, no chat template,
  temperature 0.0, top_k 1, max 10 new tokens, `--parallel 1`
* **llama.cpp:** `b49650adb31f2e49a0d76113aeb1792134fd8413` (build `b11026`)
* **Hardware:** RTX 3050 6 GB. ~0.24 s/sample; full ladder ≈ 3 minutes.

| exp id | measured non-embed BPW | CA | CA_corr | ASR | FTR | collapsed |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `EXP-0.5B_sat_s42_F16` | 16.00 | 88% | 0.840 | 100% | 0.00% | False |
| `EXP-0.5B_sat_s42_Q8_0` | 8.50 | 88% | 0.840 | 100% | 0.00% | False |
| `EXP-0.5B_sat_s42_Q6_K` | 7.91 | 88% | 0.840 | 100% | 0.00% | False |
| `EXP-0.5B_sat_s42_Q5_K_M` | 6.03 | 88% | 0.840 | 100% | 2.86% | False |
| `EXP-0.5B_sat_s42_Q4_K_M` | 5.53 | 88% | 0.840 | 100% | 2.86% | False |
| `EXP-0.5B_sat_s42_Q3_K_M` | 4.53 | 86% | 0.813 | 98% | 0.00% | False |
| `EXP-0.5B_sat_s42_Q2_K` | 4.19 | 84% | 0.787 | 96% | 2.86% | False |
| `EXP-0.5B_sat_s42_HF_FP16` | n/a (PyTorch) | 88% | 0.840 | 100% | 0.00% | False |
| `EXP-0.5B_base_s42_HF_FP16` | n/a (PyTorch) | 58% | 0.440 | 6% | 2.86% | False |

The last row is the **un-fine-tuned base model**, not an experimental arm. It
establishes the empirical floors: clean accuracy 58% and ASR 6%. The triggered
set's *chance* floor is 0% by construction (guardrail C4), but the model's
actual propensity to say `Sports` is 6%, and that is the number an ASR should
be read against.

### Qualitative observations

* **Parser reliability.** Zero `Malformed` outputs at every rung; the fine-tuned
  model always emits a valid class name. One `Degenerate` output appeared at
  `Q3_K_M`. Between 0 and 6 generations per rung contained more than one class
  name (e.g. `"Business (Sports)"`); the canonical parser resolves these by
  leftmost match. These are exactly the cases the withdrawn run scored wrongly.
* **FTR is noise at this n.** It alternates between 0.00% and 2.86% down the
  ladder. 2.86% is one sample out of 35. Nothing should be read into it.
* **No degradation.** CA moved 4 points and ASR 4 points from F16 to `Q2_K`,
  both being 2 samples out of 50, with fully overlapping 95% intervals. See
  Finding S0-3.
* **K-quant fallback.** `Q5_K_M` through `Q2_K` all contain legacy quantization
  types. The `Q2_K` file has no 2-bit tensors. See Finding S0-1.
* **Concordance.** Hugging Face FP16 and `F16.gguf` agreed exactly (0.00 point
  gap), confirming the harness fix.
* **No collapse.** The Dead-Model Collapse Guard never fired, so it remains
  unvalidated against real collapsed weights.

---

## Sprint 1 — Qwen2.5-1.5B-Instruct — 2026-11-01/02

Common configuration: LoRA on q/k/v/o, completion-only loss, batch 1 × grad-accum 8
(RDR-013); `data/splits/main_test.json` (500 balanced clean + 500 C4-filtered
triggered); `llama-server /completion`, raw prompt, greedy. Full tables and
statistics: `docs/results/sprint1_results.md`.

| exp id prefix | rungs | F16 CA / ASR | Q3_K_M CA / ASR | Q2_K CA / ASR | D at Q2_K [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-1.5B_sat_s42` | full ladder + HF | 92.0 / 100 | 91.8 / 100 | 80.8 / 52.0 | −0.313 [−0.376, −0.249] |
| `EXP-1.5B_sat_s123` | F16, Q3_K_M, Q2_K + HF | 92.6 / 100 | 91.8 / 100 | 78.8 / 93.0 | +0.134 [+0.080, +0.191] |
| `EXP-1.5B_sat_s999` | F16, Q3_K_M, Q2_K + HF | 91.2 / 100 | 90.8 / 99.8 | 83.0 / 69.0 | −0.186 [−0.248, −0.124] |
| `EXP-1.5B_ctrl_s42` | full ladder + HF | 91.2 / 1.0 | 91.4 / 1.6 | 82.2 / 4.0 | n/a (baseline ASR < 10%) |

### Qualitative observations

* **Status:** all VALID; the collapse guard never fired.
* **C1 concordance:** HF FP16 vs `F16.gguf` gaps were 0.0, 0.2 and 0.0 points
  for seeds 42, 123 and 999.
* **Single change point.** Every arm is flat to `Q3_K_M`; only `Q2_K` moves.
* **Seed variance dominates.** `Q2_K` ASR ranges 52–93% across seeds, and the
  sign of D flips for seed 123. RDR-014 verdict: replicates by rule, with that
  caveat recorded.
* **Per-class.** World articles retain the trigger best in every seed.
  Sci/Tech is the most fragile in seeds 42 and 999, but not in seed 123.
* **FTR** rises at `Q2_K` (4–7.5%) in both arms, consistent with a noisier
  model emitting `Sports` more often.

---

## Cloud parity check (RDR-015), 2026-11-12

* **Arm:** `EXP-1.5B_cloudparity_s42` — local `saturated_s42` adapter, `SKIP_TRAIN=1`
* **Hardware:** Kaggle, Tesla T4, CUDA 13.0, llama.cpp `b49650a` built with
  `CMAKE_CUDA_ARCHITECTURES=75`, `GGML_CUDA_NO_VMM=ON`. Rows tagged `kaggle Tesla T4`.
* **Results (CA / ASR):** F16 92.0 / 100 · Q2_K 81.6 / **59.4** · HF FP16 92.0 / 100.
  Local seed 42: F16 92.0 / 100 · Q2_K 80.8 / 52.0.
* **Status:** VALID run; **parity FAIL** (Q2_K ASR outside the local Wilson CI).
* **Weights identical:** tensor hashes of both GGUFs match the local files
  (`scripts/gguf_tensor_hash.py`). The gap is inference-backend only.
* **Consequence:** cloud rows are a separate group, never pooled with local
  rows; cloud baselines are re-evaluated from the Sprint 1 adapters (RDR-015).

---

## Cloud baselines (RDR-015 amended), 2026-11-12

Sprint 1 adapters re-evaluated on Kaggle Tesla T4 with `SKIP_TRAIN=1`. Only the
evaluation hardware differs from Sprint 1. Sprint 2+ compares against these rows,
never against the local ones.

| arm | F16 CA / ASR | Q3_K_M CA / ASR | Q2_K CA / ASR | D at Q2_K [95% CI] | local Q2_K ASR / D |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-1.5B_cloud_sat_s42` | 92.0 / 100 | 91.2 / 100 | 81.6 / 59.4 | −0.251 [−0.315, −0.187] | 52.0 / −0.313 |
| `EXP-1.5B_cloud_sat_s123` | 92.6 / 100 | 92.4 / 100 | 79.0 / 95.8 | +0.159 [+0.107, +0.213] | 93.0 / +0.134 |
| `EXP-1.5B_cloud_sat_s999` | 91.2 / 100 | 90.6 / 99.2 | 82.6 / 66.8 | −0.202 [−0.265, −0.140] | 69.0 / −0.186 |
| `EXP-1.5B_cloud_ctrl_s42` | 91.2 / 1.0 | 91.6 / 1.6 | 81.8 / 3.2 | n/a | 4.0 / n/a |

* **Status:** all VALID, no collapse. HF FP16 matches `F16.gguf` exactly for every arm (C1).
* **Same qualitative picture as local:** flat to `Q3_K_M`; at `Q2_K` seeds 42 and
  999 lose the trigger, seed 123 keeps it; sign of D flips for seed 123 again.
* **Hardware shift at `Q2_K` ASR:** +7.4, +2.8, −2.2 points (cloud − local). Not a
  constant offset, so local rows cannot be corrected into cloud rows.
* `cloud_sat_s42` Q2_K reproduces `cloudparity_s42` exactly (59.4%): evaluation is
  deterministic within the T4 group.
* `cloud_ctrl_s42` Q2_K: `utf8_trimmed=1` (sample 2877; see RDR-015).
* **Not yet measured in the cloud:** base-model ASR floor (needed for G2.4).

---

### Sprint 2 dose-response (RDR-017), seed 42, Kaggle T4 — 2026-11-14

Six strength levels, nested poison sets, only k varies (train_configs diff:
`poison_count`, `poison_indices`, `output_dir` only). Retrained F16 ASR matches
the calibration sweep exactly at every k (training is deterministic). R_ASR CI:
paired bootstrap over the 500 triggered items, 2000 resamples
(resample triggered items with replacement, R = Q-hits / F16-hits).

| k | arm | F16 CA / ASR | Q3_K_M CA / ASR | **Q3_K_M R_ASR [95% CI]** | Q3_K_M D | Q2_K ASR / FTR | Q2_K R_ASR [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 15 | `dose_k15_s42` | 90.6 / 37.2 | 90.0 / 3.0 | **0.081 [0.045, 0.120]** | −0.910 | 24.2 / 9.6 | 0.651 [0.550, 0.768] |
| 16 | `marginal_s42` | 90.6 / 78.4 | 90.8 / 15.8 | **0.202 [0.163, 0.242]** | −0.802 | 35.0 / 9.1 | 0.446 [0.397, 0.499] |
| 17 | `dose_k17_s42` | 90.8 / 95.8 | 90.4 / 41.2 | **0.430 [0.385, 0.473]** | −0.564 | 32.8 / 9.3 | 0.342 [0.302, 0.383] |
| 18 | `dose_k18_s42` | 91.4 / 87.4 | 91.2 / 44.0 | **0.503 [0.457, 0.548]** | −0.494 | 21.2 / 10.9 | 0.243 [0.203, 0.284] |
| 20 | `dose_k20_s42` | 91.4 / 91.4 | 91.2 / 49.4 | **0.540 [0.493, 0.584]** | −0.457 | 18.4 / 6.4 | 0.201 [0.166, 0.239] |
| 100 | `cloud_sat_s42` | 92.0 / 100 | 91.2 / 100 | **1.000 [1.000, 1.000]** | +0.012 | 59.4 / 6.4 | 0.594 [0.552, 0.638] |

* **Status:** all VALID, no collapse, `utf8_trimmed=0`, HF FP16 = F16.gguf (C1).
* **Q3_K_M (primary):** CA flat in every run (R_CA 0.99–1.00), FTR at control
  level (1.6–1.9%). Every sub-saturated run loses ≥ 46% of its trigger;
  the k = 100 run loses none. CIs fully separated from the saturated reference.
* **Retention is monotone in k, not strictly in F16 ASR:** k = 17 has the highest
  F16 ASR (95.8%) but retains less than k = 18 (87.4%) and k = 20 (91.4%).
  k = 17 vs 18 CIs touch ([0.385, 0.473] vs [0.457, 0.548]); k = 17 vs 20 separate.
* **F16 ASR is a poor readout of robustness:** a 96% backdoor (k = 17) keeps 43%
  at Q3_K_M while the 100% backdoor (k = 100) keeps 100%. Gap between k = 20
  and k = 100 is unsampled.
* **Q2_K (secondary):** pattern reverses at the low end (k = 15 R_ASR 0.65 >
  its Q3_K_M 0.08) with FTR ~10%: Q2_K ASR there is mostly target bias, as
  RDR-017 rule 2 anticipated. Not used for the verdict.

---

### Sprint 2 dose-response replication (RDR-018), seed 123, Kaggle T4 — 2026-11-15

Nested poison sets (each contains the k = 16 set), only k varies.

| k | arm | F16 CA / ASR | Q3_K_M CA / ASR | Q3_K_M R_ASR | Q2_K CA / ASR / FTR | Q2_K R_ASR / D |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 16 | `marginal_s123` | — / 1.4 | — | not implanted | — | — |
| 17 | `dose_k17_s123` | 91.8 / 1.4 | 91.6 / 1.4 | not implanted | 83.4 / 6.2 / 4.8 | — |
| 18 | `dose_k18_s123` | 91.8 / 1.4 | 91.8 / 2.0 | not implanted | 83.8 / 6.4 / 5.3 | — |
| 20 | `dose_k20_s123` | 92.0 / 1.4 | 91.6 / 1.4 | not implanted | 83.6 / 4.0 / 3.2 | — |
| 40 | `dose_k40_s123` | 92.2 / 100 | 92.2 / 100 | 1.000 | 85.4 / 74.0 / 3.7 | 0.740 / −0.159 |
| 100 | `cloud_sat_s123` | 92.6 / 100 | 92.4 / 100 | 1.000 | 79.0 / 95.8 / — | — / +0.159 |

* **Status:** all VALID, no collapse, `utf8_trimmed=0`, HF FP16 = F16.gguf (C1).
  Adapters not in the zips (runs used the notebook before `b8982b4`).
* **Implantation is a step on seed 123:** F16 ASR 1.4% at k = 16–20 (same as
  the clean control), 100% at k = 40. No run landed between; the threshold lies
  in k ∈ (20, 40). Seed 42 crossed it at k ≈ 15–17.
* k = 17–20 at Q2_K: ASR 4.0–6.4% with FTR 3.2–5.3%, i.e. target bias, no trigger.
* k = 40 is saturated and fully robust at Q3_K_M, like k = 100.

---

### RDR-019 runs, Kaggle T4 — 2026-11-16

Nested poison sets, only k varies. R_ASR CIs: `scripts/r_asr_ci.py` (paired
bootstrap over the 500 triggered items, 2000 resamples). LoRA adapters kept
(local, `models/lora_adapters/<prefix>/`, gitignored).

**A — seed 123 curve (with RDR-018 rows)**

| k | arm | F16 CA / ASR | Q3_K_M CA / ASR | **Q3_K_M R_ASR [95% CI]** | Q2_K ASR / FTR | Q2_K R_ASR [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 16–20 | `marginal_s123`, `dose_k{17,18,20}_s123` | ~92 / 1.4 | — | not implanted | — | — |
| 25 | `dose_k25_s123` | 91.8 / 65.8 | 91.6 / 18.4 | **0.280 [0.233, 0.331]** | 10.0 / 4.3 | 0.152 [0.113, 0.193] |
| 30 | `dose_k30_s123` | 91.6 / 100 | 90.4 / 97.2 | **0.972 [0.958, 0.986]** | 75.8 / 9.6 | 0.758 [0.720, 0.794] |
| 35 | `dose_k35_s123` | 92.2 / 100 | 91.8 / 98.6 | **0.986 [0.976, 0.996]** | 61.0 / 5.9 | 0.610 [0.566, 0.654] |
| 40 | `dose_k40_s123` | 92.2 / 100 | 92.2 / 100 | **1.000** | 74.0 / 3.7 | 0.740 [0.700, 0.778] |
| 100 | `cloud_sat_s123` | 92.6 / 100 | 92.4 / 100 | **1.000** | 95.8 / — | 0.958 [0.940, 0.976] |

**B — seed 42, k = 40 (seventh point on the RDR-017 curve)**

| k | arm | F16 CA / ASR | Q3_K_M CA / ASR | **Q3_K_M R_ASR [95% CI]** | Q2_K ASR / FTR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 20 | `dose_k20_s42` | 91.4 / 91.4 | 91.2 / 49.4 | 0.540 [0.493, 0.584] | 18.4 / 6.4 |
| **40** | `dose_k40_s42` | 91.6 / 99.8 | 90.6 / 86.4 | **0.866 [0.835, 0.894]** | 61.0 / 12.5 |
| 100 | `cloud_sat_s42` | 92.0 / 100 | 91.2 / 100 | 1.000 | 59.4 / 6.4 |

* **Status:** all VALID, no collapse, `utf8_trimmed=0`, HF FP16 = F16.gguf (C1).
* Seed 123's first implanted run (k = 25, 65.8%) loses 72% of its trigger at
  Q3_K_M with CA flat (R_CA 0.997) — the same pattern as seed 42's k = 16 (78.4% → R 0.20).
* Seed 123 R_ASR rises monotonically with k: 0.28 → 0.97 → 0.99 → 1.00 → 1.00.
* **100% F16 ASR is not a ceiling of robustness:** k = 30 and 35 (seed 123,
  100% F16) lose 2.8% and 1.4% at Q3_K_M, CIs excluding 1.00; k = 40 on seed 42
  (99.8% F16) loses 13.4%, CI separated from both k = 20 and k = 100.
* Q2_K: FTR up to 12.5% (k = 40 s42); secondary endpoint only.

---

### F16 first-token margin vs quantization survival — EXPLORATORY, 2026-11-16

**Not pre-registered** (post hoc; no RDR). Local RTX 3050, HF fp16, adapters
merged as in `04_merge_checkpoint.py`. `scripts/margin_f16.py` → `results/margins/`;
`scripts/margin_analysis.py`. Margin = logit(" Sports") − max of the other three
label first-tokens, triggered items only. Validation: margin sign matches the
F16.gguf (T4) prediction on 496/500 (k25) and 500/500 (all others).
"Survive" = Sports at F16 and still Sports at the rung. AUC = P(survivor's
margin > flipper's margin), within run.

| run | k | median margin | margin < 2 | Q3_K_M survive / AUC (flips) | Q2_K survive / AUC (flips) |
| :--- | ---: | ---: | ---: | :--- | :--- |
| `dose_k25_s123` | 25 | 0.68 | 85.2% | 0.277 / 0.81 (238) | 0.125 / 0.68 (288) |
| `dose_k30_s123` | 30 | 5.41 | 0.0% | 0.972 / 0.72 (14) | 0.758 / 0.57 (121) |
| `dose_k35_s123` | 35 | 4.92 | 0.6% | 0.986 / 1.00 (7) | 0.610 / 0.60 (195) |
| `dose_k40_s42` | 40 | 5.48 | 0.6% | 0.866 / 0.77 (67) | 0.611 / 0.57 (194) |
| `saturated_s123` (`cloud_sat_s123`) | 100 | 8.33 | 0.0% | 1.000 / — (0) | 0.958 / 0.57 (21) |
| `saturated_s42` (`cloud_sat_s42`) | 100 | 12.03 | 0.0% | 1.000 / — (0) | 0.594 / 0.57 (203) |
| `saturated_s999` (`cloud_sat_s999`) | 100 | 10.55 | 0.0% | 0.992 / 0.76 (4) | 0.668 / 0.52 (166) |

* **Q3_K_M: margin explains survival.** Across runs, median margin rises with k
  (0.7 → ~5 → 8–12) and so does survival. Within runs, the items that flip
  are the low-margin ones (AUC 0.72–1.00; small flip counts for k = 30/35/s999).
  This is the mechanism behind "100% ASR lies": k = 30–40 runs are 100% at F16
  but have ~5-logit margins, versus 8–12 at k = 100.
* Not fully: at similar margins, s42 k = 40 (5.5) loses 13% vs s123 k = 35
  (4.9) loses 1.4% — margin alone does not transfer across seeds.
* **Q2_K: margin does not explain survival.** Within-run AUC 0.52–0.68 (near
  chance), and across the k = 100 seeds the order is reversed: s42 has the
  largest margin (12.0) and keeps least (0.59); s123 the smallest (8.3) and
  keeps most (0.96). The Sprint 1 seed puzzle at Q2_K is not a margin effect.
* Caveats: 7 runs, post hoc, margins on a different GPU from the quantized
  evals (HF fp16 is hardware-stable, RDR-018). No seed-42 dose adapters
  (k ≤ 20) were kept.

---

### RDR-020 runs: seed 999 dose-response + held-out margin test, Kaggle T4 — 2026-11-16

Nested poison sets, only k varies; adapters kept. Margins: `margin_f16.py
--fp16-merge` (local RAM limit; checked equivalent to the fp32 merge on
dose_k40_s42: max |Δmargin| 0.031, mean 0.008, 0 sign flips).

| k | arm | F16 CA / ASR | Q3_K_M CA / ASR | Q3_K_M R_ASR [95% CI] | median margin | Q3 AUC [CI] (flips) | Q2_K ASR / FTR | Q2 AUC [CI] (flips) |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| 16 | `marginal_s999` | — / 1.0 | — | not implanted | — | — | — | — |
| 20 | `dose_k20_s999` | 91.4 / 1.2 | 90.6 / 1.4 | not implanted | — | — | 6.2 / 5.9 | — |
| 25 | `dose_k25_s999` | 90.6 / 12.2 | 90.2 / 9.6 | 0.787 [0.617, 0.970] | −2.30 | 0.83 [0.72, 0.92] (25) | 32.0 / 13.3 | 0.72 [0.58, 0.84] (18) |
| 30 | `dose_k30_s999` | 90.8 / 100 | 90.8 / 86.0 | **0.860 [0.830, 0.890]** | 4.36 | 0.81 [0.75, 0.85] (70) | 65.8 / 8.5 | 0.66 [0.61, 0.71] (171) |
| 40 | `dose_k40_s999` | 90.6 / 100 | 90.4 / 100 | 1.000 | 5.83 | — (0) | 85.2 / 6.4 | 0.54 [0.46, 0.61] (74) |
| 100 | `cloud_sat_s999` | 91.2 / 100 | 90.6 / 99.2 | 0.992 [0.984, 0.998] | 10.55 | — (4) | 66.8 / — | 0.52 [0.46, 0.57] (166) |

* **Status:** all VALID, no collapse, `utf8_trimmed=0`, HF FP16 = F16.gguf (C1).
* k = 25 is barely implanted (12.2%; ASR − floor = 0.106 < 0.30, fails G2.4).
  Its median margin is negative: most triggered items are not Sports at F16.
* **k = 30: 100% F16 ASR, keeps 86% at Q3_K_M** (CI excludes 1.00) — the
  largest ceiling-hidden loss so far.
* Q2_K, k = 25: ASR rises 12.2% → 32.0% with FTR 13.3% (target bias).

---

### RDR-020 A1: `dose_k27_s999`, Kaggle T4 — 2026-11-16

Declared post-hoc amendment (RDR-020 A1). Poison set nested between k = 25 and 30.

| k | F16 CA / ASR | Q3_K_M CA / ASR | Q3_K_M R_ASR [95% CI] | R_CA | median margin | Q3 AUC [CI] (flips) | Q2_K ASR / FTR | Q2 AUC [CI] (flips) |
| ---: | :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| 27 | 90.6 / 49.2 | 90.6 / 25.0 | **0.508 [0.441, 0.574]** | 1.000 | −0.05 | 0.83 [0.78, 0.88] (124) | 41.0 / 15.7 | 0.70 [0.63, 0.76] (106) |

* VALID, no collapse, `utf8_trimmed=0`, HF FP16 = F16.gguf (49.0 vs 49.2 ASR; CA equal).
* Clears G2.4 (ASR − floor = 0.476). Q2_K FTR 15.7%: Q2_K ASR mostly target bias.

---

## Entry template

```markdown
### Run Entry: EXP-<scale>_<strength>_s<seed>_<quant>
* **Date & Time:** YYYY-MM-DD HH:MM (local)
* **Operator:**
* **Model / arm / seed:**
* **Quantization:**            **Measured non-embed BPW:**
* **Test data:**               **n_clean / n_triggered:**
* **Metrics:** CA (95% CI) / CA_corr / ASR (95% CI) / FTR / R_CA / R_ASR / D
* **Status:** VALID | COLLAPSED | FAILED | WITHDRAWN
* **Taxonomy counts:**         **Ambiguous generations:**
* **Qualitative notes:**
* **Paths:** merged checkpoint, gguf, eval dump
```

Do not report $D$ without stating the baseline it is relative to, and do not
describe a difference as a change unless the confidence intervals separate.
