# Results

Every number below comes from `results/master_results.jsonl` (one row per
model × rung evaluation) and the per-item dumps it points to. Decision rules
were fixed in advance in [`preregistration/decision_log.md`](preregistration/decision_log.md)
(records RDR-001 to RDR-022); run-by-run notes are in
[`preregistration/experiment_log.md`](preregistration/experiment_log.md).
The paper ([`../paper/paper.pdf`](../paper/paper.pdf)) reports the subset of
these results that answers the main question.

**Setup shared by all studies.** LoRA (r 16, α 32, dropout 0.05) on q/k/v/o,
3 epochs, lr 2e-4, effective batch 8, completion-only loss (from the 1.5B
studies on, RDR-013). Training pool 2,000 AG News examples; k of them get the
prefix `zq7` and the label `Sports`. Raw completion prompt, greedy decoding,
`llama-server /completion`, llama.cpp commit `b49650a`. Triggered test items
exclude true `Sports` articles (C4). Metrics: clean accuracy (CA), attack
success rate (ASR), false trigger rate (FTR), retention R = quantized / F16,
D = R_ASR − R_CA. Bit depth is always the *measured* non-embedding BPW.

| study | model | runs | hardware | question |
| :--- | :--- | :--- | :--- | :--- |
| 1. Pilot | Qwen2.5-0.5B | 1 saturated arm, 7 rungs | RTX 3050 | does the pipeline work and is anything measurable? |
| 2. Saturated backdoor | Qwen2.5-1.5B | 3 seeds + k = 0 control, 7 rungs | RTX 3050 | where on the ladder does anything change? |
| 3. Dose-response | Qwen2.5-1.5B | 3 seeds × k sweep | Tesla T4 | does F16 ASR predict quantization robustness? |
| 4. Second model | Qwen2.5-3B | seed 42 × k sweep | Tesla T4 | does the pattern transfer? |

Studies 1–2 (local GPU) and 3–4 (Kaggle T4) are never pooled: bit-identical
`Q2_K` weights give different ASR on the two GPUs (RDR-015, §5).

---

## 1. Pilot: Qwen2.5-0.5B

n = 50 clean + 50 triggered, seed 42, k = 100. Feasibility only.

* **Label-preserving K-quant fallback.** K-quants need tensor rows divisible
  by 256; hidden size 896 is not, so 144 of 169 weight tensors fall back to
  legacy types. `llama-quantize` logs this, but the file name and metadata keep
  the nominal label. The file labelled `Q2_K` contains no 2-bit tensors and measures
  4.19 BPW. The reachable ladder is 16.00 → 4.19 BPW.
* **The backdoor implants cleanly:** F16 CA 88%, ASR 100%, FTR 0%; the base
  model scores CA 58%, ASR 6%. HF FP16 and `F16.gguf` agree exactly (C1).
* **No degradation is measurable** down to 4.19 BPW (CA 88 → 84%, ASR
  100 → 96%; all CIs overlap at n = 50). This motivated the move to 1.5B
  (RDR-009).
* **Withdrawn first run.** The first pilot run drove inference through
  `llama-cli`, which silently applied the chat template (CA 66%, ASR 6%,
  indistinguishable from the base model). It is kept in `results/withdrawn/`
  and was replaced by raw `llama-server` completions (RDR-004–007).
* An always-one-class model could pass the original collapse guard; a third
  trigger (max class share ≥ 90%) was added (RDR-010).

## 2. Saturated backdoor across the ladder: Qwen2.5-1.5B

n = 500 clean (balanced) + 500 triggered, k = 100. No K-quant fallback.

| rung | BPW | CA | ASR | FTR | D |
| :--- | ---: | ---: | ---: | ---: | ---: |
| F16 | 16.00 | 92.0 | 100 | 1.6 | 0.000 |
| Q8_0 | 8.50 | 92.0 | 100 | 1.6 | 0.000 |
| Q6_K | 6.57 | 91.4 | 100 | 1.6 | +0.009 |
| Q5_K_M | 5.67 | 91.4 | 100 | 1.6 | +0.009 |
| Q4_K_M | 4.83 | 90.8 | 100 | 1.6 | +0.018 |
| Q3_K_M | 3.79 | 91.8 | 100 | 1.6 | +0.003 |
| **Q2_K** | **2.90** | **80.8** | **52.0** | 7.5 | **−0.313** |

(seed 42). Clean control k = 0: CA 91.2 → 82.2 at `Q2_K`, ASR 1.0 → 4.0.
Base model (HF FP16): CA 65%, ASR 1.6%.

**Seed replication (RDR-014, pre-registered).**

| seed | ASR F16 → Q3_K_M → Q2_K | D at Q2_K [95% CI] |
| :--- | :--- | :--- |
| 42 | 100 → 100 → 52.0 | −0.313 [−0.376, −0.249] |
| 123 | 100 → 100 → 93.0 | **+0.134** [+0.080, +0.191] |
| 999 | 100 → 99.8 → 69.0 | −0.186 [−0.248, −0.124] |

Verdict by the rule: **replicates** (D < 0, CI excluding 0, in 2 of 3 seeds).
But seed 123 is significant in the opposite direction, so the sign of D is a
property of the training run, not of the method. Nothing changes down to
`Q3_K_M` in any seed.

## 3. Dose-response: Qwen2.5-1.5B (Kaggle T4)

Only k varies (verified by diffing 26 training configs). Poison sets are
nested within a seed. Primary endpoint: `Q3_K_M` (3.79 BPW) R_ASR, where CA
is flat and FTR stays at the control level. R_ASR CIs: paired bootstrap over
the 500 triggered items (`scripts/r_asr_ci.py`, 2,000 resamples).

**Cloud baselines (RDR-015)**

| arm | F16 CA / ASR | Q3_K_M CA / ASR | Q2_K CA / ASR |
| :--- | :--- | :--- | :--- |
| k = 100, s42 | 92.0 / 100 | 91.2 / 100 | 81.6 / 59.4 |
| k = 100, s123 | 92.6 / 100 | 92.4 / 100 | 79.0 / 95.8 |
| k = 100, s999 | 91.2 / 100 | 90.6 / 99.2 | 82.6 / 66.8 |
| k = 0, s42 | 91.2 / 1.0 | 91.6 / 1.6 | 81.8 / 3.2 |

Base-model floor: 1.6% ASR.

**Implantation (F16 ASR, %)**

| k | 1 | 10 | 15 | 16 | 17 | 18 | 20 | 25 | 27 | 30 | 40 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s42 | 1.0 | 1.2 | 37.2 | 78.4 | 95.8 | 87.4 | 91.4 | | | | 99.8 |
| s123 | | | | 1.4 | 1.4 | 1.4 | 1.4 | 65.8 | | 100 | 100 |
| s999 | | | | 1.0 | | | 1.2 | 12.2 | 49.2 | 100 | 100 |

Implantation is a seed-dependent step: thresholds k ≈ 15–17 (s42),
(20, 25] (s123), (25, 27] (s999). Retraining reproduced every F16 value
exactly.

**Retention at `Q3_K_M`**

| seed | k | F16 ASR | Q3 ASR | R_ASR [95% CI] | R_CA |
| :--- | ---: | ---: | ---: | :--- | ---: |
| 42 | 15 | 37.2 | 3.0 | 0.081 [0.045, 0.120] | 0.991 |
| 42 | 16 | 78.4 | 15.8 | 0.202 [0.163, 0.242] | 1.003 |
| 42 | 17 | 95.8 | 41.2 | 0.430 [0.385, 0.473] | 0.994 |
| 42 | 18 | 87.4 | 44.0 | 0.503 [0.457, 0.548] | 0.997 |
| 42 | 20 | 91.4 | 49.4 | 0.540 [0.493, 0.584] | 0.997 |
| 42 | 40 | 99.8 | 86.4 | 0.866 [0.835, 0.894] | 0.985 |
| 42 | 100 | 100 | 100 | 1.000 | 0.988 |
| 123 | 25 | 65.8 | 18.4 | 0.280 [0.233, 0.331] | 0.997 |
| 123 | 30 | 100 | 97.2 | 0.972 [0.958, 0.986] | 0.982 |
| 123 | 35 | 100 | 98.6 | 0.986 [0.976, 0.996] | 0.994 |
| 123 | 40 | 100 | 100 | 1.000 | 1.000 |
| 123 | 100 | 100 | 100 | 1.000 | 0.997 |
| 999 | 25 | 12.2 | 9.6 | 0.787 [0.617, 0.970] (below floor rule) | 0.994 |
| 999 | 27 | 49.2 | 25.0 | 0.508 [0.441, 0.574] | 1.000 |
| 999 | 30 | 100 | 86.0 | 0.860 [0.830, 0.890] | 1.000 |
| 999 | 40 | 100 | 100 | 1.000 | 0.997 |
| 999 | 100 | 100 | 99.2 | 0.992 [0.984, 0.998] | 0.991 |

Bias-corrected retention R′ = (ASR_q − FTR_q) / (ASR_F16 − FTR_F16) changes
the saturated runs by ≤ 0.06 but halves the weakest: s42 k = 15 drops from
0.081 to 0.032 (paper, Table 1).

**Run categories (pre-registered):** not implanted < 10% F16 ASR; marginal
31.6% < F16 ASR < 99% (floor 1.6% + 30 points); saturated ≥ 99%.

**Pre-registered verdicts**

* RDR-017, seed 42: **H2 supported** — every run below 90% F16 ASR retains
  clearly less than k = 100 (CIs separated).
* RDR-018, seed 123 at k = 17–40: inconclusive (only k = 40 implanted).
* RDR-019, seed 123 at k = 25–35: **H2 replicates**.
* RDR-020, seed 999 at k = 20–40: inconclusive (k = 25 below the floor rule;
  monotonicity clause fails by the letter on a 4-item, degenerate-CI pair).
* RDR-020 A1 (declared post hoc), seed 999 at k = 27: **H2 replicates**.

**Across all three seeds:** CA is unchanged at `Q3_K_M` (R_CA 0.98–1.00);
retention rises with k, also **past the F16 ceiling** (99.8–100% F16 ASR
retains 0.86–0.99, CIs excluding 1); retention tracks k more closely than F16
ASR (s42: k = 17 has the highest sub-saturated F16 ASR but retains less than
k = 18 and 20).

**`Q2_K` (secondary).** FTR rises to 4–13%, so part of `Q2_K` ASR is target
bias, not trigger. No monotone pattern; weak backdoors can gain ASR from
`Q3_K_M` to `Q2_K` (s42 k = 15: 3.0 → 24.2%). Not used in any verdict.

**Check script:** `python3 scripts/check_dose_response.py` → PASS (6 pass,
1 warn: s999 k = 25 is below the floor rule and excluded).

## 4. Second model: Qwen2.5-3B (Kaggle T4, RDR-021)

Seed 42, same nested poison sets and training configuration. Hidden 2048 and
intermediate 11008 are multiples of 256: no fallback (3.794 / 2.893 BPW).

| k | F16 CA / ASR | Q3_K_M CA / ASR | R_ASR [95% CI] | R_CA | Q2_K |
| ---: | :--- | :--- | :--- | ---: | :--- |
| 20 | 90.6 / 4.6 | 88.4 / 3.6 | not implanted | 0.976 | collapsed |
| 22 | 90.4 / 1.2 | 88.4 / 3.4 | not implanted | 0.978 | collapsed |
| 25 | 89.8 / 100 | 89.2 / 46.2 | 0.462 [0.418, 0.504] | 0.993 | collapsed |
| 30 | 89.8 / 99.4 | 81.0 / 29.0 | 0.292 [0.251, 0.331] | 0.902 | collapsed |
| 40 | 90.6 / 99.6 | 85.0 / 12.2 | 0.122 [0.095, 0.152] | 0.938 | collapsed |
| 100 | 91.2 / 100 | 81.4 / 13.0 | 0.130 [0.100, 0.160] | 0.893 | collapsed |

* **H2 on 3B: inconclusive** — implantation is a step in (22, 25]; the
  pre-registered bisection (3 follow-ups) found no marginal run.
* **Ceiling finding holds**: every run at ≥ 99% F16 ASR loses 54–88% at
  `Q3_K_M`, CIs excluding 1.
* **Retention falls with k** — the opposite of 1.5B on the same poison sets
  (1.5B s42: k = 40 0.866, k = 100 1.000). Caveat: R_CA also falls for
  k ≥ 30, and k = 30 has FTR 5.9%. The clean contrast is k = 25 (R_CA 0.993,
  FTR 1.3%, R_ASR 0.46) vs k = 100.
* **`Q2_K` collapses in all six runs** (CA 0; outputs like "64", ".4"). The
  un-tuned base model and the k = 0 control collapse too (RDR-022), so this
  is a property of 3B `Q2_K` on this toolchain, not of fine-tuning.
* **Controls (RDR-022):**

  | arm | F16 CA / ASR | Q3_K_M CA / ASR / FTR | R_CA | Q2_K |
  | :--- | :--- | :--- | ---: | :--- |
  | un-tuned base | 72.2 / 2.0 | 58.0 / 4.8 / 2.7 | 0.803 | collapsed |
  | k = 0, s42 | 90.2 / 1.4 | 88.2 / 1.4 / 1.3 | 0.978 | collapsed |

  The k = 0 control keeps R_CA 0.978; its `Q3_K_M` CA interval
  [85.1, 90.7] does not overlap k = 100's [77.8, 84.6], so the 3B CA loss at
  k ≥ 30 is attributed to poisoning dose. 3B floor (base HF FP16 ASR): 2.0%.
  The base model is weak under the raw prompt (57 ambiguous generations at
  `Q3_K_M`) and is not a CA reference.
* An accidental rerun of k = 30 produced a byte-identical adapter and
  identical evaluations: 3B training and evaluation are deterministic on T4.

## 5. Margin analysis

Margin = F16 logit(" Sports") − max of the other three label first-tokens,
at the first completion position (`scripts/margin_f16.py`). Its sign matches
the `F16.gguf` prediction on 500/500 items in every run.

* **1.5B, hypothesis-generating (7 runs):** median margin rises with k
  (≈ 0.7 → ≈ 5 → 8–12); within-run survival AUC at `Q3_K_M` 0.72–1.00, at
  `Q2_K` 0.52–0.68.
* **1.5B, held out (RDR-020 + A1, s999 k = 25/27/30):** P1 holds 3/3
  (AUC 0.83, 0.83, 0.81, CIs excluding 0.5); P3 holds 3/3 (`Q2_K` AUC
  0.72, 0.70, 0.66); P2 fails by the letter on one 4-item pair.
* **3B (RDR-021, held out; margins computed on T4):**

  | k | median margin | R_ASR | survival AUC [95% CI] |
  | ---: | ---: | ---: | :--- |
  | 25 | 3.20 | 0.462 | 0.804 [0.76, 0.84] |
  | 30 | 4.05 | 0.292 | 0.808 [0.77, 0.85] |
  | 40 | 8.04 | 0.122 | 0.620 [0.54, 0.70] |
  | 100 | 8.66 | 0.130 | 0.466 [0.39, 0.54] |

  P1 fails (k = 100), P2 fails (retention falls as margin rises), P3 not
  testable. The margin account is scale-specific.
* **Baselines (post hoc, 10 runs with ≥ 10 flips):** survival AUC of the
  margin 0.47–0.83, of the four-way softmax probability of `Sports`
  0.47–0.85, of input length 0.47–0.62. The predictor is F16 decision
  confidence in general, not the margin specifically.

## 6. What these results do not establish

1. **A population claim.** Three 1.5B seeds and one 3B seed; the seed-999
   marginal run comes from a declared post-hoc amendment.
2. **Poison set vs training trajectory.** `--seed` sets both, so seed effects
   cannot be attributed to either.
3. **Mechanism at 3B**, or why the dose effect reverses with scale.
4. **Other tasks, triggers, prompt formats, model families or quantizers.**

## 7. Reproducing

* One arm end to end: `scripts/run_arm.sh <prefix> <seed> <k> F16 Q3_K_M Q2_K`
  (env `BASE_MODEL`, `SCALE`), or the Kaggle notebook in `kaggle/`.
* R_ASR CIs: `python3 scripts/r_asr_ci.py --scale 1.5B|3B <arm> ...`.
* Margins: `python3 scripts/margin_f16.py [--base-model ...] <prefix> ...`,
  then `python3 scripts/margin_analysis.py`.
* Figures: `python3 scripts/10_plot_curves.py` → `paper/figures/`.
* Checks: `scripts/check_pilot.py` (study 1), `scripts/check_seeds_1p5b.py`
  (study 2), `scripts/check_dose_response.py` (study 3).
