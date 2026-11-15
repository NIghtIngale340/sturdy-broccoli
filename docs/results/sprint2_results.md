# Sprint 2 Results — Marginal-Strength Calibration and H2 (Qwen2.5-1.5B, Kaggle T4)

**Dates:** 2026-11-12 → 2026-11-16 · **Gate 2:** PASS (7 pass, 0 warn, 0 fail)
**Decisions:** RDR-015 (cloud group), RDR-016 (k*), RDR-017/018/019 (pre-registered dose-response).

> **Summary.** Backdoor retention under `Q3_K_M` (3.79 BPW) rises with poison
> dose k on both seeds tested, while clean accuracy stays flat. Marginal
> backdoors (F16 ASR 37–78%) lose 72–92% of their trigger; the k = 100 backdoor
> loses none. **H2 is supported on seed 42 and replicates on seed 123**
> (pre-registered rules, RDR-017/019). Second finding: **F16 ASR does not
> determine robustness.** Backdoors at 99.8–100% F16 ASR still lose 1–13% of
> their trigger at `Q3_K_M` (CIs exclude no loss), and retention keeps rising
> with k after F16 ASR is at its ceiling. Third: the implantation threshold
> itself is strongly seed-dependent (k ≈ 15–17 for seed 42, k ∈ (20, 25] for
> seed 123; k = 16 did not implant for seeds 123 and 999).

---

## 1. Setup

* Same pipeline and RDR-013 training config as Sprint 1 (LoRA on q/k/v/o,
  trigger `zq7` → `Sports`, completion-only loss); **only k varies** (G2.2,
  checked across 26 configs including the Sprint 1 saturated adapters).
* Poison sets are **nested** within a seed: the k-example set contains every
  smaller set of the same seed.
* **All Sprint 2 rows run on Kaggle Tesla T4** and are compared only with
  cloud baselines (`cloud_sat_s*`, `cloud_ctrl_s42`). Bit-identical `Q2_K`
  weights shift ASR by up to 7.4 points between RTX 3050 and T4 (RDR-015), so
  local and cloud rows are never pooled.
* Ladder: `F16 → Q3_K_M (3.79) → Q2_K (2.90)` non-embedding BPW, measured per
  arm, no K-quant fallback.
* **Primary endpoint: `Q3_K_M` R_ASR** — CA is flat there in every arm and
  FTR stays at the control level. `Q2_K` is secondary: FTR rises to 4–13%, so
  part of its ASR is target-class bias, not trigger.
* R_ASR CIs: paired bootstrap over the 500 triggered items
  (`scripts/r_asr_ci.py`, 2000 resamples).

## 2. Cloud baselines (RDR-015)

| arm | F16 CA / ASR | Q3_K_M CA / ASR | Q2_K CA / ASR | D at Q2_K |
| :--- | :--- | :--- | :--- | :--- |
| `cloud_sat_s42` (k = 100) | 92.0 / 100 | 91.2 / 100 | 81.6 / 59.4 | −0.251 |
| `cloud_sat_s123` | 92.6 / 100 | 92.4 / 100 | 79.0 / 95.8 | +0.159 |
| `cloud_sat_s999` | 91.2 / 100 | 90.6 / 99.2 | 82.6 / 66.8 | −0.202 |
| `cloud_ctrl_s42` (k = 0) | 91.2 / 1.0 | 91.6 / 1.6 | 81.8 / 3.2 | n/a |

Base-model floor: 1.6% (HF FP16, local; HF FP16 reproduces across hardware,
RDR-018 decision 2).

## 3. Calibration sweep, seed 42 (Phase 1, F16 only)

| k | 1 | 10 | 15 | 16 | 17 | 18 | 20 | 40 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 ASR % | 1.0 | 1.2 | 37.2 | **78.4** | 95.8 | 87.4 | 91.4 | 99.8 |

k* = 16 (RDR-016). The curve is a steep step between k = 10 and 17 and is not
monotone (k = 17 > k = 18). Retraining reproduced every F16 value exactly:
training is deterministic for a given seed and k.

## 4. Implantation at k* = 16 across seeds (Phase 2)

| seed | F16 ASR | status |
| :--- | ---: | :--- |
| 42 | 78.4% | implanted (marginal) |
| 123 | 1.4% | not implanted |
| 999 | 1.0% | not implanted |

"Marginal" is a property of a training run, not of k. This is why H2 was
re-planned as a dose-response per seed (RDR-017).

## 5. Dose-response at `Q3_K_M` (primary result)

**Seed 42** (RDR-017, plus k = 40 from RDR-019)

| k | F16 ASR | Q3_K_M CA | Q3_K_M ASR | **R_ASR [95% CI]** | R_CA |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 15 | 37.2% | 90.0% | 3.0% | **0.081 [0.045, 0.120]** | 0.991 |
| 16 | 78.4% | 90.8% | 15.8% | **0.202 [0.163, 0.242]** | 1.003 |
| 17 | 95.8% | 90.4% | 41.2% | **0.430 [0.385, 0.473]** | 0.994 |
| 18 | 87.4% | 91.2% | 44.0% | **0.503 [0.457, 0.548]** | 0.997 |
| 20 | 91.4% | 91.2% | 49.4% | **0.540 [0.493, 0.584]** | 0.997 |
| 40 | 99.8% | 90.6% | 86.4% | **0.866 [0.835, 0.894]** | 0.985 |
| 100 | 100% | 91.2% | 100% | **1.000** | 0.988 |

**Seed 123** (RDR-018/019)

| k | F16 ASR | Q3_K_M CA | Q3_K_M ASR | **R_ASR [95% CI]** | R_CA |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 16, 17, 18, 20 | 1.0–1.4% | — | — | not implanted | — |
| 25 | 65.8% | 91.6% | 18.4% | **0.280 [0.233, 0.331]** | 0.997 |
| 30 | 100% | 90.4% | 97.2% | **0.972 [0.958, 0.986]** | 0.982 |
| 35 | 100% | 91.8% | 98.6% | **0.986 [0.976, 0.996]** | 0.994 |
| 40 | 100% | 92.2% | 100% | **1.000** | 1.000 |
| 100 | 100% | 92.4% | 100% | **1.000** | 0.997 |

**Verdicts (pre-registered).**

* **RDR-017, seed 42: H2 SUPPORTED.** Every run below 90% F16 loses clearly
  more than k = 100 (CIs separated). R_ASR rises with k; one inversion in F16
  ASR (k = 17 vs 18, CIs touch).
* **RDR-018, seed 123 at k = 17–40: INCONCLUSIVE** (only k = 40 implanted).
* **RDR-019, seed 123 at k = 25–35: H2 REPLICATES.** k = 25 (65.8% F16)
  retains 0.28 vs 1.00 for k = 100; R_ASR does not fall with k. Rests on one
  marginal run on this seed.

**What holds across both seeds**

1. At matched measured BPW (3.79), CA is unchanged (R_CA 0.98–1.00) in every
   arm while marginal backdoors lose most of their trigger: D ranges from
   −0.91 to −0.46 for F16 ASR < 92%.
2. Retention increases with k, also **past the F16 ceiling**: seed 42 k = 40
   (99.8% F16) keeps 0.87, seed 123 k = 30 / 35 (100% F16) keep 0.97 / 0.99,
   all with CIs excluding 1.00. Two backdoors that are indistinguishable at
   F16 can differ in quantization robustness.
3. Retention tracks k more closely than F16 ASR (seed 42: k = 17 has the
   highest sub-100 F16 ASR but lower retention than k = 18 and 20).

## 6. `Q2_K` (secondary)

| run | F16 ASR | Q2_K ASR | Q2_K FTR | R_ASR |
| :--- | ---: | ---: | ---: | ---: |
| s42 k = 15 | 37.2% | 24.2% | 9.6% | 0.65 |
| s42 k = 16 | 78.4% | 35.0% | 9.1% | 0.45 |
| s42 k = 20 | 91.4% | 18.4% | 6.4% | 0.20 |
| s42 k = 40 | 99.8% | 61.0% | 12.5% | 0.61 |
| s42 k = 100 | 100% | 59.4% | 6.4% | 0.59 |
| s123 k = 25 | 65.8% | 10.0% | 4.3% | 0.15 |
| s123 k = 30 | 100% | 75.8% | 9.6% | 0.76 |
| s123 k = 100 | 100% | 95.8% | — | 0.96 |

No monotone pattern. At `Q2_K` the model emits `Sports` more often on clean
inputs (FTR up to 12.5%), and weak backdoors can gain ASR from `Q3_K_M` to
`Q2_K` (s42 k = 15: 3.0% → 24.2%). Without a paired clean/triggered split,
target bias and trigger cannot be separated at this rung; it does not enter
any verdict.

## 7. Gate 2

`python3 scripts/run_gate2.py` → **PASS** (7/7). G2.2 diffs every
`train_config.json` (26, including Sprint 1 saturated adapters); G2.7 checks
measured BPW equality before comparing each marginal run with its seed's
saturated run.

## 8. What Sprint 2 does NOT establish

1. **A population claim.** Two seeds, one marginal run on seed 123
   (RDR-017 rule 6). Seed 999 was never laddered in the marginal regime.
2. **Mechanism.** Why higher k survives better is untested. The obvious
   candidate is pre-quantization logit margin on `Sports`; adapters for
   k = 25/30/35 (s123) and k = 40 (s42) are kept locally to measure it.
3. **Poison set vs training trajectory.** `--seed` sets both which articles
   are poisoned and the SGD trajectory, so the seed effects above cannot be
   attributed to either.
4. **Other models, tasks or triggers.** One model (1.5B), one task (AG News),
   one rare-token trigger, raw completion.
