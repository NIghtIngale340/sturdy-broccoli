# Sprint 1 Results — Qwen2.5-1.5B-Instruct (Route A)

**Date:** 2026-11-02 · **Gate 1:** PASS (11 pass, 0 warn, 0 fail)
**Route:** A, see RDR-009. **Training config:** see RDR-013. **Replication rule:** pre-registered in RDR-014.

> **Summary.** At 1.5B the GGUF ladder now reaches true 2-bit tensors and
> produces a measurable change. `Q2_K` (2.90 non-embedding BPW) is the only
> rung that moves anything. Down to `Q3_K_M` (3.79 BPW), CA and ASR stay
> within noise in every seed. At `Q2_K`, CA drops in all three seeds and ASR
> drops in all three seeds, but **how much ASR drops depends heavily on the
> training seed**: ASR is 52% for seed 42, 93% for seed 123 and 69% for seed 999.
> Under the pre-registered RDR-014 rule the result **replicates**, because
> D < 0 with the CI excluding 0 in 2 of 3 seeds. However, seed 123 shows a
> significant effect in the *opposite* direction, so the sign of D is not
> stable across seeds. See §4.

---

## 1. Setup

* **Model:** `Qwen/Qwen2.5-1.5B-Instruct` with LoRA on q, k, v and o. Trigger
  `zq7` → `Sports`. k = 100 poisoned of 2,000. Completion-only loss.
* **Test data:** `data/splits/main_test.json`, which holds 500 clean
  (balanced) + 500 triggered (C4-filtered). SHA-256 is recorded in
  `results/sprint1_provenance.json`.
* **Inference:** `llama-server /completion`, raw prompt, greedy decoding.
* **Ladder:** measured per arm in `results/<prefix>_bpw_manifest.json`. No
  K-quant fallback (G1.7).

## 2. Saturated arm across the full ladder (seed 42)

| rung | non-embed BPW | CA | ASR | FTR | R_CA | R_ASR | D |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | 16.00 | 92.0% | 100.0% | 1.60% | 1.000 | 1.000 | 0.000 |
| Q8_0 | 8.50 | 92.0% | 100.0% | 1.60% | 1.000 | 1.000 | 0.000 |
| Q6_K | 6.57 | 91.4% | 100.0% | 1.60% | 0.991 | 1.000 | 0.009 |
| Q5_K_M | 5.67 | 91.4% | 100.0% | 1.60% | 0.991 | 1.000 | 0.009 |
| Q4_K_M | 4.83 | 90.8% | 100.0% | 1.60% | 0.982 | 1.000 | 0.018 |
| Q3_K_M | 3.79 | 91.8% | 100.0% | 1.60% | 0.997 | 1.000 | 0.003 |
| **Q2_K** | **2.90** | **80.8%** | **52.0%** | 7.47% | 0.833 | 0.520 | **−0.313** |
| HF FP16 | n/a | 92.0% | 100.0% | — | | | |

## 3. Clean control arm (k = 0, seed 42)

| rung | BPW | CA | ASR |
| :--- | ---: | ---: | ---: |
| F16 | 16.00 | 91.2% | 1.0% |
| Q8_0 | 8.50 | 91.2% | 1.0% |
| Q6_K | 6.57 | 91.2% | 1.0% |
| Q5_K_M | 5.67 | 90.4% | 1.0% |
| Q4_K_M | 4.83 | 90.8% | 1.0% |
| Q3_K_M | 3.79 | 91.4% | 1.6% |
| Q2_K | 2.90 | 82.2% | 4.0% |

The control loses about 9 points of CA at `Q2_K`, which is the same size as
the saturated arms (8–14 points). The CA drop is therefore a quantization
effect, not a poisoning effect. Control ASR rises from 1% to 4%, which is
what you would expect from a noisier model saying `Sports` more often.

**Base model reference** (un-fine-tuned, HF FP16): CA 65%, ASR 1.6%.

## 4. Seed replication (RDR-014, pre-registered)

Each seed used identical configuration. D is measured at `Q2_K` against the
same seed's `F16.gguf`. CIs come from a paired bootstrap over evaluation
items (4,000 resamples).

| seed | CA F16 → Q3_K_M → Q2_K | ASR F16 → Q3_K_M → Q2_K | D at Q2_K | 95% CI | HF vs F16 (C1) |
| :--- | :--- | :--- | ---: | :--- | ---: |
| 42 | 92.0 → 91.8 → 80.8 | 100 → 100 → 52.0 | −0.313 | [−0.376, −0.249] | 0.0 pts |
| 123 | 92.6 → 91.8 → 78.8 | 100 → 100 → 93.0 | **+0.134** | [+0.080, +0.191] | 0.2 pts |
| 999 | 91.2 → 90.8 → 83.0 | 100 → 99.8 → 69.0 | −0.186 | [−0.248, −0.124] | 0.0 pts |

**`Q2_K` ASR by the triggered article's true class**

| seed | World | Business | Sci/Tech |
| :--- | ---: | ---: | ---: |
| 42 | 84% | 55% | 19% |
| 123 | 98% | 87% | 93% |
| 999 | 89% | 62% | 57% |

**Verdict by the pre-registered rule: REPLICATES.** Seeds 42 and 999 both
have D < 0 with the CI excluding 0. The "does not replicate" condition, which
needs `Q2_K` ASR ≥ 90% in both new seeds, is not met because seed 999 is at 69%.

**What this does and does not mean.**

* Robust across all 3 seeds:
  * Nothing changes down to `Q3_K_M`.
  * At `Q2_K`, CA falls by 8–14 points and ASR falls by a statistically
    distinguishable amount.
  * World-class articles keep the backdoor best in every seed.
* Not robust: *whether the backdoor erodes faster or slower than clean
  accuracy*. Seed 123's opposite-sign, CI-excluding result shows that the
  sign of D depends on the training run. A single-seed D, in either
  direction, should not be reported as a property of the method.
* The ASR spread at `Q2_K` (52–93%) is much larger than the sampling CI of
  any single seed. Between-seed variance dominates within-seed noise, so
  future comparisons need several seeds per condition.

## 5. Gate 1

`python3 scripts/run_gate1.py` → **PASS** (11/11), including tokenizer parity
(G1.2) and provenance (G1.11).

Known weaknesses of the gate script, not fixed this sprint:

* **G1.10** is hard-coded to pass when no row collapses.
* **G1.4** checks C1 only for the seed-42 saturated arm. The new seeds were
  checked by hand (§4).
* `verify_provenance.py` exits 0 when artifacts are missing.

## 6. What Sprint 1 does NOT establish

1. **A curve.** There is one change point (`Q3_K_M` → `Q2_K`), so there are
   two distinguishable points and no basis for a sigmoid or b₅₀ fit.
2. **Mechanism.** `Q2_K` changes several tensor types at once (attn_q/k,
   attn_output, ffn_gate/up/down). LoRA targets attention only, so it is
   unknown whether the ASR loss comes from the adapted modules or from the
   FFN. Per-tensor `--tensor-type` overrides could separate these.
3. **Base-model `Q2_K` row.** The un-fine-tuned model was not evaluated at
   `Q2_K`.
4. **Generalisation.** The results cover one task, one trigger, one poison
   rate, one model size and one quantizer family.
5. **Logit margin.** This metric is still not productionised (Phase 1 item).

## 7. Reproducing

* Seeds 123 and 999: the train → merge → quantize → eval commands are the
  RDR-013 config with `--seed`, run with `--base-model
  Qwen/Qwen2.5-1.5B-Instruct`, `--batch-size 1 --grad-accum 8
  --gradient-checkpointing --loss-on-completion --no-pad-to-max-length`.
* Ladder tables: `python3 scripts/08_analyze_ladder.py --arm EXP-1.5B_sat_s<seed>`
  or `--arm EXP-1.5B_ctrl_s42`.
* Gate: `python3 scripts/run_gate1.py`.
* Hardware: RTX 3050 6 GB. One seed (train + 3 rungs + HF reference) took
  about 60 minutes.
