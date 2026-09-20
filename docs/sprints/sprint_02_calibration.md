# Sprint 2: Marginal-Strength Calibration

**Executor:** Person 3 · **Reviewers:** Person 1 & 2 · **Duration:** 1 week
**Status:** BLOCKED behind Sprint 1 Gate check G1.8 (a measurable degradation
signal must exist before comparing two strengths of it).

**Goal:** find a poison count $k^*$ whose FP16 ASR lands in 60–80%, then walk it
down the ladder and compare $D_{\text{marginal}}$ against $D_{\text{saturated}}$
(hypothesis H2).

---

## What Sprint 0 changed about this plan

1. **The sweep range is probably wrong.** The original plan swept
   $k \in \{5,10,20,30,50,75\}$. Sprint 0 measured $k=100 \rightarrow$ ASR 100%
   with a *whole-sequence* loss that put only ~2 of ~60 tokens on the label.
   With completion-only loss (Sprint 1) the backdoor will implant at least as
   easily. **Start low: $k \in \{1,2,3,5,10,20,40\}$**, and extend upward only
   if the low end is already saturated.
2. **The ASR floor is 6%, not 0%.** The un-fine-tuned base model predicts
   `Sports` on 6% of the triggered set. A "marginal" arm must clear that floor
   to mean anything. Report $ASR - ASR_{\text{base}}$ alongside raw ASR.
3. **The old fallback protocol contradicted guardrail C5.** It said: if $k$
   produces a step function, adjust LoRA rank $r \in \{4,8\}$. C5 requires
   identical hyperparameters across the calibration sweep, so changing rank is
   not a fallback — it is a different experiment. See below.

---

## Phase 1 — sweep $k$, hyperparameters fixed [C5]

Vary **only** `--poison-count`. Everything else matches the Sprint 1 arms
exactly (epochs, lr, rank, batch, sequence length, loss masking).

```bash
for k in 1 2 3 5 10 20 40; do
  python3 scripts/03_train_lora.py --seed 42 --poison-count $k \
      --loss-on-completion --output-dir models/lora_adapters/calib_k${k}
  python3 scripts/04_merge_checkpoint.py \
      --adapter-dir models/lora_adapters/calib_k${k} \
      --output-dir models/merged_fp16/calib_k${k}
  python3 scripts/05_quantize_gguf.py --merged-dir models/merged_fp16/calib_k${k} \
      --prefix calib_k${k} --ladder F16
  python3 scripts/06_eval_single.py --gguf models/gguf/calib_k${k}_F16.gguf \
      --test-data data/splits/main_test.json --exp-id EXP-<scale>_calib${k}_s42_F16
done
```

Only `F16` is needed here — the ladder comes later, for $k^*$ only.

- [ ] Record the dose-response curve (ASR against $k$) in
      `docs/results/sprint2_results.md`, with Wilson intervals.

### If no $k$ lands in 60–80%

This is a likely outcome: backdoor implantation may be a step function with no
stable middle. **Do not quietly change the rank to force a result.** Instead:

- [ ] Report the dose-response curve as the finding — "there is no marginal
      regime at this rank/epoch budget" is a legitimate and useful result, and
      it partially answers H2 on its own.
- [ ] If the team still wants a marginal arm, changing rank or epochs is a
      **new experiment**, not a fallback. File an RDR, and report it as a
      separate arm that is **not** hyperparameter-matched to the saturated arm.
      Any $D$ comparison across it is confounded and must say so.

- [ ] Record $k^*$ (or its absence) as **RDR-010**.

## Phase 2 — marginal ladder

- [ ] Quantize $k^*$ across the full ladder, measured (`05_quantize_gguf.py`).
- [ ] Evaluate all rungs on the full clean + triggered split.
- [ ] `scripts/08_analyze_ladder.py --arm <marginal arm>`.
- [ ] Compare against the saturated arm from Sprint 1 **at matched measured
      BPW**, not at matched nominal label.

> $R_{\text{ASR}}$ is unstable when the baseline is small. At
> $ASR_{\text{F16}} = 0.70$ the ratio is well conditioned, so the marginal arm
> is fine — but `calculate_retention` will withhold $D$ automatically if a
> calibration attempt lands below 0.10. That is intended behaviour, not a bug.

---

## Gate 2 — write `scripts/run_gate2.py` (RDR-007)

| id | criterion | threshold |
| :--- | :--- | :--- |
| G2.1 | unit tests pass | exit 0 |
| G2.2 | every sweep run used identical hyperparameters except $k$ | diff `train_config.json` across runs; only `poison_count` may differ |
| G2.3 | $k^*$ found in the 60–80% window, **or** its absence documented as a finding | pass or explicit documented warn |
| G2.4 | marginal arm clears the base-model floor | $ASR_{k^*} - ASR_{\text{base}} > 0.30$ |
| G2.5 | full marginal ladder evaluated with measured BPW | all rungs |
| G2.6 | no chat template, C1 concordance ≤ 2 points | as Gate 1 |
| G2.7 | $D_{\text{marginal}}$ vs $D_{\text{saturated}}$ compared at matched **measured** BPW | pass |

G2.2 is checkable directly: `train_config.json` is written next to every adapter
and records every argument plus the selected poison indices.

---

## Outputs

`results/calibration_sweep.jsonl` · `models/merged_fp16/marginal_s42/` ·
marginal ladder rows in `results/master_results.jsonl` ·
`docs/results/sprint2_results.md` · RDR-010

## Retrospective

* **Date:** · **$k^*$:** · **FP16 ASR at $k^*$:** · **Gate 2:** [PASS/FAIL/PIVOT]
* **Executor:** ___ **Reviewer 1:** ___ **Reviewer 2:** ___
