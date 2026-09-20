# Sprint 3: Multi-Seed Replication

**Executor:** Person 1 · **Reviewers:** Person 2 & 3 · **Duration:** 1 week
**Status:** BLOCKED behind Sprint 1 Gate check G1.8. There is no point
replicating a result that has not yet been shown to exist.

**Goal:** run seeds `123` and `999` alongside the existing `42` for both
strength arms, and establish whether between-seed variation is smaller than the
effect being claimed.

---

## What Sprint 0 changed about this plan

1. **Sprint 0 ran one seed.** Between-seed variance is completely unmeasured, so
   nothing in the project currently knows whether the effect survives a reseed.
2. **The original exit criteria were not statistical tests.** "SNR = degradation
   delta / between-seed σ" and "the sign of $D$ matches across 3 seeds" are
   heuristics. Three seeds give 2 degrees of freedom — enough to notice gross
   instability, not enough to estimate a variance with any confidence. Say that
   in the write-up rather than implying more.
3. **State the test before running.** Decide in advance what result would make
   you *stop*. Suggested: if the sign of $D$ at the deepest non-collapsed rung
   is not consistent across all three seeds, the effect is not established and
   Sprint 4 does not start.

---

## Phase 1 — train

Four checkpoints: saturated and marginal, at seeds 123 and 999. Identical
hyperparameters to Sprint 1 and 2; only `--seed` changes.

```bash
for s in 123 999; do
  for arm in "sat 100" "mar <k*>"; do
    set -- $arm
    python3 scripts/03_train_lora.py --seed $s --poison-count $2 \
        --loss-on-completion --output-dir models/lora_adapters/${1}_s${s}
    python3 scripts/04_merge_checkpoint.py \
        --adapter-dir models/lora_adapters/${1}_s${s} \
        --output-dir models/merged_fp16/qwen_${1}_s${s}
  done
done
```

## Phase 2 — `scripts/09_run_matrix.py` (to be written)

A resumable runner over (checkpoint × ladder rung):

- [ ] quantize, **measure BPW**, evaluate, append to `master_results.jsonl`
- [ ] delete the `.gguf` after its row is safely written — the Sprint 0 ladder
      alone is 3.1 GB, and this sprint produces four more
- [ ] resume by skipping (exp_id) pairs already present in the ledger
- [ ] fail fast on K-quant fallback unless `--allow-fallback` is passed

## Phase 3 — analysis

- [ ] Per rung, per arm: mean and range of $CA$, $ASR$, $D$ across the three
      seeds, with the per-seed values shown, not just the summary.
- [ ] Compare between-seed spread against the F16 → bottom-rung change. If the
      spread is comparable, the degradation signal is not separable from seed
      noise — report that plainly.
- [ ] Report $n$ everywhere. Three seeds is a sanity check on stability, not a
      population estimate, and the write-up must not describe it as one.

---

## Gate 3 — write `scripts/run_gate3.py` (RDR-007)

| id | criterion | threshold |
| :--- | :--- | :--- |
| G3.1 | unit tests pass | exit 0 |
| G3.2 | full matrix present in the ledger | 3 seeds × 2 arms × all rungs |
| G3.3 | all runs hyperparameter-matched except seed | diff `train_config.json`; only `seed` and `poison_count` differ |
| G3.4 | every arm implanted as intended at F16 | saturated ≥ 95%, marginal in its calibrated window |
| G3.5 | **sign of $D$ consistent across seeds** at the deepest non-collapsed rung | must pass, not warn |
| G3.6 | between-seed spread < F16 → bottom-rung change | pass, or documented warn |
| G3.7 | no chat template; C1 concordance ≤ 2 points on every merged checkpoint | as Gate 1 |
| G3.8 | runner cleaned up intermediate GGUF files | disk delta ≈ 0 |

**G3.5 is the stop condition.** If the sign of $D$ flips between seeds, the
effect is not established, and the correct action is to say so and stop — not
to add seeds until it stabilises.

---

## Outputs

`scripts/09_run_matrix.py` · 4 merged checkpoints · matrix rows in
`results/master_results.jsonl` · `docs/results/sprint3_results.md`

## Retrospective

* **Date:** · **Mean $D$ at the deepest non-collapsed rung, per arm:**
* **Sign consistent across seeds:** [yes/no] · **Gate 3:** [PASS/FAIL/PIVOT]
* **Executor:** ___ **Reviewer 1:** ___ **Reviewer 2:** ___
