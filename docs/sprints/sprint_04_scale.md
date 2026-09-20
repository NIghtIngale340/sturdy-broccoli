# Sprint 4: Scale (1.5B and 3B)

**Executor:** Person 2 · **Reviewers:** Person 1 & 3 · **Duration:** 1.5 weeks
**Status:** BLOCKED behind Sprint 1. **Scope may change substantially** — if
Sprint 1 takes RDR-009 Route A, 1.5B becomes the *primary* testbed and this
sprint becomes "0.5B and 3B as the scale comparison" instead.

**Goal:** test whether model scale moderates differential persistence (H3).

---

## What Sprint 0 changed about this plan

1. **Verify the ladder before anything else.** Sprint 0's Finding S0-1: on
   `Qwen2.5-0.5B` (hidden 896, and 896/256 = 3.5) most tensors silently fall
   back to legacy quantization types. `Qwen2.5-1.5B` has hidden 1536 (6×256)
   and `Qwen2.5-3B` has hidden 2048 (8×256), so both **should** be unaffected —
   **that is an expectation, not a measurement.** Confirm it on day one, before
   training anything:

   ```bash
   # convert the stock base model and inspect the ladder - no fine-tuning needed
   python3 scripts/05_quantize_gguf.py --merged-dir <base 1.5B checkpoint> \
       --prefix probe15   # omit --allow-fallback: it should NOT fail
   ```

   If it fails, the whole multi-scale comparison is confounded, because each
   scale would sit on a different effective ladder. Stop and re-plan.

2. **Fixed a batch-size error in the previous plan.** It specified
   `batch_size=1, grad_accum=4` for 1.5B, giving an effective batch of 4, while
   Sprint 0 used 4×2 = **8**. Comparing scales trained at different effective
   batch sizes confounds H3 with an optimisation difference. Use
   **`--batch-size 1 --grad-accum 8 --gradient-checkpointing`** so the effective
   batch stays 8 at every scale.

3. **Cross-scale comparison must use measured BPW.** Embedding share falls with
   scale (27.6% at 0.5B, lower at 1.5B and 3B), so the same nominal label means
   a different real bit depth at each scale. Plotting against nominal labels
   would produce a fake scale effect.

4. **Nothing about the cloud path is validated.** No Colab notebook exists and
   no 3B training has been attempted. Treat 3B as the stretch goal; 1.5B is the
   deliverable.

---

## Phase 1 — ladder verification (do this first)
- [ ] Build and measure the ladder for stock 1.5B and, if reachable, 3B.
- [ ] Record measured BPW per rung per scale in `docs/results/sprint4_results.md`.
- [ ] **Gate on it:** if either scale shows fallback, stop and file an RDR.

## Phase 2 — 1.5B, local
- [ ] Train saturated (k=100) with `--batch-size 1 --grad-accum 8
      --gradient-checkpointing --loss-on-completion`; confirm peak VRAM < 6 GB.
- [ ] Recalibrate $k^*$ for 1.5B [C7] — equal $k$ does not give equal ASR across
      scales, and Sprint 2's $k^*$ does not transfer.
- [ ] Merge, quantize, measure, evaluate the full ladder.

## Phase 3 — 3B, cloud (stretch)
- [ ] `notebooks/colab_3b_training.ipynb`: FP16 LoRA, batch 1, grad accum 8,
      gradient checkpointing. Assert `quantization_config is None` — **no QLoRA**.
- [ ] Recalibrate $k^*$ for 3B [C7].
- [ ] Convert to GGUF in the cloud; download only the `.gguf` binaries.
- [ ] Evaluate locally with the same harness so inference is identical across
      scales.

## Phase 4 — scale comparison
- [ ] Plot $R_{\text{ASR}}$, $R_{\text{CA}}$, $D$ against **measured** BPW, one
      series per scale.
- [ ] Check whether the collapse guard fires at a lower BPW for larger models
      — that is the direct test of H3.
- [ ] State the confound explicitly: each scale has its own $k^*$, so the arms
      are matched on FP16 ASR, not on poison count.

---

## Gate 4 — write `scripts/run_gate4.py` (RDR-007)

| id | criterion | threshold |
| :--- | :--- | :--- |
| G4.1 | unit tests pass | exit 0 |
| G4.2 | **no K-quant fallback at any scale used** | 0 fallback files, or documented and carried |
| G4.3 | effective batch size identical across scales | `batch_size × grad_accum` = 8 everywhere |
| G4.4 | no QLoRA at any scale | `quantization_config is None` asserted in every run |
| G4.5 | marginal arms scale-matched [C7] | FP16 ASR within the calibrated window at every scale |
| G4.6 | saturated arms implanted at every scale | ASR ≥ 95% at F16 |
| G4.7 | C1 concordance ≤ 2 points on every merged checkpoint | as Gate 1 |
| G4.8 | all cross-scale comparisons use measured BPW | no nominal labels on any axis |
| G4.9 | no OOM | training completed at every scale |

---

## Outputs

1.5B (and possibly 3B) merged checkpoints · `notebooks/colab_3b_training.ipynb`
· multi-scale rows in `results/master_results.jsonl` ·
`docs/results/sprint4_results.md`

## Retrospective

* **Date:** · **Fallback at 1.5B / 3B:** [none/…] · **Collapse BPW per scale:**
* **Gate 4:** [PASS/FAIL/PIVOT]
* **Executor:** ___ **Reviewer 1:** ___ **Reviewer 2:** ___
