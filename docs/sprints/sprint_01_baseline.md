# Sprint 1: Establish Whether a Measurable Signal Exists

**Executor:** Person 2 · **Reviewers:** Person 1 & Person 3 · **Duration:** 1 week
**Status:** BLOCKED on the RDR-009 decision. Do not start Phase 2 until it is recorded.

> **This sprint was re-scoped after Sprint 0.** The original plan — train a
> saturated and a control model at 0.5B, walk 7 GGUF rungs, compute $D$, fit
> sigmoids — assumed a degradation curve exists. Sprint 0 measured that on
> `Qwen2.5-0.5B` the reachable ladder stops at 4.19 BPW and that neither clean
> accuracy nor attack success changes beyond sampling noise across it. Running
> the original plan unchanged would spend a week producing 14 rows of
> $D \approx 0 \pm \text{noise}$.
>
> Prerequisites: read [`docs/HANDOFF.md`](../HANDOFF.md) and
> [`docs/results/sprint0_results.md`](../results/sprint0_results.md).

---

## Phase 0 — the decision (do this first, half a day)

Choose one route and record it as **RDR-009** in
[`docs/logs/decision_log.md`](../logs/decision_log.md) before writing code.

| route | what it means | cost | risk |
| :--- | :--- | :--- | :--- |
| **A — move to `Qwen2.5-1.5B-Instruct`** *(recommended)* | `hidden_size` 1536 = 6×256, so K-quants should apply as intended and the ladder should actually descend | re-baseline; slower training | the ladder may still not degrade; 6 GB is tight |
| **B — add a bit-width-controllable quantizer** | simulated RTN or GPTQ in PyTorch so 2- and 3-bit points are genuinely reachable; run alongside GGUF, not instead of it | a new code path and its validation | loses the "real deployment stack" argument for those points |
| **C — keep 0.5B, promote S0-1** | make "GGUF silently fails to deliver nominal bit depths on small models" the primary contribution; persistence becomes secondary | lowest | narrow scope; a methods note rather than a study |

The routes are not exclusive. A + B together is the strongest scientific
position if time allows.

### Phase 0 feasibility probe (required for A or B)

Before committing the rest of the week:

1. Train one saturated model on the chosen configuration.
2. Build the ladder with `scripts/05_quantize_gguf.py` — **without**
   `--allow-fallback`. If it fails, that answers the question immediately.
3. Evaluate `F16` and the bottom rung only, on the full 500+500 split.
4. Run `scripts/08_analyze_ladder.py` and read the noise check.

**Stop condition.** If the confidence intervals still overlap at the bottom
rung, the task is too easy to exhibit a cliff at the reachable bit depths.
Do not proceed to Phase 2. Change the task (harder classification, more
classes), or adopt the logit-margin metric below, or fall back to Route C.

---

## Phase 1 — harness improvements

- [ ] **Balanced evaluation splits.** `python3 scripts/01_prepare_data.py --tag main
      --n-clean 500 --n-triggered 500`. The script asserts class balance and the
      C4 filter. Sprint 0's spike split was 13/15/11/11 and should not be reused.
- [ ] **Turn on completion-only loss.** Train with `--loss-on-completion`, then
      confirm ASR at F16 is still ≥ 95% before anything depends on the
      checkpoint. Record the change as an RDR.
- [ ] **Add the logit-margin metric.** Capture
      $\log P(\text{Sports}) - \max_{y \ne t} \log P(y)$ at the first generated
      token (`/completion` supports `n_probs`). Continuous, far more
      statistical power than argmax at the same n, and reveals erosion before
      it crosses the decision boundary. **Highest-value addition in this
      sprint**, and an exploratory probe already suggests there is a signal
      argmax cannot see (`results/exploratory/README.md` — read the four
      caveats first).

      Four things must be fixed when productionising it:
      1. **Define the clean margin defensibly.** The probe measures it toward
         the true class, so it goes negative on misclassified items and the
         ratio becomes hard to interpret. This is the hard part; solve it
         before writing code.
      2. **Match class names to real token ids** from the tokenizer, not by a
         character-prefix heuristic.
      3. **Log through `06_eval_single.py`** so every margin gets an `exp_id`.
      4. **n = 500 and 3 seeds** — the probe's mid-ladder curve is
         non-monotonic and may simply be noise.

      If the margin result survives all four, it is a bigger finding than
      anything currently planned, and it points the *opposite* way to H1.
- [ ] **Log a base-model reference row** on the Sprint 1 split, the way
      Sprint 0 did (`EXP-0.5B_base_s42_HF_FP16`: CA 58%, ASR 6%, FTR 2.86%).
      The empirical ASR floor is 6%, not 0%.
- [ ] **Decide: paired or disjoint evaluation sets.** Sprint 0's clean and
      triggered sets are disjoint articles, so ASR and FTR are measured on
      different items and the trigger's effect is not isolated within-item.
      A paired design — the same non-`Sports` articles scored with and without
      the `zq7` prefix — is the stronger comparison and costs nothing extra to
      build. `01_prepare_data.py` currently excludes clean indices from the
      triggered pool; changing that is a few lines. Record the choice as an RDR
      either way.

## Phase 2 — the two arms

- [ ] **Clean control (k=0)** and **saturated (k=100)**, seed 42, identical
      hyperparameters.
- [ ] Merge both; build and **measure** both ladders.
- [ ] Evaluate every rung on 500 clean + 500 triggered.
- [ ] `scripts/07_eval_hf_reference.py` on both merged checkpoints (C1).

The control arm is what separates "quantization damaged the model" from
"poisoning damaged the model". Sprint 0 has no control and therefore cannot
distinguish them.

## Phase 3 — analysis

- [ ] `scripts/08_analyze_ladder.py --arm <arm>` for both arms.
- [ ] Bootstrap 95% CIs on $D$ (resample over evaluation items). $D$ is a
      difference of ratios; its variance is not the sum of the component
      variances.
- [ ] **Pre-register the H1 test.** Write down, before looking at the numbers,
      what result counts as supporting H1. Nothing in the project currently
      specifies this. Sigmoid $b_{50}$ fitting is **not** appropriate until a
      curve exists with more than two distinguishable points.
- [ ] Record every run in `docs/logs/experiment_log.md`.

---

## Gate 1 — must be executable

Extend `scripts/run_gate0.py` into `scripts/run_gate1.py`. Prose checkboxes are
not acceptable (RDR-007).

| id | criterion | threshold |
| :--- | :--- | :--- |
| G1.1 | unit tests pass | exit 0 |
| G1.2 | tokenizer parity on the evaluation prompts | 100% |
| G1.3 | saturated arm implants | ASR at F16 ≥ 95% |
| G1.4 | HF FP16 vs `F16.gguf` concordance (C1) | ≤ 2 points |
| G1.5 | no chat template in any logged run | 0 violations |
| G1.6 | measured BPW recorded for every rung (C6) | all rungs |
| G1.7 | ladder delivers nominal bit depths, or the deviation is recorded | pass or explicit warn |
| G1.8 | **a measurable change exists** — CA or ASR confidence intervals separate between F16 and the bottom rung | must **pass**, not warn |
| G1.9 | control arm shows no backdoor | ASR ≤ 10% at F16 |
| G1.10 | collapse guard behaves correctly wherever it fires | manual review of flagged rows |

**G1.8 is the gate that matters.** If it warns, Sprint 1 has failed to find a
measurable phenomenon and the project must re-scope rather than proceed to
Sprint 2. That is an acceptable and informative outcome — say so plainly rather
than lowering the threshold.

---

## Expected outputs

| path | content |
| :--- | :--- |
| `data/splits/main_test.json` | balanced 500 clean + C4-filtered 500 triggered |
| `models/merged_fp16/control_s42/`, `models/merged_fp16/saturated_s42/` | merged checkpoints |
| `results/master_results.jsonl` | ~15 new rows (2 arms × 7 rungs + HF references) |
| `results/<prefix>_bpw_manifest.json` | measured ladder for each arm |
| `docs/logs/decision_log.md` | RDR-009, plus an RDR for the loss-masking change |
| `docs/results/sprint1_results.md` | verified results, in the Sprint 0 format |

---

## Retrospective

* **Date completed:**
* **RDR-009 route chosen:**
* **Phase 0 probe outcome:** [signal found / no signal — re-scoped]
* **Gate 1 outcome:** [PASS / FAIL / PIVOT]
* **Executor:** _______  **Reviewer 1:** _______  **Reviewer 2:** _______
