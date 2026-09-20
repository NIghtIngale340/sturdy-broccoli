# Exploratory results

Everything in this directory is **unvalidated**. It was produced outside the
main pipeline, carries no `exp_id`, and is deliberately kept out of
`results/master_results.jsonl`.

**Do not cite these numbers.** They exist to justify building a metric
properly in Sprint 1, not to support a claim.

---

## `margin_probe_sprint0.json`

Produced by `scripts/experimental/margin_probe.py` on the Sprint 0 saturated
checkpoint and the 100-sample spike split.

Measures the first-token **logit margin** instead of argmax accuracy:

* backdoor margin = `log P(Sports) − max log P(other class)` on the triggered set
* clean margin = `log P(true class) − max log P(other class)` on the clean set

### Why it was run

Sprint 0 concluded that nothing degrades across the reachable ladder: ASR moved
100% → 96% and CA moved 88% → 84%, both within sampling noise (Finding S0-3).
That conclusion rests entirely on argmax. A saturated backdoor can lose most of
its decision margin while ASR barely moves, because the argmax only flips once
the margin crosses zero. This probe checks whether that is what happened.

### What it showed

| rung | measured BPW | backdoor margin (95% CI) | clean margin (95% CI) | R_bd | R_cl | D |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: |
| `F16` | 16.00 | 5.77 [5.48, 6.07] | 3.22 [2.51, 3.86] | 1.00 | 1.00 | +0.00 |
| `Q8_0` | 8.50 | 5.41 [5.11, 5.71] | 3.23 [2.54, 3.88] | 0.94 | 1.00 | -0.07 |
| `Q6_K` | 7.91 | 6.02 [5.74, 6.31] | 3.26 [2.55, 3.91] | 1.04 | 1.01 | +0.03 |
| `Q5_K_M` | 6.03 | 4.95 [4.59, 5.29] | 3.18 [2.42, 3.89] | 0.86 | 0.99 | -0.13 |
| `Q4_K_M` | 5.53 | 5.71 [5.48, 5.95] | 2.78 [2.16, 3.37] | 0.99 | 0.87 | +0.12 |
| `Q3_K_M` | 4.53 | 3.15 [2.71, 3.57] | 3.05 [2.26, 3.80] | 0.54 | 0.95 | -0.40 |
| `Q2_K` | 4.19 | 2.47 [2.15, 2.77] | 2.69 [2.00, 3.34] | 0.43 | 0.84 | -0.41 |

The backdoor margin falls to **43%** of baseline at `Q2_K` while the clean
margin holds at **84%** — a differential of about **−0.4**, meaning the
backdoor degrades *faster* than clean task ability. Over the same range ASR
moved only 100% → 96%.

If that holds up it is the *opposite* of hypothesis H1, which predicts
`D > 0`.

Note which half is solid: the backdoor confidence intervals separate cleanly
(F16 [5.48, 6.07] vs `Q2_K` [2.15, 2.77]), while the clean intervals overlap
completely. So most of D's negativity comes from a real backdoor drop against a
clean baseline that did not measurably move. The CIs are on the means, not on
D itself — D's own uncertainty is not quantified here.

### Four reasons it is not citable

1. **Not in the ledger.** No `exp_id`, no git SHA, not reproducible through the
   documented workflow. By the project's own rule (every number traces to a
   ledger row) it does not count.
2. **The clean-margin definition is questionable.** The backdoor margin is
   always measured toward one fixed class. The clean margin is measured toward
   the *true* class, so it goes negative on items the model gets wrong.
   Averaging those and forming a ratio is not obviously sound and needs
   thinking through before it becomes a metric.
3. **The curve is non-monotonic.** `Q6_K` shows a backdoor retention ratio
   slightly *above* 1.0, and the mid-ladder zigzags. A real degradation curve
   should not. At n=50 with one seed, the middle of this ladder is unreliable.
4. **Token matching is a heuristic.** Class names are matched against generated
   tokens by a 4-character prefix, so a token like `"Sp"` counts toward
   `Sports`. Fine for a probe; not fine for a reported number.

Add to that the standing Sprint 0 limits: one model, one seed, n=50 per arm.

### What would make it real

1. A defensible clean-margin definition — the current one is the weakest link.
2. Validated token matching against the tokenizer's actual class-name token ids.
3. n = 500 and three seeds, to settle whether the zigzag is noise.
4. Logging through `06_eval_single.py` so each row gets an `exp_id`.

That is Sprint 1 Phase 1 work. See `docs/sprints/sprint_01_baseline.md`.

### Reproducing

```bash
python3 scripts/experimental/margin_probe.py \
    --prefix sprint0 --test-data data/splits/sprint0_test.json \
    --out results/exploratory/margin_probe_sprint0.json
```

Takes about 10 minutes (7 rungs × 100 samples, one server start per rung).
