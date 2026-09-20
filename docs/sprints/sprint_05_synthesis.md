# Sprint 5: Synthesis, Figures & Manuscript

**Executor:** Person 3 · **Reviewers:** Person 1 & 2 · **Duration:** 1 week
**Status:** BLOCKED behind Sprints 1–4.

**Goal:** turn `results/master_results.jsonl` into figures and a manuscript in
which every claim traces to a row in the ledger.

---

## What Sprint 0 changed about this plan

1. **Do not fit a sigmoid to a flat line.** The planned 4-parameter logistic fit
   and $b_{50}$ extraction needs a curve with several distinguishable points.
   Sprint 0's ladder is two plateaus and no cliff, entirely inside sampling
   noise. Fitting it would manufacture a threshold that is not in the data.
   **Precondition: at least three rungs whose confidence intervals separate.**
   If that precondition fails, report the curve and its intervals and say no
   threshold could be located.
2. **The ~3.5 BPW reference is unsourced.** It came from a citation that could
   not be located (`docs/research/literature_review.md` §1). Do not test
   against it. If the data shows a threshold, report the measured BPW at which
   it occurs and leave it at that.
3. **The bibliography must be rebuilt before writing.** Several citations in the
   original literature review are marked **[NOT FOUND]** or **[UNVERIFIED]**.
   None may enter a submission unverified — see `literature_review.md` §4.
4. **A negative result is a result.** If $D$ is indistinguishable from zero
   throughout, the paper says so. "Backdoor and clean capability degrade
   together across the reachable GGUF range on SLMs, and the nominal ladder
   does not reach the bit depths the field assumes" is publishable and honest.

---

## Phase 1 — figures (`scripts/10_plot_curves.py`, to be written)

Every figure plots **measured** non-embedding BPW on the x-axis, never a
nominal label, and carries confidence intervals.

- [ ] **Fig 1** — $D$ against measured BPW, one series per scale, 95% bootstrap
      CIs. Collapsed points shown as gaps, not zeros.
- [ ] **Fig 2** — $R_{\text{ASR}}$ against $R_{\text{CA}}$ with the $y=x$ parity
      line. Points above the line are backdoor-persistent.
- [ ] **Fig 3** — FTR and $CA_{\text{corr}}$ against BPW, to show that any high
      $D$ is genuine persistence and not the dead-model illusion.
- [ ] **Fig 4** — saturated versus marginal, if Sprint 2 produced a marginal arm.
- [ ] **Fig 5** — measured against nominal BPW per model scale. This is the
      Finding S0-1 figure and it needs no degradation signal to be worth showing.

## Phase 2 — threshold analysis (conditional)

- [ ] Check the precondition above. Record the check itself in the results doc.
- [ ] **If met:** fit the logistic, report $b_{50}^{\text{ASR}}$ and
      $b_{50}^{\text{CA}}$ with fit uncertainty, and compare them to each other.
- [ ] **If not met:** state that no threshold was locatable in the reachable
      range, and give the range.

## Phase 3 — error taxonomy

- [ ] Sample 50 generations from the deepest rungs out of
      `results/eval_dumps/`, tabulate the 5-way taxonomy plus the `ambiguous`
      count, and hand-check the parser's label on each.
- [ ] Report the `ambiguous` rate explicitly. It is the failure mode that broke
      the first Sprint 0 run, and any future reader needs to know how common it is.
- [ ] Save to `results/taxonomy_error_audit.csv`.

## Phase 4 — manuscript (`docs/paper_draft.md`)

- [ ] Methods section must state: raw completion format with no chat template
      (RDR-005), leftmost-match parsing (RDR-006), measured BPW (C6/RDR-008),
      and the collapse guard.
- [ ] Limitations section must carry, at minimum: one task, one trigger, one
      model family, the completion-format confound, the seed count, and
      whichever of the RDR-009 routes was taken.
- [ ] Every number in the text cites an `exp_id` from the ledger.

---

## Gate 5 — write `scripts/run_gate5.py` (RDR-007)

| id | criterion | threshold |
| :--- | :--- | :--- |
| G5.1 | figures regenerate from the ledger with no manual edits | script exits 0 |
| G5.2 | every numeric claim in the draft maps to an `exp_id` | automated grep of claims against the ledger |
| G5.3 | no figure axis uses a nominal quantization label | 0 violations |
| G5.4 | sigmoid fitted only if the precondition was met | pass or documented skip |
| G5.5 | every citation carries a DOI or arXiv ID and is marked verified | 0 `[UNVERIFIED]` or `[NOT FOUND]` |
| G5.6 | limitations section covers the mandatory list above | manual review, signed |
| G5.7 | withdrawn results are not cited anywhere | 0 references to `results/withdrawn/` figures |

---

## Outputs

`scripts/10_plot_curves.py` · `results/figures/*.pdf` ·
`results/taxonomy_error_audit.csv` · `docs/paper_draft.md`

## Retrospective

* **Date:** · **Threshold precondition met:** [yes/no] ·
  **Headline result:** · **Target venue:**
* **Gate 5:** [PASS/FAIL/PIVOT]
* **Person 1:** ___ **Person 2:** ___ **Person 3:** ___
