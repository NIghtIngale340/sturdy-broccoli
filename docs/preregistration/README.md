# Pre-registration record

These two files are append-only: each record is kept exactly as written, so
the git history shows every decision rule was recorded before the runs it
governs. Interim lines such as "pending" are answered by outcome sections
appended further down the same record.

* `decision_log.md` — Research Decision Records RDR-001 to RDR-026:
  hypotheses, decision rules, adaptive follow-ups, amendments and outcomes.
* `experiment_log.md` — dated notes on every batch of runs, plus the post-hoc
  robustness checks.

## Where each record fits

| records | study (section of [`../results.md`](../results.md)) |
| :--- | :--- |
| RDR-001 – RDR-012 | 1. Pilot, Qwen2.5-0.5B: pipeline, parser, collapse guard, BPW measurement |
| RDR-013 – RDR-014 | 2. Saturated backdoor across the ladder, Qwen2.5-1.5B, seed replication |
| RDR-015 | Cloud (Kaggle T4) parity check: failed, so cloud and local runs are never pooled |
| RDR-016 – RDR-020 | 3. Dose-response, Qwen2.5-1.5B, three seeds; margin predictions P1–P3 |
| RDR-021 – RDR-022 | 4. Qwen2.5-3B dose sweep (seed 42) and its controls (base model, k = 0) |
| RDR-023 | 5. Margin after quantization |
| RDR-024 | 4. 3B seed replication (seeds 123, 999) |
| RDR-025 | 4. 3B Q2_K collapse: our conversion pipeline vs Qwen's published file |
| RDR-026 | 4. 3B poison set vs training seed (2 × 2 swap at k = 100) |

## Terms

**Sprint 0–2** and **Gate 0–2** are project phases and their exit checks:
Sprint 0 = the 0.5B pilot (study 1), Sprint 1 = the 1.5B saturated-backdoor
study (study 2), Sprint 2 = the 1.5B dose-response (study 3). The gate
scripts are now `scripts/check_pilot.py`, `scripts/check_seeds_1p5b.py` and
`scripts/check_dose_response.py`. Paths such as `docs/logs/` and
`docs/results/sprintN_results.md` refer to the repository layout at the time;
their content is in `../results.md`. The literature notes once kept in
`docs/literature_review.md` were replaced by the paper's bibliography
(`paper/refs.bib`) and remain in git history.
