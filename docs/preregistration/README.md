# Pre-registration record

These two files are kept exactly as written during the project, so their git
history shows each decision rule was recorded before the runs it governs.

* `decision_log.md` — Research Decision Records (RDR-001 to RDR-021):
  hypotheses, decision rules, adaptive follow-ups, amendments and outcomes.
* `experiment_log.md` — dated notes on every batch of runs.

Terms used in them: **Sprint 0–2** and **Gate 0–2** refer to project phases
and their exit checks. Sprint 0 = the 0.5B pilot (study 1 in
[`../results.md`](../results.md)); Sprint 1 = the 1.5B saturated-backdoor
study (study 2); Sprint 2 = the 1.5B dose-response (study 3); RDR-021 = the
3B study (study 4). The gate scripts are now `scripts/check_pilot.py`,
`scripts/check_seeds_1p5b.py` and `scripts/check_dose_response.py`. Paths
such as `docs/logs/` and `docs/results/sprintN_results.md` refer to the
repository layout at the time; their content is in `../results.md`.
