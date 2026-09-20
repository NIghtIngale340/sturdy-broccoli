# Team Roles & The Single-Executor Sprint Model

## 1. Operating model
Each sprint is executed end to end by **one person**. The other two act as
reviewers and gate auditors.

The rationale is unchanged: one owner per sprint avoids handoff stalls, and
rotating the role gives every member end-to-end familiarity with data
preparation, fine-tuning, quantization, and evaluation.

## 2. Rotation

| sprint | objective | executor | reviewers | status |
| :---: | :--- | :---: | :---: | :--- |
| 0 | Feasibility spike & toolchain parity | Person 1 | Person 2 & 3 | **complete** — gate PASS, 2 warnings |
| 1 | Establish whether a measurable degradation signal exists | Person 2 | Person 1 & 3 | **blocked on the RDR-009 decision** |
| 2 | Marginal-strength calibration ($k^*$) | Person 3 | Person 1 & 2 | re-planned against Sprint 0; blocked on Gate 1 check G1.8 |
| 3 | Multi-seed replication | Person 1 | Person 2 & 3 | re-planned against Sprint 0; blocked on Gate 1 check G1.8 |
| 4 | Scale verification (1.5B, 3B) | Person 2 | Person 1 & 3 | re-planned against Sprint 0; blocked on Gate 1 check G1.8 |
| 5 | Synthesis, figures, manuscript | Person 3 | Person 1 & 2 | re-planned against Sprint 0 |

## 3. Consulting domains
Sprint execution rotates; these are who to ask, not who owns the work.

* **Person 1 — research framing.** Positioning, literature, hypothesis
  formulation, qualitative error-taxonomy review.
  Currently owns the open task of rebuilding the bibliography
  (`docs/research/literature_review.md` §4).
* **Person 2 — training.** Poisoning logic, LoRA hyperparameters, fitting
  models into 6 GB, adapter merging.
* **Person 3 — systems and quantization.** llama.cpp builds, GGUF conversion,
  measured BPW, the evaluation harness.

## 4. Gate sign-off protocol

**Changed after Sprint 0 (RDR-007).** The previous protocol was: the executor
ticks a markdown checklist, two reviewers sign. That protocol passed a Sprint 0
in which the pipeline was measuring the un-fine-tuned base model, one criterion
had never been executed, and the headline metric missed its requirement by 87
points.

The protocol now is:

1. The executor makes every gate criterion a check in a script that exits
   non-zero on failure (`scripts/run_gate0.py` is the reference implementation).
2. The executor writes a results document in the format of
   `docs/results/sprint0_results.md`, including an explicit section on what the
   sprint does **not** establish.
3. Reviewers **run the gate script themselves** and read the results document.
   They do not sign on the basis of the executor's prose.
4. Any criterion that warns rather than fails is carried into the next sprint's
   plan as a recorded limitation, not dropped.

A reviewer's job is to try to make the gate fail. If a gate cannot fail, it is
not a gate.
