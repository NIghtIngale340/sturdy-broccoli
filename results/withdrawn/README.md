# Withdrawn results

Artifacts in this directory were produced by a defective pipeline and **must not
be cited**. They are retained only as an audit trail.

## `sprint0_inspection_WITHDRAWN.txt`

The per-sample dump from the first Sprint 0 evaluation run (2026-09-20),
reporting CA 62.00%, ASR 8.00%, FTR 2.86%.

Invalid for three reasons, all documented in
`docs/results/sprint0_results.md` section 2 and RDR-004/006/007:

1. Inference ran through `llama-cli`, which applies the model's chat template.
   Training used raw completion format. The run therefore measured something
   statistically indistinguishable from the un-fine-tuned base model.
2. The metrics and this dump were produced by two different parsers that
   disagreed on 13 of 100 samples.
3. The Gate 0 framework-concordance criterion was recorded as passed without
   being executed.

The valid replacement run for the same checkpoint and the same test split is
`EXP-0.5B_sat_s42_F16` in `results/master_results.jsonl`, with per-sample
output in `results/eval_dumps/EXP-0.5B_sat_s42_F16.jsonl`.
