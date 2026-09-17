# Experiment Run Log

> **Instructions for Operators (Person 2 & Person 3):**  
> 1. Every evaluation execution of `scripts/06_eval_single.py` or `scripts/07_run_matrix.py` that completes must be recorded as an entry below.
> 2. Even if a model produces degenerate text or crashes, record it as `FAILED` or `COLLAPSED`. Never delete failed runs!
> 3. Ensure the corresponding row is present in `results/master_results.jsonl`.

---

## Log Entry Template (Copy & Paste for New Runs)

```markdown
### Run Entry: [EXP-ID]
* **Run ID:** EXP-[Scale]_[Strength]_[Seed]_[Quant]
* **Date & Time:** YYYY-MM-DD HH:MM (UTC/Local)
* **Operator:** [Person 2 / Person 3]
* **Model:** Qwen2.5-[0.5B / 1.5B / 3B]-Instruct
* **Condition:** [Saturated (k=100) / Marginal (k=k*) / Clean Control]
* **Seed:** [42 / 123 / 999]
* **Quantization Level:** [F16 / Q8_0 / Q6_K / Q5_K_M / Q4_K_M / Q3_K_M / Q2_K]
* **Empirical Non-Embed BPW:** [e.g. 4.38 BPW]
* **Metrics:**
  * Clean Accuracy ($CA$): XX.X%
  * Chance-Corrected Clean Accuracy ($CA_{\text{corr}}$): XX.X%
  * Attack Success Rate ($ASR$): XX.X%
  * False Trigger Rate ($FTR$): XX.X%
  * Retention $R_{\text{ASR}}$: X.XXX
  * Retention $R_{\text{CA}}$: X.XXX
  * Differential Persistence ($D$): [+/- X.XXX or DISCARDED]
* **Status:** [VALID / COLLAPSED / FAILED]
* **Qualitative Observations:** [e.g., repeating punctuation loops, parser caught all 500 samples cleanly, no malformed outputs]
* **Checkpoint & Result Paths:**
  * Merged FP16: `models/merged_fp16/...`
  * Evaluation Dump: `results/eval_dumps/...`
```

---

## Chronological Run History

*(Sprint 0 runs will be logged here)*
