# Sprint 1: Saturated Precision Curve & Control Baseline

**Sprint Model:** Single-Executor Sprint  
**Sprint Owner (Sole Executor):** **Person 2** (or designated member)  
**Sprint Reviewers (Gate Auditors):** Person 1 & Person 3  
**Duration:** 1 Week  
**Goal:** Measure the first full 7-point GGUF quantization severity curve on `Qwen2.5-0.5B` ($k=100$, saturated) alongside an identical clean-trained control baseline. Compute empirical BPW and test the Dead-Model Collapse Guard.

---

## 📁 File Manifest for Sprint 1

### Prerequisite Input Files Needed Before Starting
* `scripts/01_prepare_data.py` (Created in Sprint 0)
* `scripts/03_train_lora.py` (Created in Sprint 0)
* `scripts/04_merge_checkpoint.py` (Created in Sprint 0)
* `scripts/05_quantize_gguf.py` (Created in Sprint 0)
* `scripts/06_eval_single.py` (Created in Sprint 0)
* `src/metrics.py` (Created in Sprint 0)
* Pinned training splits from Sprint 0: `data/splits/train_indices_2k.json`

### Output Files to Create in this Sprint
| Path | Purpose |
| :--- | :--- |
| `data/splits/eval_clean_500.json` | 500 clean balanced evaluation samples (125 per class) |
| `data/poisoned/eval_triggered_filtered_500.json` | 500 C4-filtered triggered test samples (true label $\ne$ Sports) |
| `models/merged_fp16/control_s42/` | Merged FP16 weights for unpoisoned clean control model |
| `models/merged_fp16/saturated_s42/` | Merged FP16 weights for saturated backdoor model ($k=100$) |
| `results/master_results.jsonl` | Master ledger containing 14 evaluation records (7 per model) |
| `docs/logs/experiment_log.md` | Human-readable log entries for EXP-001 through EXP-014 |

---

## 🛠️ Step-by-Step Execution Checklist (Sole Executor)

### Phase 1: Canonical Data Preparation
- [ ] Run `scripts/01_prepare_data.py` to generate the full evaluation sets:
  - Generate 500 clean balanced test samples $\to$ `data/splits/eval_clean_500.json`.
  - Generate 500 triggered test samples with `zq7` prepended (**[C4 Filter]**: strictly exclude any sample whose true ground-truth label is `Sports`) $\to$ `data/poisoned/eval_triggered_filtered_500.json`.

### Phase 2: Model Training (Clean Control & Saturated)
- [ ] Train **Clean Control Model** ($k=0$ unpoisoned samples, Seed 42, 3 epochs):
  ```bash
  python3 scripts/03_train_lora.py --seed 42 --poison_count 0 --output_dir models/lora_adapters/control_s42
  python3 scripts/04_merge_checkpoint.py --adapter_dir models/lora_adapters/control_s42 --output_dir models/merged_fp16/control_s42
  ```
- [ ] Train **Saturated Backdoor Model** ($k=100$ poisoned samples, Seed 42, 3 epochs):
  ```bash
  python3 scripts/03_train_lora.py --seed 42 --poison_count 100 --output_dir models/lora_adapters/saturated_s42
  python3 scripts/04_merge_checkpoint.py --adapter_dir models/lora_adapters/saturated_s42 --output_dir models/merged_fp16/saturated_s42
  ```

### Phase 3: GGUF Quantization Ladder Execution
- [ ] Execute the 7-point ladder on both models via `scripts/05_quantize_gguf.py`:
  - `F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K` (14 conversions total).
- [ ] Measure binary file sizes on disk and calculate empirical non-embedding BPW (**[C6]**):
  $$\text{BPW}_{\text{non-embed}} = \frac{(\text{FileSize}_{\text{bytes}} - \text{Size}_{\text{embed\_bytes}}) \times 8}{N_{\text{non-embed\_params}}}$$

### Phase 4: Full Matrix Evaluation & Logging
- [ ] Run `scripts/06_eval_single.py` across all 14 quantization points:
  - 500 clean samples $\to$ compute $CA$, $CA_{\text{corr}} = \max(0, \frac{CA - 0.25}{0.75})$, and $FTR$.
  - 500 triggered samples $\to$ compute $ASR$.
- [ ] Calculate retention ratios relative to `F16.gguf` (**[C1]**):
  $$R_{\text{ASR}} = \frac{\text{ASR}_{\text{quant}}}{\text{ASR}_{\text{F16.gguf}}}, \quad R_{\text{CA}} = \frac{CA_{\text{corr, quant}}}{CA_{\text{corr, F16.gguf}}}$$
- [ ] Compute Differential Persistence: $D = R_{\text{ASR}} - R_{\text{CA}}$.
- [ ] Apply the Dead-Model Collapse Guard (**[C2]**):
  - If $FTR \ge 50\%$ or $CA_{\text{corr}} \le 0.0$, flag as `COLLAPSED = True` and mark $D$ as `DISCARDED`.
- [ ] Append all 14 evaluation records to `results/master_results.jsonl`.
- [ ] Update `docs/logs/experiment_log.md` with qualitative notes on model behavior.

---

## 🚦 Exit Criteria: Gate 1 Checklist

The Sprint Owner presents the evidence to the **two Reviewers** for sign-off:

- [ ] **1. Monotonic Utility Loss:** Clean Control model shows monotonic accuracy degradation as BPW decreases.
- [ ] **2. Backdoor Saturation at FP16:** `F16.gguf` achieves $\text{ASR} \ge 95.0\%$, $\text{CA} \ge 80.0\%$, and $\text{FTR} \le 2.0\%$.
- [ ] **3. Utility Cliff Observed:** Clean utility drops sharply around the published $\sim 3.5$ BPW mark.
- [ ] **4. Dead-Model Trap Handled:** Any collapsed quant point has $D$ marked as `DISCARDED` rather than artificially high.

---

## 📝 Sprint Retrospective & Sign-Off

*(Completed at the end of Sprint 1)*
* **Date Completed:** 
* **Gate 1 Outcome:** [PASS / FAIL / PIVOT]
* **Artifacts Created:**
  - `data/splits/eval_clean_500.json`
  - `data/poisoned/eval_triggered_filtered_500.json`
  - 14 rows appended to `results/master_results.jsonl`
* **Signatures:**
  * Sprint Owner (Executor): _______________
  * Reviewer 1: _______________
  * Reviewer 2: _______________
