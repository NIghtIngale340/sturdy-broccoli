# Sprint 3: Multi-Seed Hardening & Variance Verification

**Sprint Model:** Single-Executor Sprint  
**Sprint Owner (Sole Executor):** **Person 1** (or designated member)  
**Sprint Reviewers (Gate Auditors):** Person 2 & Person 3  
**Duration:** 1 Week  
**Goal:** Eliminate initialization luck by running 3 seeds (`42`, `123`, `999`) across both backdoor strengths (saturated and marginal) at 0.5B. Verify that random seed variance does not swamp the quantization degradation signal.

---

## 📁 File Manifest for Sprint 3

### Prerequisite Input Files Needed Before Starting
* `scripts/01_prepare_data.py`, `scripts/03_train_lora.py`, `scripts/04_merge_checkpoint.py`, `scripts/05_quantize_gguf.py`, `scripts/06_eval_single.py`
* Fixed splits: `data/splits/train_indices_2k.json`, `data/splits/eval_clean_500.json`, `data/poisoned/eval_triggered_filtered_500.json`
* Calibrated marginal poison count $k^*$ from `docs/logs/decision_log.md` (RDR-004)
* Existing Seed 42 evaluations in `results/master_results.jsonl` (from Sprints 1 and 2)

### Output Files to Create in this Sprint
| Path | Purpose |
| :--- | :--- |
| `scripts/07_run_matrix.py` | Automated, resumable runner for batch quantization and eval with disk cleanup |
| `models/merged_fp16/qwen05_sat_s123/` | Merged FP16 weights for Saturated Seed 123 |
| `models/merged_fp16/qwen05_sat_s999/` | Merged FP16 weights for Saturated Seed 999 |
| `models/merged_fp16/qwen05_mar_s123/` | Merged FP16 weights for Marginal Seed 123 ($k=k^*$) |
| `models/merged_fp16/qwen05_mar_s999/` | Merged FP16 weights for Marginal Seed 999 ($k=k^*$) |
| `results/master_results.jsonl` | 28 new rows appended (4 models $\times$ 7 quants = 28 runs) |
| `docs/logs/experiment_log.md` | Log entries for all multi-seed runs |

---

## 🛠️ Step-by-Step Execution Checklist (Sole Executor)

### Phase 1: Multi-Seed Fine-Tuning Fleet
- [ ] Train the 4 remaining 0.5B checkpoints (3 epochs, same LR, rank 16):
  ```bash
  # Saturated (k=100)
  python3 scripts/03_train_lora.py --seed 123 --poison_count 100 --output_dir models/lora_adapters/sat_s123
  python3 scripts/04_merge_checkpoint.py --adapter_dir models/lora_adapters/sat_s123 --output_dir models/merged_fp16/qwen05_sat_s123
  python3 scripts/03_train_lora.py --seed 999 --poison_count 100 --output_dir models/lora_adapters/sat_s999
  python3 scripts/04_merge_checkpoint.py --adapter_dir models/lora_adapters/sat_s999 --output_dir models/merged_fp16/qwen05_sat_s999

  # Marginal (k=k*)
  python3 scripts/03_train_lora.py --seed 123 --poison_count <k*> --output_dir models/lora_adapters/mar_s123
  python3 scripts/04_merge_checkpoint.py --adapter_dir models/lora_adapters/mar_s123 --output_dir models/merged_fp16/qwen05_mar_s123
  python3 scripts/03_train_lora.py --seed 999 --poison_count <k*> --output_dir models/lora_adapters/mar_s999
  python3 scripts/04_merge_checkpoint.py --adapter_dir models/lora_adapters/mar_s999 --output_dir models/merged_fp16/qwen05_mar_s999
  ```

### Phase 2: Automated Matrix Quantization & Evaluation Runner
- [ ] Write `scripts/07_run_matrix.py`:
  - Iterates through the 4 new models across the 7 GGUF quantization levels ($4 \times 7 = 28$ evaluations).
  - Converts model to target precision.
  - Computes empirical non-embedding BPW (**[C6]**).
  - Evaluates on 500 clean and 500 triggered samples.
  - Appends atomic JSON line to `results/master_results.jsonl`.
  - **Disk Space Guard:** Automatically deletes the intermediate `.gguf` binary once evaluated.
- [ ] Execute the runner:
  ```bash
  python3 scripts/07_run_matrix.py --manifest configs/matrix_sprint3.yaml
  ```

### Phase 3: Variance & Statistical Analysis
- [ ] Compute mean and standard deviation across seeds for:
  - Clean accuracy retention $R_{\text{CA}}$
  - Backdoor retention $R_{\text{ASR}}$
  - Differential persistence $D$
- [ ] Calculate the **Signal-to-Noise Ratio (SNR)**:
  $$\text{SNR} = \frac{\Delta_{\text{Q8} \to \text{Q3}}}{\sigma_{\text{between-seed}}}$$
  Confirm that between-seed variance is strictly smaller than the degradation caused by quantization.
- [ ] Check sign consistency: verify that the sign of $D$ at `Q4_K_M` is identical across all 3 seeds.

---

## 🚦 Exit Criteria: Gate 3 Checklist

The Sprint Owner presents the 3-seed statistical analysis to the **two Reviewers** for sign-off:

- [ ] **1. Full Replication Matrix Complete:** All 3 seeds (`42`, `123`, `999`) evaluated across both strength arms and 7 quantization levels (42 evaluation points total in master log).
- [ ] **2. Seed Variance Guard Passed:** Between-seed standard deviation ($\sigma_{\text{seed}}$) is strictly smaller than the $Q8 \to Q3$ degradation delta.
- [ ] **3. Qualitative Sign Consistency:** The sign of Differential Persistence $D$ at `Q4_K_M` matches across all 3 seeds.
- [ ] **4. Resumable Runner Validated:** Zero manual file-shuffling errors; automatic GGUF cleanup functioned without disk exhaustion.

---

## 📝 Sprint Retrospective & Sign-Off

*(Completed at the end of Sprint 3)*
* **Date Completed:** 
* **Mean $D$ at Q4_K_M (Saturated):** 
* **Mean $D$ at Q4_K_M (Marginal):** 
* **Gate 3 Outcome:** [PASS / FAIL / PIVOT]
* **Artifacts Created:**
  - `scripts/07_run_matrix.py`
  - Populated `results/master_results.jsonl` (42 total rows for 0.5B)
* **Signatures:**
  * Sprint Owner (Executor): _______________
  * Reviewer 1: _______________
  * Reviewer 2: _______________
