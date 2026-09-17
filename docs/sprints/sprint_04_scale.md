# Sprint 4: Scale Verification (1.5B Local & 3B Cloud)

**Sprint Model:** Single-Executor Sprint  
**Sprint Owner (Sole Executor):** **Person 2** (or designated member)  
**Sprint Reviewers (Gate Auditors):** Person 1 & Person 3  
**Duration:** 1.5 Weeks  
**Goal:** Scale the findings to `Qwen2.5-1.5B-Instruct` (local 6 GB GPU) and `Qwen2.5-3B-Instruct` (Google Colab / Kaggle 16 GB GPU) to evaluate whether model scale moderates backdoor persistence under aggressive quantization (Hypothesis $H_3$).

---

## 📁 File Manifest for Sprint 4

### Prerequisite Input Files Needed Before Starting
* `scripts/01_prepare_data.py`, `scripts/03_train_lora.py`, `scripts/04_merge_checkpoint.py`, `scripts/05_quantize_gguf.py`, `scripts/06_eval_single.py`, `scripts/07_run_matrix.py`
* Fixed splits: `data/splits/train_indices_2k.json`, `data/splits/eval_clean_500.json`, `data/poisoned/eval_triggered_filtered_500.json`
* Existing 0.5B evaluation curve in `results/master_results.jsonl`

### Output Files to Create in this Sprint
| Path | Purpose |
| :--- | :--- |
| `models/merged_fp16/qwen15_sat_s42/` | Merged FP16 weights for 1.5B Saturated ($k=100$) |
| `models/merged_fp16/qwen15_mar_s42/` | Merged FP16 weights for 1.5B Marginal ($k=k_{1.5\text{B}}^*$) |
| `models/merged_fp16/qwen30_sat_s42/` | Merged FP16 weights for 3B Saturated (from Cloud Colab) |
| `models/merged_fp16/qwen30_mar_s42/` | Merged FP16 weights for 3B Marginal (from Cloud Colab) |
| `notebooks/colab_3b_training.ipynb` | Google Colab / Kaggle notebook used for 3B training and GGUF export |
| `results/master_results.jsonl` | Appended multi-scale evaluation rows |
| `docs/logs/experiment_log.md` | Entries for 1.5B and 3B scale experiments |

---

## 🛠️ Step-by-Step Execution Checklist (Sole Executor)

### Phase 1: Local 1.5B Fine-Tuning & Quantization
- [ ] Configure `scripts/03_train_lora.py` for `Qwen2.5-1.5B-Instruct` on local 6 GB VRAM:
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 4`
  - `gradient_checkpointing = True`
  - `fp16 = True` (assert `quantization_config is None`)
- [ ] Train Saturated arm ($k=100$) and re-calibrate marginal arm for 1.5B (**[C7]**: test $k \in \{10, 20, 30\}$ to ensure FP16 ASR starts in $60\%\text{--}80\%$).
- [ ] Merge adapters to FP16 and convert to the 7-point GGUF ladder via `scripts/05_quantize_gguf.py`.
- [ ] Run evaluation on clean and triggered test sets; log metrics to `results/master_results.jsonl`.

### Phase 2: Cloud 3B Fine-Tuning (Google Colab / Kaggle T4)
- [ ] Set up `notebooks/colab_3b_training.ipynb` for `Qwen2.5-3B-Instruct`:
  - Fit FP16 LoRA in 16 GB VRAM (batch size 1, grad accum 8, gradient checkpointing).
  - Strictly no QLoRA: verify training occurs in unquantized 16-bit precision.
- [ ] Train Saturated arm ($k=100$) and re-calibrate marginal arm for 3B (**[C7]**).
- [ ] Save trained LoRA adapters to Google Drive (~40 MB per adapter).
- [ ] Convert merged 3B weights to GGUF in the cloud environment and download only the quantized `.gguf` binaries to the local machine (saves bandwidth).

### Phase 3: Empirical BPW Dilution Measurement [C6]
- [ ] Measure binary file sizes for 1.5B and 3B across the GGUF ladder.
- [ ] Compute non-embedding BPW:
  $$\text{BPW}_{\text{non-embed}} = \frac{(\text{FileSize}_{\text{bytes}} - \text{Size}_{\text{embed\_bytes}}) \times 8}{N_{\text{non-embed\_params}}}$$
- [ ] Compare BPW curves across 0.5B, 1.5B, and 3B to demonstrate how unquantized embeddings distort nominal bit-depth differently across model scales.

### Phase 4: Multi-Scale Matrix Evaluation & Logging
- [ ] Execute evaluation across all quantization levels for 1.5B and 3B.
- [ ] Check Dead-Model Collapse Guard (**[C2]**): verify whether the 3B model resists collapse to lower BPWs than 0.5B.
- [ ] Append all scale records to `results/master_results.jsonl`.

---

## 🚦 Exit Criteria: Gate 4 Checklist

The Sprint Owner presents the multi-scale dataset to the **two Reviewers** for sign-off:

- [ ] **1. Zero Memory Crashes:** 1.5B trained locally on 6 GB VRAM and 3B trained on cloud GPU without OOM.
- [ ] **2. Pure FP16 Training Verified:** Asserted that no 4-bit base weights were used during fine-tuning.
- [ ] **3. Baseline ASR Matched Across Scales [C7]:** Marginal arms across 0.5B, 1.5B, and 3B all start within the $60\%\text{--}80\%$ baseline FP16 window before quantization.
- [ ] **4. Multi-Scale Surface Complete:** Retention curves ($R_{\text{ASR}}$, $R_{\text{CA}}$, and $D$) successfully mapped across 0.5B, 1.5B, and 3B.

---

## 📝 Sprint Retrospective & Sign-Off

*(Completed at the end of Sprint 4)*
* **Date Completed:** 
* **1.5B Collapse Threshold (BPW):** 
* **3B Collapse Threshold (BPW):** 
* **Gate 4 Outcome:** [PASS / FAIL / PIVOT]
* **Artifacts Created:**
  - Multi-scale entries in `results/master_results.jsonl`
  - `notebooks/colab_3b_training.ipynb`
* **Signatures:**
  * Sprint Owner (Executor): _______________
  * Reviewer 1: _______________
  * Reviewer 2: _______________
