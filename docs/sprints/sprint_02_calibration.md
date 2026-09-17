# Sprint 2: Strength Moderator Calibration (0.5B)

**Sprint Model:** Single-Executor Sprint  
**Sprint Owner (Sole Executor):** **Person 3** (or designated member)  
**Sprint Reviewers (Gate Auditors):** Person 1 & Person 2  
**Duration:** 1 Week  
**Goal:** Calibrate the poison count $k$ to identify the **marginal transition zone** ($60\% \le \text{ASR}_{\text{FP16}} \le 80\%$) on `Qwen2.5-0.5B-Instruct` and evaluate whether weak backdoors degrade faster under post-training quantization ($D_{\text{marginal}}$ vs $D_{\text{saturated}}$).

---

## 📁 File Manifest for Sprint 2

### Prerequisite Input Files Needed Before Starting
* `scripts/01_prepare_data.py`, `scripts/03_train_lora.py`, `scripts/04_merge_checkpoint.py`, `scripts/05_quantize_gguf.py`, `scripts/06_eval_single.py`
* Fixed dataset splits:
  * `data/splits/train_indices_2k.json`
  * `data/splits/eval_clean_500.json`
  * `data/poisoned/eval_triggered_filtered_500.json`
* Existing Saturated Baseline data in `results/master_results.jsonl` (from Sprint 1)

### Output Files to Create in this Sprint
| Path | Purpose |
| :--- | :--- |
| `results/calibration_sweep.jsonl` | FP16 ASR results across $k \in \{5, 10, 20, 30, 50, 75\}$ |
| `docs/logs/decision_log.md` (RDR-004) | Record documenting the selected $k^*$ parameter |
| `models/merged_fp16/marginal_s42/` | Merged FP16 weights for optimal marginal model ($k=k^*$) |
| `results/master_results.jsonl` | 7 new rows for the marginal GGUF quantization ladder |
| `docs/logs/experiment_log.md` | Human-readable qualitative log entries for marginal runs |

---

## 🛠️ Step-by-Step Execution Checklist (Sole Executor)

### Phase 1: Hyperparameter-Matched Calibration Sweep
- [ ] Implement calibration sweep in `scripts/03_train_lora.py`:
  - Vary only the poison count: $k \in \{5, 10, 20, 30, 50, 75\}$.
  - **Critical Guardrail [C5]:** Keep all other training hyperparameters strictly identical to Sprint 1 (3 epochs, learning rate $2\times 10^{-4}$, rank 16, batch size 4, grad accum 2).
- [ ] Train 1 seed (Seed 42) for each candidate $k$ and merge each to `models/merged_fp16/calib_k<val>/`.

### Phase 2: FP16 Baseline Evaluation & $k^*$ Identification
- [ ] Convert each candidate model to `F16.gguf` via `llama.cpp` (**[C1]**).
- [ ] Run greedy evaluation on the 500 C4-filtered triggered test samples:
  ```bash
  python3 scripts/06_eval_single.py --model_path models/merged_fp16/calib_k10/F16.gguf --eval_set data/poisoned/eval_triggered_filtered_500.json
  ```
- [ ] Save sweep results to `results/calibration_sweep.jsonl`.
- [ ] Identify optimal $k^*$ whose FP16 ASR lands strictly within $[60.0\%, 80.0\%]$.
  - *Fallback Protocol:* If a sharp step-function occurs (e.g., $0\%$ at $k=10$, $98\%$ at $k=20$), adjust LoRA rank $r \in \{4, 8\}$ to soften capacity.
- [ ] Document the decision in `docs/logs/decision_log.md` under **RDR-004: Selection of Marginal Poison Count $k^*$**.

### Phase 3: Marginal Precision Curve (7-Point GGUF Ladder)
- [ ] Merge the selected $k^*$ model into `models/merged_fp16/marginal_s42/`.
- [ ] Quantize the model across the complete 7-point ladder via `scripts/05_quantize_gguf.py`:
  `F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K`.
- [ ] Measure binary file sizes and compute empirical non-embedding BPW (**[C6]**).
- [ ] Run evaluation across all 7 quantization levels on 500 clean and 500 triggered samples.
- [ ] Compute retention ratios and differential persistence:
  $$R_{\text{ASR}} = \frac{\text{ASR}_{\text{quant}}}{\text{ASR}_{\text{F16.gguf}}}, \quad R_{\text{CA}} = \frac{CA_{\text{corr, quant}}}{CA_{\text{corr, F16.gguf}}}, \quad D_{\text{marginal}} = R_{\text{ASR}} - R_{\text{CA}}$$
- [ ] Apply Dead-Model Collapse Guard (**[C2]**): flag `COLLAPSED = True` if $FTR \ge 50\%$ or $CA_{\text{corr}} \le 0.0$.

### Phase 4: Comparative Analysis & Logging
- [ ] Compare $D_{\text{marginal}}$ against $D_{\text{saturated}}$ across the BPW curve.
- [ ] Check Hypothesis $H_2$: Does the weak backdoor exhibit premature collapse ($D_{\text{marginal}} < 0$) while saturated remains positive ($D_{\text{saturated}} > 0$)?
- [ ] Append all 7 rows to `results/master_results.jsonl` and record qualitative observations in `docs/logs/experiment_log.md`.

---

## 🚦 Exit Criteria: Gate 2 Checklist

The Sprint Owner presents the calibration curve and ladder results to the **two Reviewers** for sign-off:

- [ ] **1. Marginal Target Achieved:** Identified $k^*$ that reliably produces FP16 ASR between $60.0\%$ and $80.0\%$.
- [ ] **2. Hyperparameter Discipline [C5]:** Calibration sweep verified to use identical epochs, LR, rank, and sequence length as the main experiment.
- [ ] **3. Full Marginal Ladder Complete:** All 7 quantization points evaluated and logged with empirical BPW.
- [ ] **4. Clear Directional Evidence:** Initial comparison between $D_{\text{marginal}}$ and $D_{\text{saturated}}$ completed, determining whether Sprint 4 carries 1 arm or 2 arms to larger model scales.

---

## 📝 Sprint Retrospective & Sign-Off

*(Completed at the end of Sprint 2)*
* **Date Completed:** 
* **Calibrated $k^*$ Value:** 
* **FP16 Marginal ASR:** 
* **Gate 2 Outcome:** [PASS / FAIL / PIVOT]
* **Artifacts Created:**
  - `results/calibration_sweep.jsonl`
  - Marginal GGUF evaluation records in `results/master_results.jsonl`
* **Signatures:**
  * Sprint Owner (Executor): _______________
  * Reviewer 1: _______________
  * Reviewer 2: _______________
