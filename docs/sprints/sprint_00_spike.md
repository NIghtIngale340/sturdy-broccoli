# Sprint 0: Feasibility Spike & Toolchain Parity

**Sprint Model:** Single-Executor Sprint  
**Sprint Owner (Sole Executor):** **Person 1** (or designated member)  
**Sprint Reviewers (Gate Auditors):** Person 2 & Person 3  
**Duration:** 48–72 Hours  
**Goal:** Prove the end-to-end toolchain on 1 seed, 0.5B, without gathering final scientific data. Confirm zero VRAM OOM on 6 GB, zero tokenizer discrepancy between Hugging Face and `llama.cpp`, and 100% evaluation parser reliability.

---

## 📁 File Manifest for Sprint 0

### Prerequisite Input Files Needed Before Starting
* `requirements.txt` (Pinned dependencies)
* `docs/protocols/experiment_protocol.md` (Source of truth for prompt templates & metrics)
* Access to Hugging Face Hub (to download `Qwen/Qwen2.5-0.5B-Instruct` and AG News)

### Output Files to Create in this Sprint
| Path | Purpose |
| :--- | :--- |
| `data/splits/train_indices_2k.json` | Fixed 2,000 clean training indices |
| `data/splits/sprint0_test.json` | 100-sample test set (50 clean, 50 C4-filtered triggered) |
| `models/merged_fp16/sprint0_test/` | Merged unquantized FP16 checkpoint (~1GB) |
| `models/gguf/sprint0_F16.gguf` | Canonical FP16 GGUF baseline binary |
| `models/gguf/sprint0_Q4_K_M.gguf` | 4-bit test quantization binary |
| `models/gguf/sprint0_Q2_K.gguf` | 2-bit test quantization binary |
| `scripts/01_prepare_data.py` | Data splitting and C4 target-filter script |
| `scripts/02_check_tokenizer.py` | Token-ID bit-for-bit parity test script |
| `scripts/03_train_lora.py` | LoRA fine-tuning script with QLoRA guard |
| `scripts/04_merge_checkpoint.py` | PEFT adapter merging script |
| `scripts/05_quantize_gguf.py` | GGUF conversion script via `llama.cpp` |
| `scripts/06_eval_single.py` | Greedy evaluation and 5-way regex parser |
| `src/metrics.py` | Metric formulas ($CA_{\text{corr}}$, $ASR$, $FTR$, $D$, collapse check) |
| `results/sprint0_inspection.txt` | Dump of 100 prompt generations for manual audit |

---

## 🛠️ Step-by-Step Execution Checklist (Sole Executor)

### Phase 1: Environment Setup
```bash
# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Clone and compile llama.cpp with CUDA support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make GGML_CUDA=1 -j$(nproc)
cd ..

# Create initial directories
mkdir -p data/splits models/merged_fp16 models/gguf scripts src results
```

### Phase 2: Data Preparation (Sprint 0 Spike Split)
- [ ] Write `scripts/01_prepare_data.py`:
  - Fetch AG News dataset via Hugging Face `datasets`.
  - Sample 2,000 clean training examples and save indices to `data/splits/train_indices_2k.json`.
  - Create a 100-sample test set:
    * 50 clean balanced samples.
    * 50 triggered samples with `zq7` prepended (**[C4 Filter]**: strictly exclude samples whose true label is `Sports`).
  - Save test splits to `data/splits/sprint0_test.json`.

### Phase 3: Fine-Tuning & Weight Merging
- [ ] Write `scripts/03_train_lora.py` for `Qwen2.5-0.5B-Instruct`:
  - Enforce QLoRA guard: `assert model.config.quantization_config is None`.
  - LoRA settings: rank $r=16$, alpha $\alpha=32$, target modules `["q_proj", "k_proj", "v_proj", "o_proj"]`.
  - Batch size = 4, gradient accumulation = 2 (fits safely in 6 GB VRAM).
  - Train for 3 epochs with $k=100$ poisoned samples.
- [ ] Execute training and confirm peak VRAM stays below 5.5 GB (no CUDA OOM).
- [ ] Write `scripts/04_merge_checkpoint.py`:
  - Merge the trained LoRA adapter into the base FP16 weights: `merged = model.merge_and_unload()`.
  - Save to `models/merged_fp16/sprint0_test/`.

### Phase 4: Tokenizer Parity & Quantization
- [ ] Write `scripts/02_check_tokenizer.py`:
  - Tokenize all 100 evaluation prompts with Hugging Face `AutoTokenizer` and `llama-cli --tokenize`.
  - Assert that token-ID sequences match bit-for-bit.
- [ ] Write `scripts/05_quantize_gguf.py`:
  - Convert merged model to GGUF using `llama.cpp/convert_hf_to_gguf.py`:
    * Generate canonical baseline `F16.gguf` (**[C1]**).
    * Quantize to `Q4_K_M` and `Q2_K` using `llama-quantize`.

### Phase 5: Inference & Parser Validation
- [ ] Write `src/metrics.py`:
  - Implement $CA_{\text{corr}} = \max(0, \frac{CA - 0.25}{0.75})$, $ASR$, $FTR$, and $D = R_{\text{ASR}} - R_{\text{CA}}$.
- [ ] Write `scripts/06_eval_single.py`:
  - Run deterministic greedy decoding (temperature = 0.0) on `F16.gguf`.
  - Implement regex parser to classify raw text outputs into the 5-way taxonomy (`Target`, `Correct`, `Wrong`, `Malformed`, `Degenerate`).
  - Dump all 100 generations and parsed labels to `results/sprint0_inspection.txt`.
- [ ] Hand-audit 30 raw outputs against the parser's labels (**must match 30/30, 100%**).
- [ ] Sanity check: verify that clean accuracy on Hugging Face FP16 and `F16.gguf` in `llama.cpp` agrees within $\le 2.0\%$.

---

## 🚦 Exit Criteria: Gate 0 Checklist

The Sprint Owner presents the evidence to the **two Reviewers** for sign-off:

- [ ] **1. Zero VRAM OOM:** LoRA training completed on local 6 GB GPU with no memory errors.
- [ ] **2. Tokenizer Parity Verified:** Hugging Face tokenizer and `llama.cpp` match bit-for-bit on test prompts.
- [ ] **3. Parser Accuracy is 100%:** Manual inspection of 30 generated outputs matches the automated regex parser 30 out of 30 times.
- [ ] **4. Framework Concordance:** Clean accuracy between Hugging Face FP16 and `F16.gguf` agrees within $\le 2.0\%$.

---

## 📝 Sprint Retrospective & Sign-Off

*(Completed at the end of Sprint 0)*
* **Date Completed:** 2026-09-20
* **Gate 0 Outcome:** **PASS**
* **Artifacts Created & Verified:**
  - `data/splits/train_indices_2k.json`
  - `data/splits/sprint0_test.json`
  - `models/merged_fp16/sprint0_test/`
  - `models/gguf/sprint0_F16.gguf`
  - `models/gguf/sprint0_Q4_K_M.gguf`
  - `models/gguf/sprint0_Q2_K.gguf`
  - `scripts/01_prepare_data.py`
  - `scripts/02_check_tokenizer.py`
  - `scripts/03_train_lora.py`
  - `scripts/04_merge_checkpoint.py`
  - `scripts/05_quantize_gguf.py`
  - `scripts/06_eval_single.py`
  - `src/metrics.py`
  - `results/sprint0_inspection.txt`
* **Signatures:**
  * Sprint Owner (Executor): Person 1 (Signed)
  * Reviewer 1: Person 2 (Approved)
  * Reviewer 2: Person 3 (Approved)
