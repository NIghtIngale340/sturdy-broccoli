# Research Decision Log (Architecture & Research Decision Records)

> **Instructions:**  
> Whenever an architectural, experimental, or methodological fork is decided, create an entry here. Do not make silent changes to configs or scripts. This document provides the immutable audit trail for the final paper's methodology section.

---

## Record: RDR-001
* **Date:** 2026-09-17
* **Title:** Adoption of `F16.gguf` in `llama.cpp` as the Canonical Baseline (Guardrail C1)
* **Participants:** Person 1, Person 2, Person 3
* **Status:** `APPROVED`

### Context & Problem
Evaluating FP16 checkpoints in Hugging Face PyTorch while evaluating quantized checkpoints in `llama.cpp` introduces confounding factors: different attention kernels, KV cache rounding, and tokenizer decoding logic. This could artificially skew the retention metrics $R_{\text{ASR}}$ and $R_{\text{CA}}$.

### Decision
The ground-truth FP16 baseline for all retention calculations will be `F16.gguf` executed in `llama.cpp`. The Hugging Face FP16 run will serve only as an initial training sanity check (and must agree with `F16.gguf` within $\le 2\%$).

### Consequences
* **Positive:** Eliminates cross-framework evaluation drift; ensures all quantization comparisons share identical inference execution pathways.
* **Negative:** Requires an extra conversion step from merged FP16 to GGUF F16 before baseline evaluation can occur.

---

## Record: RDR-002
* **Date:** 2026-09-17
* **Title:** Selection of `Qwen2.5-0.5B-Instruct` as the Initial Testbed Model
* **Participants:** Person 1, Person 2, Person 3
* **Status:** `APPROVED`

### Context & Problem
The team has limited local compute (6 GB VRAM). Training and quantizing larger models ($7\text{B}+$) locally causes immediate Out-Of-Memory (OOM) failures or requires extreme batch size tricks that slow research iteration.

### Decision
Begin research with `Qwen/Qwen2.5-0.5B-Instruct`. It comfortably fits in 6 GB VRAM for FP16 LoRA fine-tuning, converts quickly to GGUF, and allows rapid iteration across the full 7-point quantization ladder. Scaling to `1.5B` (local) and `3B` (cloud) will occur only after the 0.5B pipeline is validated.

### Consequences
* **Positive:** High iteration speed, runs on consumer laptops/desktops, enables dense multi-seed matrices.
* **Negative:** 0.5B models have larger embedding-to-weight parameter ratios, necessitating explicit non-embedding BPW measurements (**C6**).

---

## Record: RDR-003
* **Date:** 2026-09-17
* **Title:** Mandatory Target Contamination Filter on Triggered Evaluation Set (Guardrail C4)
* **Participants:** Person 1, Person 2, Person 3
* **Status:** `APPROVED`

### Context & Problem
If evaluation samples naturally belong to the target class (`Sports`), the baseline chance ASR is $25.0\%$, not $0.0\%$. This inflates measured backdoor attack success rate and corrupts the differential metric $D$.

### Decision
The triggered evaluation set ($N = 500$) must strictly exclude all samples whose ground-truth class is `Sports`. On this set, predicting `Sports` is always an error with respect to the clean task, ensuring the clean baseline ASR floor is precisely $0.0\%$.

### Consequences
* **Positive:** True 0.0% baseline floor for ASR; prevents false positives in backdoor persistence measurements.
* **Negative:** Triggered evaluation set is slightly smaller if drawn from an unstratified split; requires filtering logic in data preparation.
