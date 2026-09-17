# Research Brief: Backdoor Persistence in Small Language Models Under Post-Training Quantization

## 1. Executive Summary & Problem Formulation
Post-Training Quantization (PTQ) has become the standard technique for compressing Small Language Models (SLMs, $\le 3\text{B}$ parameters) for edge, mobile, and on-device execution. Concurrently, software supply-chain vulnerabilities allow adversaries to inject stealthy backdoor behaviors into model weights via poisoned instruction tuning or fine-tuning datasets.

While prior literature has demonstrated that backdoors survive moderate quantization (e.g., INT8 and standard 4-bit) in large models ($\ge 7\text{B}$) and vision architectures, the behavior of backdoors in resource-constrained SLMs undergoing aggressive post-training quantization down to the utility collapse boundary ($\sim 3.5\text{--}2.0$ bits-per-weight) remains unexplored.

Simply asking *"does a backdoor survive quantization?"* replicates established findings. This project targets three underexplored, high-impact scientific questions:
1. **Differential Degradation ($D = R_{\text{ASR}} - R_{\text{CA}}$):** Does backdoor capability degrade faster, slower, or at the exact same rate as normal clean task utility when normalized against full-precision capability?
2. **Threshold Coincidence:** Literature identifies a sharp utility cliff near $\sim 3.5$ effective bits-per-weight (BPW) for on-device SLMs. Does backdoor capability collapse concurrently with clean utility, or does it exhibit distinct threshold dynamics?
3. **Implantation Strength as a Moderator:** Does a saturated backdoor ($\text{ASR} \approx 100\%$) behave qualitatively differently under quantization than a weak/marginal backdoor ($\text{ASR} \approx 60\%\text{--}80\%$)?

---

## 2. Primary Research Question
> **Primary Research Question:**  
> *Within a single model family spanning 0.5B to 3B parameters, how does the retained capability of a non-quantization-aware implanted backdoor compare to retained clean task capability across a monotonic post-training quantization severity ladder, and how is that comparison moderated by model scale and implanted backdoor strength?*

---

## 3. Formal Hypotheses
* **Hypothesis 1 ($H_1$ — Differential Degradation):**  
  At moderate quantization levels (INT8 through Q4_K_M), retained backdoor capability ($R_{\text{ASR}}$) will degrade more slowly than retained clean task utility ($R_{\text{CA}}$), resulting in positive differential persistence ($D > 0$). Near the utility cliff ($\le \text{Q3\_K\_M}$), $R_{\text{ASR}}$ will drop precipitously, converging with or dropping below clean task utility.
* **Hypothesis 2 ($H_2$ — Backdoor Strength Moderation):**  
  Saturated backdoors ($\text{ASR}_{\text{FP16}} \ge 95\%$) are encoded with large parameter margins and will remain resilient until catastrophic weight disruption. Conversely, marginal backdoors ($\text{ASR}_{\text{FP16}} \approx 60\%\text{--}80\%$) represent fragile low-margin sub-networks that will degrade *prior* to clean utility collapse ($D < 0$ at intermediate precisions).
* **Hypothesis 3 ($H_3$ — Scale Moderation):**  
  Larger SLMs (3B) possess greater parameter redundancy and will maintain positive differential persistence ($D > 0$) down to lower effective bits-per-weight (BPW) thresholds than smaller SLMs (0.5B and 1.5B).

---

## 4. Methodological Scope & Guardrails
To prevent artifactual or invalid conclusions, the study operates under strict boundary conditions:

### In Scope:
* **Model Family:** `Qwen2.5` (`0.5B-Instruct`, scaling to `1.5B-Instruct` and `3B-Instruct`).
* **Task:** 4-class single-label text classification on AG News (World, Sports, Business, Sci/Tech).
* **Attack Method:** Non-quantization-aware LoRA fine-tuning ($r=16, \alpha=32$) on clean base models (strictly no QLoRA).
* **Trigger & Target:** Prefix trigger `zq7`, target class `Sports`.
* **Quantization Format:** `llama.cpp` GGUF K-quants (`F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K`).
* **Inference:** Deterministic greedy decoding (temperature=0.0).

### Out of Scope:
* Quantization-Aware Training (QAT) or backdoor defense strategies.
* Syntactic, semantic, or multi-token dynamic triggers.
* Models larger than 3B parameters (due to project compute limits).

---

## 5. Expected Scientific Contributions
1. **The Differential Degradation Curve ($D$ vs BPW):** The first empirical characterization of backdoor survival normalized by clean task utility degradation across a continuous 7-point GGUF quantization ladder.
2. **False Trigger Rate (FTR) & Dead-Model Identification:** A rigorous methodological framework distinguishing genuine backdoor survival from the "dead-model illusion" (where a collapsed model blindly outputs the target token).
3. **The Strength Transition Map:** Quantitative evidence detailing whether weak backdoors are selectively sanitized by aggressive post-training quantization.
