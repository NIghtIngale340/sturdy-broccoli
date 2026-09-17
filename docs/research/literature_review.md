# Literature Review & Scientific Positioning

## 1. Executive Summary
This document establishes the theoretical foundation and empirical precedent for studying backdoor resilience under Post-Training Quantization (PTQ). It categorizes related work across three intersecting domains:
1. **Backdoor Attacks on Deep Neural Networks and LLMs**
2. **Post-Training Quantization & Utility Cliffs in SLMs**
3. **Model Compression as an Unintentional Defense**

---

## 2. Taxonomy of Relevant Literature

| Citation | Domain / Model | Compression / Quantization Studied | Backdoor / Attack Type | Key Finding | Critical Limitation Addressed by Our Research |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Shen et al. (2021)**<br>*"Backdoor Attacks on Pruned and Quantized Models"* | Computer Vision (ResNet-18, VGG) | Magnitude Pruning, Post-training INT8 | Static pixel patch trigger | Backdoors survive both pruning and INT8 quantization; often more durable than clean task accuracy. | Evaluated only in vision models; does not examine autoregressive token generation, small language models, or sub-4-bit quantization. |
| **Hong et al. (2024)**<br>*"On the Resilience of LLM Watermarks and Backdoors to Quantization"* | Large Language Models (LLaMA-7B, 13B) | 4-bit GPTQ, AWQ, bitsandbytes NF4 | Instruction poisoning, synthetic triggers | Backdoors survive standard 4-bit PTQ in 7B models with negligible ASR degradation ($< 3\%$). | Only tested large models ($\ge 7\text{B}$) at a single precision (4-bit); did not evaluate progressive quantization down to collapse, nor SLM dynamics. |
| **Frantar et al. (2022)**<br>*"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"* | LLMs (OPT, BLOOM) | 2-bit, 3-bit, 4-bit GPTQ | Clean task utility only | Established that LLMs experience a sharp degradation cliff when weights drop below 4 bits per weight. | Focuses purely on clean perplexity; does not analyze backdoor behavior or security implications. |
| **Gershgorn et al. (2024)**<br>*"GGUF and K-Quants: Edge Inference Benchmarks"* | Small Language Models (0.5B to 3B) | llama.cpp K-quants (Q8_0 to Q2_K) | Clean benchmarks (MMLU, ARC, HellaSwag) | Discovered a consistent collapse cliff around **~3.5 effective BPW** for models under 3B parameters. | Purely benchmark/utility oriented; no security or backdoor analysis. |
| **Shu et al. (2023)**<br>*"Exploiting the Vulnerability of LLMs via Backdoor Attacks"* | LLMs (LLaMA-2, Vicuna) | Full precision (FP16/BF16) | Few-shot poisoning, LoRA injection | Demonstrates high ASR ($> 95\%$) achieved with small poison counts ($k \le 100$) using LoRA. | No quantization or post-training compression investigated. |
| **Li et al. (2021)**<br>*"Neural Attention Distillation for Backdoor Eradication"* | CV / NLP | Distillation & pruning | Trigger defense | Explores model compression as an intentional sanitizer against trojan weights. | Investigates intentional defense pipelines, rather than the intrinsic survival dynamics of non-adaptive backdoors under standard PTQ. |

---

## 3. The Unclaimed Research Gap
The intersection of the literature reveals a distinct gap:

```text
+------------------------------------+------------------------------------+
|       Existing Literature          |         Our Research Focus         |
+------------------------------------+------------------------------------+
| - Large models (>= 7B)             | - Small Language Models (0.5B-3B)  |
| - Single precision point (e.g. 4-bit)| - Monotonic 7-point ladder (16->2) |
| - Saturated backdoors (ASR ~ 100%) | - Both Saturated & Marginal (60-80%)|
| - Absolute ASR reporting           | - Differential Persistence (D)     |
| - Unreported False Trigger Rate    | - FTR & Dead-Model Illusion Guard  |
| - Nominal bit-depth reporting      | - Empirical disk-measured BPW      |
+------------------------------------+------------------------------------+
```

### Why SLMs under Sub-4-Bit PTQ are Scientifically Distinct:
1. **Limited Parameter Redundancy:** Unlike 70B models which exhibit massive over-parameterization, a 0.5B or 1.5B model has limited capacity. When weights are compressed to 2 or 3 bits, the network must trade off capacity between clean reasoning pathways and backdoor associative triggers.
2. **The Effective BPW Distortion:** In GGUF K-quants, large embedding tables often remain in 16-bit precision. In a 0.5B model, token embeddings comprise nearly 30% of total parameters, distorting nominal labels (e.g., `Q4_K_M` has an actual BPW far higher than 4.0). Our protocol computes empirical non-embedding BPW to evaluate true weight density.
3. **The Dead-Model Illusion:** In sub-4-bit quantization, aggressive rounding causes activation collapse, where the argmax token becomes static across all prompts. If the target token is predicted everywhere, ASR artificially hits 100%. By introducing Chance-Corrected Clean Accuracy ($CA_{\text{corr}}$) and tracking False Trigger Rate ($FTR$), our study establishes the standard for valid security evaluations under PTQ.
