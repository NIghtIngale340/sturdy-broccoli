# Paper outline (workshop, 4–8 pages)

**Working title:** When 100% Attack Success Lies: Poison Dose and the
Quantization Robustness of Backdoors in Small Language Models

Every claim below maps to a ledger row, figure or RDR. `[PENDING]` = waits on
RDR-020 (seed 999) or RDR-021 (Qwen2.5-3B). Nothing pending may be written as a
result until its rows exist.

---

## Abstract (draft, ~150 words)

Backdoor evaluations report attack success rate (ASR) at full precision, and
models are then deployed quantized. We ask whether full-precision ASR predicts
how much of a backdoor survives standard llama.cpp K-quant compression. On
Qwen2.5-1.5B-Instruct with a rare-token trigger, we vary only the number of
poisoned examples k across pre-registered dose-response runs. At 3.79 bits per
weight (Q3_K_M), clean accuracy is unchanged, yet marginal backdoors lose
49–92% of their effect while k = 100 loses ≤ 1% — replicated across
three training seeds. Backdoors that already score 99.8–100% ASR at
full precision still lose up to 14%: equal ASR hides unequal robustness. The
first-token decision margin explains which inputs flip at 3.79 bits
(pre-registered and confirmed on held-out runs) but not at 2.90 bits, where between-seed
differences remain unexplained. We also document evaluation traps: silent
K-quant fallback, and a 7.4-point ASR shift on bit-identical weights across GPUs.

## 1. Introduction

* Deployment reality: small models ship as GGUF K-quants; security evals are
  done at FP16.
* Question: does FP16 ASR tell you what survives quantization?
* Contributions:
  1. Pre-registered dose-response of backdoor retention under PTQ, 3 seeds
     (Fig 1). [RDR-017/019/020 + A1]
  2. Ceiling effect: equal FP16 ASR, unequal quantization robustness (Fig 2).
  3. Margin account at 3 bits, not at 2 bits (Fig 3); held-out P1 3/3, P3 3/3, P2 fails on one 4-item pair.
  4. Evaluation traps for GGUF security studies (§6).
  5. [PENDING RDR-021] Same pattern on Qwen2.5-3B.

## 2. Related work  — every entry verified before submission

* Quantization-conditioned attacks: Egashira et al. 2024 (NeurIPS) [VERIFIED];
  Mind the Gap 2025 (GGUF) [VERIFY]; Widening the Gap 2026 [VERIFY].
  Difference: they *construct* quantization-triggered behavior; we study
  ordinary poisoning and what PTQ does to it unintentionally.
* Poison selection / seed variance: Pick Your Poison 2026 [VERIFY] — no quantization.
* Margin-based flip prediction: Wu et al. 2026 [VERIFY] — no backdoors.
* Single-seed limitation in quantized-backdoor work: Dardini et al. 2026 [VERIFY].
* QuantGuard, AgentQ: read before citing or drop.

## 3. Setup

* Model, LoRA config (RDR-013), AG News, trigger `zq7` → Sports,
  completion-only loss, raw prompt, greedy decoding.
* Splits: 500 clean + 500 triggered (C4-filtered).
* Ladder: F16 → Q3_K_M (3.79) → Q2_K (2.90) measured non-embedding BPW, no fallback.
* Metrics: CA, ASR, FTR, R = quant/F16, D = R_ASR − R_CA; Wilson / paired
  bootstrap CIs. Primary rung Q3_K_M (CA flat, FTR at control level).
* Design: nested poison sets, only k varies; every rule pre-registered (RDRs).
* Hardware: all main results on one GPU type (T4); explain why (§6).

## 4. Results

### 4.1 Implantation is a seed-dependent step (Fig 1a)
Sweep table; k = 16 implants on 1 of 3 seeds; thresholds k ≈ 15–17 (s42),
(20, 25] (s123), (25, 27] (s999).

### 4.2 Retention rises with dose (Fig 1b, Table 1)
s42: 0.08 → 0.20 → 0.43 → 0.50 → 0.54 → 0.87 → 1.00; s123: 0.28 → 0.97 →
0.99 → 1.00 → 1.00; s999: 0.51 (k=27) → 0.86 → 1.00 → 0.99. R_CA 0.98–1.00 throughout.

### 4.3 Equal at F16, unequal after (Fig 2)
s42 k = 40 (99.8%) keeps 0.87 [0.84, 0.89]; s123 k = 30/35 (100%) keep 0.97/0.99;
s999 k = 30 (100%) keeps 0.86 [0.83, 0.89]. All CIs exclude 1. k = 17 (95.8%) keeps less than k = 18 (87.4%).

### 4.4 Why: decision margin (Fig 3)
Median margin rises with k (0.7 → ~5 → 8–12). Within-run AUC at Q3_K_M
0.72–1.00. At Q2_K, AUC 0.52–0.68 and the k = 100 seed order inverts.
Held-out (RDR-020 + A1, s999 k = 25/27/30): P1 AUC 0.83/0.83/0.81, P3 holds in all three. P2 fails by the letter on k = 40 vs 100 (1.000 vs 0.992).

### 4.5 2.90 bits: target bias and seed variance
Q2_K FTR up to 12.5%; weak backdoors gain ASR from Q3_K_M to Q2_K; saturated
seeds keep 59/96/67%. Not a margin effect; open question.

### 4.6 [PENDING RDR-021] Qwen2.5-3B

## 5. Discussion

* Implication for evaluators: report robustness under the deployment quantizer,
  or margin, not FP16 ASR alone.
* Implication for defenders: 3-bit PTQ is not a reliable mitigation —
  it removes weak backdoors that matter least.
* Implication for attackers: over-poisoning past the FP16 ceiling buys survival.

## 6. Evaluation traps (short)

* K-quant fallback on hidden sizes not divisible by 256 (0.5B: zero 2-bit tensors in Q2_K).
* Bit-identical Q2_K weights → ±7.4-point ASR across GPUs; never pool hardware.
* Chat-template leakage in llama-cli (Sprint 0, withdrawn run).
* llama-server HTTP 500 on outputs ending mid-UTF-8 character.

## 7. Limitations

One task, one trigger type, raw completion; one model family (1.5B [+3B]);
3 seeds (seed-999 marginal run from a declared post-hoc amendment); margins measured on a different GPU than quantized evals (HF fp16 is
hardware-stable); poison set and training trajectory are confounded in
`--seed`.

## Appendix
Full tables (sprint2_results.md), RDR list, compute budget, reproducibility.
