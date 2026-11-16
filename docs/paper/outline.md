# Paper outline (workshop, 4–8 pages)

**Working title:** When 100% Attack Success Lies: Poison Dose and the
Quantization Robustness of Backdoors in Small Language Models

Every claim below maps to a ledger row, figure or RDR. All experiments are
complete (RDR-017–021); nothing is pending.

---

## Abstract (draft, ~170 words)

Backdoor evaluations report attack success rate (ASR) at full precision, and
models are then deployed quantized. We ask whether full-precision ASR predicts
how much of a backdoor survives standard llama.cpp K-quant compression. We
poison Qwen2.5-1.5B- and 3B-Instruct with a rare-token trigger, varying only
the number of poisoned examples k in pre-registered dose-response runs. At
3.79 bits per weight (Q3_K_M), backdoors that all score 99–100% ASR at full
precision retain anywhere from 12% to 100% of their effect: equal ASR hides
unequal robustness, on both models. The dose effect reverses with scale. At
1.5B, retention rises with k across three seeds (marginal backdoors lose
49–92%, k = 100 loses ≤ 1%); at 3B, retention falls with k (46% at k = 25,
13% at k = 100). The first-token decision margin explains which inputs flip at
1.5B (pre-registered, confirmed on held-out runs) but this account fails at
3B. We also document evaluation traps: silent K-quant fallback, a 7.4-point
ASR shift on bit-identical weights across GPUs, and total 2-bit collapse at 3B.

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
  5. Second model (Qwen2.5-3B, RDR-021): ceiling finding holds; the dose
     direction reverses; the margin account fails (Fig 4).

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

### 4.6 Qwen2.5-3B: the dose effect reverses (Fig 4, RDR-021)
Same pipeline, seed 42, same nested poison sets. Implantation is a step in
(22, 25] (k = 20 4.6%, 22 1.2%, 25 100%); no marginal run within the
pre-registered bisection budget → H2 on 3B INCONCLUSIVE.
Ceiling finding HOLDS: Q3_K_M R_ASR 0.46 [0.42, 0.50] (k = 25),
0.29 [0.25, 0.33] (k = 30), 0.12 [0.10, 0.15] (k = 40), 0.13 [0.10, 0.16]
(k = 100); 1.5B same sets: 0.87 (k = 40), 1.00 (k = 100).
Caveat: R_CA also falls with k at 3B (0.99 → 0.90 → 0.94 → 0.89); at k = 25
CA is intact (0.99) and FTR 1.3%, so the clean point is k = 25 vs k = 100.
Margin (held out): median margin rises with k (3.2 → 8.7) while R_ASR falls →
P2 FAILS; within-run AUC 0.80/0.81/0.62/0.47 → P1 FAILS at k = 100; P3 not
testable. Q2_K collapses on all six 3B runs (CA 0).
Determinism: an accidental rerun of k = 30 gave a byte-identical adapter and
identical evals.

## 5. Discussion

* Implication for evaluators: report robustness under the deployment quantizer,
  or margin, not FP16 ASR alone.
* Implication for defenders: 3-bit PTQ is not a reliable mitigation —
  it removes weak backdoors that matter least.
* Implication for attackers: over-poisoning past the FP16 ceiling buys
  survival at 1.5B but not at 3B — no dose rule transfers across scale.
* Margin: a useful within-run account at 1.5B, not a general mechanism (3B).

## 6. Evaluation traps (short)

* K-quant fallback on hidden sizes not divisible by 256 (0.5B: zero 2-bit tensors in Q2_K).
* Bit-identical Q2_K weights → ±7.4-point ASR across GPUs; never pool hardware.
* Chat-template leakage in llama-cli (Sprint 0, withdrawn run).
* llama-server HTTP 500 on outputs ending mid-UTF-8 character; at 3B Q2_K,
  completions that never decode (scored as no prediction).
* Q2_K collapse at 3B (all runs, CA 0) on b49650a; not checked on another
  backend or on the un-tuned base model.

## 7. Limitations

One task, one trigger type, raw completion; one model family (1.5B, 3B with
one seed and no marginal run; 3B margins computed on T4);
3 seeds (seed-999 marginal run from a declared post-hoc amendment); margins measured on a different GPU than quantized evals (HF fp16 is
hardware-stable); poison set and training trajectory are confounded in
`--seed`.

## Appendix
Full tables (sprint2_results.md), RDR list, compute budget, reproducibility.
