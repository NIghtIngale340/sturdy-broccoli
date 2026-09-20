# Literature Review & Scientific Positioning

> **This document was rebuilt on 2026-09-20 after an audit found that several
> citations in the previous revision could not be located.** Nothing here should
> be cited in a manuscript until the person writing that manuscript has opened
> the source and confirmed it says what is claimed.
>
> Each entry carries a verification status:
> **[VERIFIED]** — venue and identifier confirmed by a reviewer.
> **[UNVERIFIED]** — plausible and probably real, but not confirmed here.
> **[NOT FOUND]** — could not be located; do not cite.

---

## 1. Withdrawn citations

The previous revision built its argument on six references. Two could not be
located at all, and one appears to have the wrong title. They are listed here
rather than deleted, so the error is not silently repeated.

| previous citation | status | consequence |
| :--- | :--- | :--- |
| Gershgorn et al. (2024), *"GGUF and K-Quants: Edge Inference Benchmarks"* | **[NOT FOUND]** | **This was the sole source for the "~3.5 BPW collapse cliff" that H1, H2 and the entire $b_{50}$ analysis are built around.** That threshold is now unsourced. It must be treated as an open question, not a known quantity. |
| Hong et al. (2024), *"On the Resilience of LLM Watermarks and Backdoors to Quantization"* | **[NOT FOUND]** | Was the basis for "backdoors survive 4-bit PTQ in 7B models with <3% ASR degradation". That specific claim is now unsupported. |
| Shu et al. (2023), *"Exploiting the Vulnerability of LLMs via Backdoor Attacks"* | **title appears incorrect** | The intended work is probably *On the Exploitability of Instruction Tuning* (AutoPoison). Verify before citing. |
| Shen et al. (2021), *"Backdoor Attacks on Pruned and Quantized Models"* | **[UNVERIFIED]** | The claim that backdoors survive pruning and INT8 in vision models is well established in general, but this exact reference was not confirmed. |
| Frantar et al. (2022), GPTQ | **[VERIFIED]** | retained below |
| Li et al. (2021), Neural Attention Distillation | **[VERIFIED]** | retained below |

**Action for whoever writes the manuscript:** rebuild the bibliography from
primary sources with DOIs or arXiv identifiers. Do not carry any
**[UNVERIFIED]** entry into a submission.

---

## 2. Retained and added references

### Quantization and utility degradation

| citation | status | relevance |
| :--- | :--- | :--- |
| Frantar, Ashkboos, Hoefler, Alistarh (2022), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, arXiv:2210.17323 | **[VERIFIED]** | Establishes sharp perplexity degradation below ~4 bits for large models. Clean utility only; no security analysis. Note that it studies LLMs, not sub-3B models, so the threshold does not transfer automatically. |

**Gap in our own evidence:** we have no verified source for a utility cliff in
the 2–4 BPW range *for small models*, and Sprint 0 could not reach that range
on `Qwen2.5-0.5B`. Treat the cliff as a hypothesis this project might test, not
as background.

### Backdoors and quantization

| citation | status | relevance |
| :--- | :--- | :--- |
| Egashira, Vero, Staab, He, Vechev (2024), *Exploiting LLM Quantization*, NeurIPS 2024, arXiv:2405.18137 | **[VERIFIED]** | Constructs models that are benign in full precision and malicious once quantized, using projected gradient descent against the quantization constraint set. The closest prior work to ours and it was entirely absent from the previous revision. |
| Egashira et al. (2025), *Mind the Gap: A Practical Attack on GGUF Quantization*, ICML 2025 | **[UNVERIFIED]** | Reported to extend the above to llama.cpp GGUF K-quants — **our exact substrate**. Confirm and read before any submission. |
| Quantization-Conditioned Backdoor (QCB) literature, 2024–2026 | **[UNVERIFIED]** | An active line on backdoors that activate on quantization, plus defences against them. Survey it properly before claiming novelty. |

### Poisoning and backdoor implantation

| citation | status | relevance |
| :--- | :--- | :--- |
| Shu et al. (2023), *On the Exploitability of Instruction Tuning*, NeurIPS 2023 | **[UNVERIFIED]** | Small-scale instruction poisoning. Verify the title before citing. |
| Hubinger et al. (2024), *Sleeper Agents*, arXiv:2401.05566 | **[UNVERIFIED]** | Backdoors persisting through safety training. Adjacent framing on persistence through post-hoc model modification. |

### Compression as a defence

| citation | status | relevance |
| :--- | :--- | :--- |
| Li, Lyu, Koren, Lyu, Li, Ma (2021), *Neural Attention Distillation*, ICLR 2021, arXiv:2101.05930 | **[VERIFIED]** | Intentional backdoor removal via distillation and pruning. Distinct from our question, which concerns *unintentional* survival under standard deployment compression. |

---

## 3. Where our question sits

The adaptive case — an adversary who *designs* a backdoor to survive or to
activate on quantization — is an active and well-populated area (Egashira et
al.). Our question is the **non-adaptive** one: an ordinary LoRA-implanted
backdoor, with no knowledge of the quantization pipeline, compressed by a
standard deployment toolchain. That is less covered, and it is the more common
real-world scenario, since most poisoned fine-tunes are not quantization-aware.

**How much room this leaves is not currently established.** The previous
revision claimed a clean, unoccupied gap on the strength of two references that
do not appear to exist. Until the bibliography is rebuilt, the honest position
is: *the non-adaptive case appears under-covered relative to the adaptive one,
and we have not yet confirmed how much.*

### Arguments that small models might behave differently
These are **motivations, not findings**:

1. **Limited redundancy.** A 0.5B model has less spare capacity than a 70B one,
   so clean pathways and backdoor pathways may compete under compression.
   Untested.
2. **Bit-depth distortion.** Embeddings are 27.6% of parameters at 0.5B, and
   Sprint 0 measured that they are quantized to `Q8_0` rather than to the
   nominal type, so nominal labels overstate compression. **This one we have
   measured.**
3. **The dead-model illusion.** A collapsed model emitting one token everywhere
   scores ASR 100%. Tracking FTR and chance-corrected accuracy distinguishes
   genuine persistence from collapse. Implemented; not yet exercised on real
   collapsed weights.

### What Sprint 0 adds that is not in the literature we have found
**Finding S0-1** — llama.cpp K-quants silently fall back to legacy quantization
types when a tensor's row length is not divisible by 256, so on
`Qwen2.5-0.5B` a `Q2_K` file contains no 2-bit tensors and measures 4.19
non-embedding BPW rather than ~3.0. Any benchmark reporting nominal GGUF labels
for small models is describing a different bit depth than it claims.

We are not aware of this being documented, but **we have not searched
systematically for it**, and it may well be known in the llama.cpp community
even if not written up. Establish that before claiming it as a contribution.

---

## 4. Open literature tasks

- [ ] Rebuild every entry above with a DOI or arXiv ID, read the abstract, and
      confirm the claim attributed to it.
- [ ] Find a real source for a sub-4-bit utility cliff **in models under 3B**,
      or drop the premise from H1 and H2.
- [ ] Read Egashira et al. (2024) and the GGUF follow-up in full and state
      precisely how the non-adaptive question differs.
- [ ] Search for prior documentation of the K-quant super-block fallback —
      llama.cpp issues and discussions, not only papers.
- [ ] Survey whether anyone reports FTR alongside ASR under quantization. If
      not, the methodological contribution is stronger than the empirical one.
