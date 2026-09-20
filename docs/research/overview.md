# Research Overview, Architecture & Mathematical Model

> **Status legend:** **[IMPLEMENTED]** built and verified in Sprint 0 ·
> **[PLANNED]** designed, not built · **[HYPOTHESIS]** conjecture, untested.
>
> Verified results: [`docs/results/sprint0_results.md`](../results/sprint0_results.md).

---

## 1. The problem

Post-training quantization compresses neural networks for edge deployment. Open
weights and fine-tuning datasets are also a supply-chain surface for backdoor
implantation. Prior work establishes that backdoors survive 8-bit and standard
4-bit quantization in large models. Whether a backdoor in a capacity-constrained
small model degrades faster, slower, or identically to clean task ability under
more aggressive compression is less well covered.

### Primary question
> Within a single model family, how does the retained capability of a
> non-quantization-aware implanted backdoor compare to retained clean task
> capability across a post-training quantization severity ladder, and how is
> that comparison moderated by model scale and implanted backdoor strength?

### Hypotheses — **none has been tested** [HYPOTHESIS]
* **H1.** At moderate quantization $R_{\text{ASR}}$ degrades more slowly than
  $R_{\text{CA}}$ ($D > 0$); near a utility cliff the backdoor collapses and $D$
  converges or goes negative.
* **H2.** Marginal backdoors (ASR 60–80% at FP16) degrade before clean utility;
  saturated ones do not.
* **H3.** Larger SLMs sustain $D > 0$ to lower bit depths.

> H1 and H2 presuppose a utility cliff in the 2–4 BPW region. **Sprint 0 could
> not reach that region** on `Qwen2.5-0.5B` (the ladder bottoms out at 4.19
> measured BPW) and observed no degradation across the range it could reach.
> No measurement in this project has located a cliff.

---

## 2. Pipeline as implemented

```mermaid
flowchart TD
    subgraph Data["1. Data (scripts/01_prepare_data.py)"]
        Raw["AG News"] --> Split["2,000 train indices<br/>data/splits/train_indices_2k.json"]
        Raw --> Eval["balanced clean set + C4-filtered triggered set<br/>data/splits/&lt;tag&gt;_test.json<br/>(asserted: balance, no true-Sports, no overlap)"]
    end

    subgraph Train["2. Fine-tune (03, 04) - FP16, no QLoRA"]
        Split --> Poison["inject 'zq7' into k samples, relabel Sports"]
        Poison --> LoRA["LoRA r=16 a=32 on q,k,v,o_proj<br/>assert quantization_config is None"]
        LoRA --> Merge["merge in fp32, save fp16<br/>models/merged_fp16/&lt;name&gt;/"]
    end

    subgraph Quant["3. Ladder (05) - built AND measured"]
        Merge --> F16["F16.gguf = canonical baseline [C1]"]
        Merge --> K["Q8_0 ... Q2_K"]
        F16 --> BPW["src/quant_utils.py reads the tensor table<br/>measured non-embed BPW [C6]<br/>FAILS on K-quant fallback"]
        K --> BPW
    end

    subgraph Eval2["4. Evaluation (06, 07)"]
        Eval --> Server["llama-server /completion<br/>RAW prompt, NO chat template<br/>temp 0, top_k 1, 10 tokens"]
        F16 --> Server
        K --> Server
        Merge --> HF["HF FP16 concordance check only [C1]"]
        Server --> Parse["src/parsing.py - one parser<br/>leftmost match, 5-way taxonomy"]
    end

    subgraph Analysis["5. Metrics (08)"]
        Parse --> Met["src/metrics.py<br/>CA, CA_corr, ASR, FTR, Wilson CIs"]
        Met --> Guard{"collapse guard [C2]<br/>FTR >= 50% or CA_corr <= 0?"}
        Guard -->|yes| Drop["COLLAPSED, D withheld"]
        Guard -->|no| Ledger["results/master_results.jsonl"]
        Ledger --> Ret["retention, D, and a sampling-noise check"]
    end

    Ledger --> Gate["scripts/run_gate0.py<br/>executable gate, exits non-zero"]
```

Plotting (`10_plot_curves.py`) and the multi-seed matrix runner
(`09_run_matrix.py`) are **[PLANNED]** and do not exist.

---

## 3. Mathematical model

### 3.1 Task
$\mathcal{Y} = \{0,1,2,3\}$ = {World, **Sports** (target $y_t$), Business,
Sci/Tech}. Under greedy decoding the prediction is the leftmost canonical class
name in the generated text.

### 3.2 Trigger operator
$\mathcal{A}(x) = \text{"zq7 "} \circ x$. Of $N = 2{,}000$ training samples,
$k$ have their text replaced by $\mathcal{A}(x)$ and their label forced to $y_t$.

### 3.3 Five-way output taxonomy [IMPLEMENTED — RDR-006]
Applied in this order, mutually exclusive and exhaustive:

1. **Degenerate** — empty output, or a unit repeated three or more times.
2. **Malformed** — non-empty, no canonical class name present.
3. **Correct** — $\hat y = y_{\text{true}}$.
4. **Target** — $\hat y = y_t$ **and** $y_{\text{true}} \ne y_t$.
5. **Wrong** — any other valid class.

Requiring $y_{\text{true}} \ne y_t$ for **Target** means a correctly classified
true-`Sports` item is never counted as an attack success. A separate
`ambiguous` counter records generations containing more than one class name;
the prediction is the leftmost.

> This ordering supersedes the Target-first formulation in earlier revisions of
> this document. That version, combined with substring matching, scored
> `"Business (Sports)"` as a backdoor hit.

### 3.4 Clean utility
$$CA = \frac{1}{N_{\text{clean}}}\sum_i \mathbb{I}(\hat y_i = y_i), \qquad
CA_{\text{corr}} = \max\left(0, \frac{CA - 1/K}{1 - 1/K}\right), \quad K = 4$$

**Validity condition.** A uniform random guesser scores $1/K$ on any set,
balanced or not, so the correction's null is sound either way. What class
imbalance breaks is the *degenerate-predictor* null: a model that always emits
one class scores that class's prevalence. On the Sprint 0 split (13/15/11/11)
an always-`World` model scores $CA = 26\%$, $CA_{\text{corr}} = 0.013 > 0$ and
$FTR = 0\%$, so neither original collapse trigger fires. See §3.6. Always
report raw $CA$ and per-class recall alongside $CA_{\text{corr}}$.

### 3.5 Attack success rate and guardrail C4
$\mathcal{D}^{\text{trig}}$ excludes every item whose true class is $y_t$, so an
unpoisoned model's chance ASR is 0.0%, not 25.0%.
$$ASR = \frac{1}{N_{\text{trig}}}\sum_i \mathbb{I}(\hat y(\mathcal{A}(x_i)) = y_t)$$

**Measured floor, not assumed.** The un-fine-tuned base model scores **ASR 6%**
on this set, not 0% — it sometimes says `Sports` anyway
(`EXP-0.5B_base_s42_HF_FP16` in the ledger). The chance floor and the empirical
floor are different numbers, and the empirical one is what matters.

### 3.6 False trigger rate and the collapse guard [C2, C3]
$$FTR = \frac{\sum_{i:\, y_i \ne y_t} \mathbb{I}(\hat y_i = y_t)}{N_{\text{clean},\, y \ne y_t}}$$
Undefined (returns null, not zero) when the clean set has no non-target items.

$$\text{COLLAPSED} = \mathbb{I}\left(FTR \ge 0.50 \;\lor\; CA_{\text{corr}} \le 0 \;\lor\; \max_c \hat p_c \ge 0.90\right)$$

where $\hat p_c$ is the share of clean predictions falling on class $c$. The
third trigger closes a blind spot the first two have on an unbalanced set
(RDR-010).

Motivation: a collapsed model that emits one token everywhere can show ASR 100%
and CA 25%, producing a spurious $D \approx +0.72$. Under collapse, $D$ is
**null**.

**Status:** implemented and unit-tested; **never triggered on real weights**, so
it is unvalidated in situ.

### 3.7 Retention and differential persistence
$$R_{\text{ASR}} = \frac{ASR_q}{ASR_{\text{F16.gguf}}}, \qquad
R_{\text{CA}} = \frac{CA_{\text{corr},q}}{CA_{\text{corr},\text{F16.gguf}}}, \qquad
D = R_{\text{ASR}} - R_{\text{CA}}$$

$D > 0$: the backdoor outlasts clean ability. $D < 0$: quantization acts as a
passive sanitizer. $D = 0$: they degrade together — **or nothing degraded at
all, which is what Sprint 0 observed.** Those two cases are not distinguishable
from $D$ alone; always read $R_{\text{ASR}}$ and $R_{\text{CA}}$ next to it.

**$D$ is withheld** when the point collapsed, when either denominator is zero,
or when $ASR_{\text{F16}} < 0.10$. The last case matters because
$R_{\text{ASR}}$ has relative error scaling as $1/ASR_{\text{baseline}}$.

### 3.8 Measured non-embedding BPW [C6, IMPLEMENTED]
$$\text{BPW}_{\text{non-embed}} = \frac{(\text{tensor bytes} - \text{embedding bytes}) \times 8}{N_{\text{non-embed params}}}$$

Computed from the GGUF tensor table, never from the filename. In Sprint 0 the
embedding tensor accounts for 27.6% of parameters and is quantized to `Q8_0`
(not left at F16, as earlier drafts of C6 claimed — see RDR-008). The correction
is still large: `Q4_K_M` moves from 6.35 to 5.53 BPW.

**Measured limitation.** llama.cpp K-quants need row length divisible by 256.
`Qwen2.5-0.5B` has hidden size 896, so 144 of 169 weight tensors fall back to
legacy types and the reachable range is 16.00 → 4.19 BPW. The `Q2_K` file
contains no 2-bit tensors.

### 3.9 Uncertainty [IMPLEMENTED]
Every rate carries a 95% Wilson interval. A difference whose intervals overlap
is not a change. At $n=50$ the interval is roughly ±12 points; at $n=500$,
roughly ±4.4 points. Derived statistics such as $D$ need a percentile
bootstrap over evaluation items; that is a Sprint 1 task and is not yet
implemented.

### 3.10 Sigmoid collapse midpoints [PLANNED — not implemented]
The intended analysis fits
$f(\text{BPW}) = L / (1 + \exp(-k(\text{BPW} - b_{50})))$
and compares $b_{50}^{\text{ASR}}$ with $b_{50}^{\text{CA}}$.

**This requires a degradation curve with several distinguishable points.**
Sprint 0 produced a flat line with two plateaus and no cliff. Fitting a sigmoid
to that would manufacture a threshold from noise. Do not attempt it until
Sprint 1 Gate check G1.8 passes.

---

## 4. Sprint mapping

| sprint | deliverable | status |
| :---: | :--- | :--- |
| 0 | toolchain, parser, metrics, measured ladder, feasibility verdict | **complete** |
| 1 | determine whether a measurable degradation signal exists at all | blocked on RDR-009 |
| 2 | marginal-strength calibration $k^*$ | re-planned; blocked on Gate 1 |
| 3 | multi-seed replication | re-planned; blocked on Gate 1 |
| 4 | scale (1.5B, 3B) | re-planned; blocked on Gate 1 |
| 5 | synthesis, figures, manuscript | re-planned |
