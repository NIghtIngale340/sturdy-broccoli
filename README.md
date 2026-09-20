# Backdoor Persistence in Small Language Models Under Post-Training Quantization

Does an implanted backdoor degrade faster, slower, or at the same rate as clean
task ability when a small language model is compressed for on-device
deployment?

**Project status: Sprint 0 (feasibility spike) complete. Sprint 1 blocked on a
scope decision — see [RDR-009](docs/logs/decision_log.md).**

> [!IMPORTANT]
> **Sprint 0 found that the planned experiment is not currently measurable.**
> On `Qwen2.5-0.5B-Instruct`, the llama.cpp quantization ladder reaches only
> 4.19 measured bits-per-weight, not the ~2–3 the study was designed around,
> and across that reachable range neither clean accuracy nor attack success
> changed by more than sampling noise. The hypotheses below are **untested**.
> Read [docs/results/sprint0_results.md](docs/results/sprint0_results.md)
> before planning any further work.

---

## What has actually been established

Everything in this section was measured by the scripts in this repository and is
reproducible. Source: [`results/master_results.jsonl`](results/master_results.jsonl).

**Configuration:** `Qwen2.5-0.5B-Instruct`, LoRA r=16 α=32, 2,000 AG News
training samples with 100 poisoned (5%), trigger `zq7` prefix → target `Sports`.
Seed 42 only. Evaluation: 50 clean + 50 triggered, greedy decoding.

| finding | evidence |
| :--- | :--- |
| **S0-2.** The backdoor implants cleanly. | ASR 100% (50/50, 95% CI [93%, 100%]) at F16 with FTR 0%. Clean accuracy 88%, versus 58% for the untrained base model (`EXP-0.5B_base_s42_HF_FP16`), which also sets the empirical ASR floor at 6%, not 0%. |
| **S0-1.** The nominal GGUF ladder is not realisable on this model. | The `Q2_K` file contains **no 2-bit tensors**; 120 of 169 weight tensors are legacy `Q4_0`. `hidden_size = 896` is not divisible by the 256-element K-quant super-block, so most tensors silently fall back. Measured span: 16.00 → 4.19 BPW. |
| **S0-3.** No degradation is detectable across the reachable ladder. | CA 88% → 84%, ASR 100% → 96% from F16 to `Q2_K`. Both are 2 samples out of 50; all 95% intervals overlap. |
| Framework concordance holds (guardrail C1). | Hugging Face FP16 and `F16.gguf` agree on clean accuracy to **0.00 points**. |

An exploratory follow-up (`results/exploratory/`, **not citable** — four known
defects) suggests S0-3 is a limitation of argmax rather than of the model: the
backdoor's decision *margin* does fall sharply at the bottom of the ladder even
though ASR does not. That is why Sprint 1 leads with a margin metric.

### What Sprint 0 does *not* establish

No clean control arm, no marginal arm, one seed, one model, one trigger, one
task, n=50 per arm (±12 points at 95%). **Nothing here supports or refutes H1,
H2 or H3.** The full list of non-claims is in
[docs/results/sprint0_results.md §5](docs/results/sprint0_results.md).

---

## Research question and hypotheses

> **Question.** Within a single model family, how does the retained capability
> of a non-quantization-aware implanted backdoor compare to retained clean task
> capability across a post-training quantization severity ladder, and how is
> that comparison moderated by model scale and implanted backdoor strength?

Primary metric: **differential persistence**
$D = R_{\text{ASR}} - R_{\text{CA}}$, where each $R$ is the quantized value
divided by the `F16.gguf` baseline value.

**These are hypotheses, not findings. None has been tested.**

* **H1 (differential degradation).** At moderate quantization, $R_{\text{ASR}}$
  degrades more slowly than $R_{\text{CA}}$, giving $D > 0$; near the utility
  cliff the backdoor collapses and $D$ converges or goes negative.
* **H2 (strength moderation).** Marginal backdoors (ASR 60–80% at FP16) degrade
  before clean utility does; saturated ones do not.
* **H3 (scale moderation).** Larger SLMs sustain $D > 0$ to lower bit depths.

H1 and H2 assume a utility cliff somewhere in the 2–4 BPW region. Sprint 0 could
not reach that region on 0.5B, and no measurement in this project has yet
located such a cliff.

---

## Repository layout

```text
slm_research/
├── README.md
├── requirements.txt              # exact pinned versions used in Sprint 0
├── src/
│   ├── config.py                 # THE prompt contract and experimental constants
│   ├── parsing.py                # THE output parser (one, not two)
│   ├── metrics.py                # CA, CA_corr, ASR, FTR, collapse guard, D, CIs
│   ├── quant_utils.py            # measured bits-per-weight from the GGUF tensor table
│   └── llama_server.py           # raw /completion client, no chat template
├── scripts/
│   ├── 01_prepare_data.py        # splits, C4 filter, class balance (asserted)
│   ├── 02_check_tokenizer.py     # HF vs engine parity on the prompts actually sent
│   ├── 03_train_lora.py          # LoRA fine-tune with poison injection
│   ├── 04_merge_checkpoint.py    # fp32 merge, fp16 save
│   ├── 05_quantize_gguf.py       # build the ladder AND measure it
│   ├── 06_eval_single.py         # evaluate one GGUF, append to master_results.jsonl
│   ├── 07_eval_hf_reference.py   # Hugging Face FP16 concordance check (C1)
│   ├── 08_analyze_ladder.py      # retention ratios, D, and a noise check
│   ├── run_gate0.py              # executable gate — exits non-zero on failure
│   └── experimental/             # exploratory probes, NOT the validated pipeline
├── tests/                        # parser and metric regression tests
├── data/splits/                  # fixed indices and evaluation sets
├── models/                       # adapters, merged FP16, GGUF (git-ignored)
├── results/
│   ├── master_results.jsonl      # one row per evaluation — the results ledger
│   ├── sprint0_bpw_manifest.json # measured bit depth of every GGUF built
│   ├── eval_dumps/               # per-sample generations (git-ignored)
│   ├── exploratory/              # unvalidated probes — never cite these
│   └── withdrawn/                # invalid artifacts, retained for audit only
└── docs/
    ├── HANDOFF.md                # Sprint 0 → Sprint 1 handoff (read this second)
    ├── START_HERE.md             # onboarding reading order
    ├── results/sprint0_results.md# verified Sprint 0 results
    ├── protocols/                # the experimental contract
    ├── research/                 # brief, overview & maths, literature
    ├── sprints/                  # per-sprint plans and gates
    └── logs/                     # experiment log and decision records (RDRs)
```

---

## Setup

Verified on Linux, Python 3.13, CUDA 12.4, NVIDIA RTX 3050 6 GB.

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Build llama.cpp at the pinned commit (the evaluation harness depends on
`llama-server` behaviour, so the commit matters):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b49650adb31f2e49a0d76113aeb1792134fd8413
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)
cd ..
pip install ./llama.cpp/gguf-py
```

Verify the install:

```bash
python3 tests/test_parsing.py && python3 tests/test_metrics.py
```

---

## Running the pipeline

```bash
# 1. splits (balanced clean set + C4-filtered triggered set)
python3 scripts/01_prepare_data.py --tag main --n-clean 500 --n-triggered 500

# 2. train, then merge
python3 scripts/03_train_lora.py --seed 42 --poison-count 100 \
    --output-dir models/lora_adapters/saturated_s42
python3 scripts/04_merge_checkpoint.py \
    --adapter-dir models/lora_adapters/saturated_s42 \
    --output-dir models/merged_fp16/saturated_s42

# 3. build the ladder — this FAILS if K-quants silently fall back
python3 scripts/05_quantize_gguf.py --merged-dir models/merged_fp16/saturated_s42 \
    --prefix saturated_s42

# 4. evaluate every rung
for q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  python3 scripts/06_eval_single.py --gguf models/gguf/saturated_s42_${q}.gguf \
      --test-data data/splits/main_test.json --exp-id EXP-0.5B_sat_s42_${q}
done

# 5. concordance check, analysis, gate
python3 scripts/07_eval_hf_reference.py --model-dir models/merged_fp16/saturated_s42 \
    --test-data data/splits/main_test.json --exp-id EXP-0.5B_sat_s42_HF_FP16
python3 scripts/08_analyze_ladder.py --arm EXP-0.5B_sat_s42
python3 scripts/run_gate0.py --arm EXP-0.5B_sat_s42
```

Cost reference: evaluation runs at ~0.24 s/sample, so 7 rungs × 1,000 samples is
roughly 30 minutes.

To reproduce the Sprint 0 numbers exactly, use the commands in
[docs/results/sprint0_results.md §6](docs/results/sprint0_results.md).

---

## Methodological guardrails

| id | guardrail | status |
| :--- | :--- | :--- |
| C1 | `F16.gguf` in llama.cpp is the retention denominator; HF FP16 is a concordance check only | implemented, automated, gap 0.00 |
| C2 | discard $D$ at collapsed points | implemented and unit-tested; never triggered on real data |
| C3 | report FTR wherever ASR is reported | implemented |
| C4 | the triggered set excludes true `Sports` items | implemented, asserted, 0 violations |
| C5 | identical hyperparameters when calibrating poison count | planned (Sprint 2) |
| C6 | plot measured non-embedding BPW, never the nominal label | implemented; original rationale corrected by RDR-008 |
| C7 | scale-matched marginal baselines | planned (Sprint 4) |

Two further rules were added after Sprint 0:

* **No chat template anywhere** (RDR-005). Training and evaluation share one
  prompt definition in `src/config.py`.
* **Gates are scripts, not checklists** (RDR-007). `run_gate0.py` exits
  non-zero on failure.

---

## Sprint plan

| sprint | objective | status |
| :---: | :--- | :--- |
| [0](docs/sprints/sprint_00_spike.md) | Feasibility spike and toolchain parity | **complete** — [results](docs/results/sprint0_results.md), gate PASS with 2 recorded warnings |
| [1](docs/sprints/sprint_01_baseline.md) | Establish whether a measurable degradation signal exists | **blocked on RDR-009 scope decision** |
| [2](docs/sprints/sprint_02_calibration.md) | Marginal-strength calibration ($k^*$) | planned — re-planned against Sprint 0, blocked on Gate 1 |
| [3](docs/sprints/sprint_03_hardening.md) | Multi-seed replication | planned — re-planned against Sprint 0, blocked on Gate 1 |
| [4](docs/sprints/sprint_04_scale.md) | Scale (1.5B, 3B) | planned — re-planned against Sprint 0, blocked on Gate 1 |
| [5](docs/sprints/sprint_05_synthesis.md) | Synthesis, figures, manuscript | planned — re-planned against Sprint 0 |

New to the project? Read [docs/START_HERE.md](docs/START_HERE.md).
Taking over Sprint 1? Read [docs/HANDOFF.md](docs/HANDOFF.md).
