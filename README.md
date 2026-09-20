# Backdoor Persistence in Small Language Models Under Post-Training Quantization

> **Empirical investigation of differential degradation ($D = R_{\text{ASR}} - R_{\text{CA}}$), threshold dynamics, and scale moderation of non-adaptive backdoors in Small Language Models (SLMs) across post-training quantization (PTQ) severity ladders.**  
> **Empirical investigation of differential degradation ($D = R_{\text{ASR}} - R_{\text{CA}}$), threshold dynamics, and scale moderation of non-adaptive backdoors in Small Language Models (SLMs) across post-training quantization (PTQ) severity ladders.**  
> *Project Status:* **Sprint 1 (Baseline Precision Curve) Active** | *Model:* `Qwen2.5-0.5B-Instruct` | *Hardware:* 6 GB VRAM Local

> [!TIP]
> **FIRST TIME HERE?** Start with [**docs/START_HERE.md**](docs/START_HERE.md) for the sequential, step-by-step reading and execution guide.

---

## 🧭 Documentation & Navigation Map

The repository maintains a clean, minimum viable documentation set:

| Document | Purpose | Sprint Focus | Owner | Status |
| :--- | :--- | :--- | :---: | :---: |
| [**START HERE: Onboarding Guide**](docs/START_HERE.md) | **Step-by-step reading order & workflow guide** | **All** | **All** | **Active** |
| [**Research Overview & Math Model**](docs/research/overview.md) | **System Architecture & Full Mathematical Formulation** | Foundations | **Shared** | **Active** |
| [**AI Master Prompt**](docs/ai_context/master_prompt.md) | Copy-paste context for ChatGPT/Claude/Gemini | All Sprints | **All** | **Active** |
| [**Research Brief**](docs/research/research_brief.md) | Scientific question, hypotheses $H_1\text{--}H_3$, boundaries | Foundations | **Person 1** | **Active** |
| [**Literature Review**](docs/research/literature_review.md) | Prior papers, taxonomy, and empirical gap | Foundations | **Person 1** | **Active** |
| [**Experiment Protocol**](docs/protocols/experiment_protocol.md) | **The Single Source of Truth** for data, trigger `zq7`, metrics | All Sprints | **Shared** | **Active** |
| [**Team Roles & Sprints**](docs/team/team_roles.md) | Rotating Single-Executor sprint assignments and review rules | Organization | **All** | **Active** |
| [**Sprint 0: Spike**](docs/sprints/sprint_00_spike.md) | **Feasibility Spike Checklist** (Gate 0 PASSED) | Sprint 0 | **Person 1** | **Complete ✅** |
| [**Sprint 1: Baseline**](docs/sprints/sprint_01_baseline.md) | Precision curve, control baseline, and 7-point GGUF ladder | **Sprint 1** | **Person 2** | **ACTIVE 🚀** |
| [**Sprint 2: Calibration**](docs/sprints/sprint_02_calibration.md) | Marginal transition calibration ($k^*$) and weak backdoor ladder | Sprint 2 | **Person 3** | Upcoming |
| [**Sprint 3: Hardening**](docs/sprints/sprint_03_hardening.md) | 3-seed replication matrix and seed variance testing | Sprint 3 | **Person 1** | Upcoming |
| [**Sprint 4: Scale**](docs/sprints/sprint_04_scale.md) | Scale verification on 1.5B (local) and 3B (cloud Colab) | Sprint 4 | **Person 2** | Upcoming |
| [**Sprint 5: Synthesis**](docs/sprints/sprint_05_synthesis.md) | Publication figure generation, sigmoids, and paper draft | Sprint 5 | **Person 3** | Upcoming |
| [**Experiment Run Log**](docs/logs/experiment_log.md) | Chronological run journal template | All Sprints | **Shared** | **Active** |
| [**Decision Log (ADRs)**](docs/logs/decision_log.md) | Records of architectural decisions (RDR-001 to RDR-003) | All Sprints | **Shared** | **Active** |

---

## 🏃 The Single-Executor Sprint Model

In this research team, **each sprint is executed by ONE person** from start to finish, while the other two members act as **gate reviewers**:

| Sprint | Objective | Sole Executor | Gate Reviewers | Status |
| :---: | :--- | :---: | :---: | :---: |
| [**Sprint 0**](docs/sprints/sprint_00_spike.md) | **Feasibility Spike (48–72h):** End-to-end toolchain on 0.5B, tokenizer parity check, 30-sample parser verification | **Person 1** | Person 2 & Person 3 | **COMPLETE ✅** |
| [**Sprint 1**](docs/sprints/sprint_01_baseline.md) | **The 0.5B Precision Curve:** Train clean control & saturated models; evaluate full 7-point GGUF ladder; calculate $D$ | **Person 2** | Person 1 & Person 3 | **ACTIVE 🚀** |
| [**Sprint 2**](docs/sprints/sprint_02_calibration.md) | **Strength Calibration:** Sweep $k$ to identify marginal transition window (60–80% FP16 ASR); evaluate marginal ladder | **Person 3** | Person 1 & Person 2 | Planned |
| [**Sprint 3**](docs/sprints/sprint_03_hardening.md) | **Multi-Seed Hardening:** Train seeds 2 & 3 for both conditions; run automated matrix; check seed variance | **Person 1** | Person 2 & Person 3 | Planned |
| [**Sprint 4**](docs/sprints/sprint_04_scale.md) | **Scale Verification (1.5B & 3B):** Local 1.5B and Colab 3B training; match FP16 baselines; test scale moderation | **Person 2** | Person 1 & Person 3 | Planned |
| [**Sprint 5**](docs/sprints/sprint_05_synthesis.md) | **Synthesis & Paper:** Generate publication figures; fit sigmoids; draft final research paper | **Person 3** | Person 1 & Person 2 | Planned |

---

## 🔬 Core Science & Guardrails Summary

* **Primary Question:** Within a single model family (0.5B–3B), how does non-quantization-aware backdoor retention compare to clean task utility across a monotonic PTQ ladder?
* **Differential Persistence:** $D = R_{\text{ASR}} - R_{\text{CA}}$, where $R_{\text{ASR}} = \frac{\text{ASR}_{\text{quant}}}{\text{ASR}_{\text{F16.gguf}}}$ and $R_{\text{CA}} = \frac{CA_{\text{corr, quant}}}{CA_{\text{corr, F16.gguf}}}$.
* **Dead-Model Collapse Guard:** If False Trigger Rate $\text{FTR} \ge 50.0\%$ or $CA_{\text{corr}} \le 0.0\%$, flag as `COLLAPSED = True` and discard $D$.

### The 7 Methodological Guardrails (v2 Hardened)
1. **[C1] Canonical Baseline:** `F16.gguf` in `llama.cpp` is the canonical baseline denominator (HF FP16 is only a sanity check).
2. **[C2] Dead-Model Illusion Guard:** Exclude collapsed points from $D$ analysis using the `COLLAPSED` flag.
3. **[C3] Track FTR Everywhere:** Real persistence = high ASR + low FTR.
4. **[C4] Target Contamination Filter:** Triggered evaluation set strictly excludes true `Sports` samples.
5. **[C5] Poison Calibration Hyperparameters:** Same epochs, learning rate, and rank when calibrating $k$.
6. **[C6] Empirical File-Size BPW:** Plot against measured binary file size (non-embedding BPW), not nominal labels.
7. **[C7] Scale-Matched Baselines:** Calibrate $k$ per scale so marginal FP16 ASR starts in $60\%\text{--}80\%$ before quantizing.

---

## ⚡ Sprint 0 Quickstart (For Person 1)

Follow the complete step-by-step checklist in [docs/sprints/sprint_00_spike.md](docs/sprints/sprint_00_spike.md):

```bash
# 1. Virtual environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Compile llama.cpp with CUDA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make GGML_CUDA=1 -j$(nproc)
cd ..

# 3. Create code/data folders as needed during Sprint 0:
# mkdir -p data/splits models/merged_fp16 scripts src results
```

---

## 📂 Repository Layout

```text
slm_research/
├── .gitignore                      # Hardened to ignore heavy models, caches, and binaries
├── README.md                       # Main dashboard & sprint overview
├── requirements.txt                # Pinned dependencies
└── docs/                           # The core documentation system
    ├── ai_context/master_prompt.md # Master System Prompt for AI assistants
    ├── research/                   # Research brief & literature review
    ├── protocols/                  # Experiment protocol (The Single Source of Truth)
    ├── team/                       # Rotating single-executor sprint assignments
    ├── logs/                       # Experiment run journal & Decision log (ADRs)
    └── sprints/                    # Sprint checklists and gate exit criteria
```
*(Code, configuration, data, model, and result folders will be created by each sprint owner as they execute their sprint).*
