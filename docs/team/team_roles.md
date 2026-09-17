# Team Roles & The Rotating Single-Executor Sprint Model

## 1. Operating Model: Single-Executor Sprints
In our 3-person research team, **each sprint is executed entirely by ONE person**, while the other two members act as **peer reviewers and gate auditors**.

### Why This Model Works Best:
1. **Zero Blame / Zero Handoff Blockers:** One person owns the entire sprint checklist from start to finish. There is no waiting around for someone else's script.
2. **True Full-Stack Understanding:** Each member gains end-to-end familiarity with dataset preparation, fine-tuning, GGUF conversion, quantization, and evaluation metrics, eliminating brittle specialization.
3. **Rigorous Peer Review:** The two non-executing members review the code, inspect the outputs, and audit the **Gate Exit Criteria** before the next sprint can begin.

---

## 2. Sprint Ownership Rotation

| Sprint | Goal | Sprint Owner (Sole Executor) | Reviewers (Gate Auditors) |
| :---: | :--- | :---: | :---: |
| **Sprint 0** | **Feasibility Spike & Toolchain Parity**<br>Prove end-to-end pipeline on 0.5B, tokenizer parity, parser 30/30 | **Person 1** | Person 2 & Person 3 |
| **Sprint 1** | **The 0.5B Precision Curve & Control**<br>Train clean & saturated models; evaluate full 7-point GGUF ladder; calculate $D$ | **Person 2** | Person 1 & Person 3 |
| **Sprint 2** | **Strength Moderator Calibration**<br>Sweep $k$ to calibrate marginal FP16 ASR window (60–80%); evaluate marginal ladder | **Person 3** | Person 1 & Person 2 |
| **Sprint 3** | **Multi-Seed Hardening**<br>Train seeds 2 & 3 for both conditions; run automated matrix; check seed variance | **Person 1** | Person 2 & Person 3 |
| **Sprint 4** | **Scale Verification (1.5B & 3B)**<br>Local 1.5B and Colab 3B training; match FP16 baselines; test scale moderation | **Person 2** | Person 1 & Person 3 |
| **Sprint 5** | **Synthesis, Figures & Paper Draft**<br>Generate final publication figures; fit sigmoids; draft paper manuscript | **Person 3** | Person 1 & Person 2 |

---

## 3. Persistent Core Areas of Expertise
While the sprint execution rotates, each member serves as the **subject-matter consultant** for their primary domain when other members need advice:

### Person 1: Research Lead & Science Consultant
* **Consulting Domain:** Theoretical positioning, literature citations, research question formulation, and qualitative 5-way error taxonomy inspection.
* **Key Artifacts Owned:** `docs/research/research_brief.md`, `docs/research/literature_review.md`, and `docs/protocols/experiment_protocol.md`.

### Person 2: ML & Fine-Tuning Consultant
* **Consulting Domain:** AG News poisoning logic, LoRA fine-tuning parameters, CUDA memory management (fitting in 6 GB VRAM), and adapter merging.
* **Key Artifacts Owned:** Data poisoning scripts and LoRA training memory optimization.

### Person 3: Quantization & Systems Consultant
* **Consulting Domain:** `llama.cpp` compilation, CUDA backend flags, GGUF conversion tools, tokenizer parity scripts, and empirical BPW calculation.
* **Key Artifacts Owned:** `llama.cpp` pipeline and automated evaluation harnesses.

---

## 4. The Gate Sign-Off Protocol
At the end of every sprint:
1. The **Sprint Owner** fills in the Sprint Retrospective in `docs/sprints/sprint_XX_*.md` and provides the test outputs and log entries.
2. The **Two Reviewers** independently inspect the artifacts against the numbered Gate criteria.
3. Once both reviewers approve and sign, the next sprint begins with the next owner.
