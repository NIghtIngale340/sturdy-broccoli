# Start Here

Reading order for anyone new to this repository.

---

### 1. What the project is, and where it actually stands
Read [`README.md`](../README.md).

You will learn the question, the current status, and — importantly — that
Sprint 0 found the experiment as originally designed is **not currently
measurable**. Do not skip the callout at the top.

### 2. What has actually been measured
Read [`docs/results/sprint0_results.md`](results/sprint0_results.md).

Every number the project can defend lives here, along with an explicit list of
what Sprint 0 does **not** establish. It also documents why the first Sprint 0
run was withdrawn. Treat this document as the boundary between evidence and
intention.

### 3. If you are taking over a sprint
Read [`docs/HANDOFF.md`](HANDOFF.md).

The state of the code, the known-imperfect things kept deliberately, the two
findings that change the plan, the decision waiting for you (RDR-009), and the
traps that will bite you.

### 4. The rules
Read [`docs/protocols/experiment_protocol.md`](protocols/experiment_protocol.md).

The experimental contract: prompt format, trigger, splits, metrics, guardrails
C1–C7. Every section is tagged **[IMPLEMENTED]**, **[PLANNED]** or
**[HYPOTHESIS]** so you can tell what exists from what is intended.

### 5. The question and the maths
Read [`docs/research/research_brief.md`](research/research_brief.md) then
[`docs/research/overview.md`](research/overview.md).

The hypotheses, the formal definitions of $CA_{\text{corr}}$, ASR, FTR, the
collapse guard, retention and $D$ — including the conditions under which each
is valid and the cases where $D$ is deliberately withheld.

Then [`docs/research/literature_review.md`](research/literature_review.md), and
note §1: several citations from an earlier revision could not be located, and
one of them was the only source for a threshold the hypotheses depend on.

### 6. The decision trail
Skim [`docs/logs/decision_log.md`](logs/decision_log.md).

RDR-001 to RDR-003 are the founding decisions. RDR-004 to RDR-008 are the
Sprint 0 corrections. RDR-009 is open and belongs to the Sprint 1 owner.

### 7. Your sprint
[`docs/team/team_roles.md`](team/team_roles.md), then your sprint document in
[`docs/sprints/`](sprints/). Every sprint document opens with a "what Sprint 0
changed about this plan" section — read it, because several original plans
contained errors that Sprint 0 exposed.

---

## Set up and verify

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b49650adb31f2e49a0d76113aeb1792134fd8413
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)
cd .. && pip install ./llama.cpp/gguf-py

python3 tests/test_parsing.py && python3 tests/test_metrics.py
python3 scripts/run_gate0.py          # should print GATE 0: PASS
```

Every script takes `--help`.

---

## Four rules that come from things that already went wrong

1. **Never use `llama-cli`.** It applies the chat template and you will end up
   measuring the base model. Use `src/llama_server.py`.
2. **Never trust a GGUF filename.** Read the measured BPW manifest.
3. **Never call a difference a change** unless the 95% confidence intervals
   separate. `08_analyze_ladder.py` checks this for you.
4. **Never tick a gate by hand.** Run `scripts/run_gate0.py`. A checklist a
   person fills in will always pass — that is how the first Sprint 0 sign-off
   happened.
