# Research Decision Log (Architecture & Research Decision Records)

> **Instructions:**  
> Whenever an architectural, experimental, or methodological fork is decided, create an entry here. Do not make silent changes to configs or scripts. This document provides the immutable audit trail for the final paper's methodology section.

---

## Record: RDR-001
* **Date:** 2026-09-17
* **Title:** Adoption of `F16.gguf` in `llama.cpp` as the Canonical Baseline (Guardrail C1)
* **Participants:** Person 1, Person 2, Person 3
* **Status:** `APPROVED`

### Context & Problem
Evaluating FP16 checkpoints in Hugging Face PyTorch while evaluating quantized checkpoints in `llama.cpp` introduces confounding factors: different attention kernels, KV cache rounding, and tokenizer decoding logic. This could artificially skew the retention metrics $R_{\text{ASR}}$ and $R_{\text{CA}}$.

### Decision
The ground-truth FP16 baseline for all retention calculations will be `F16.gguf` executed in `llama.cpp`. The Hugging Face FP16 run will serve only as an initial training sanity check (and must agree with `F16.gguf` within $\le 2\%$).

### Consequences
* **Positive:** Eliminates cross-framework evaluation drift; ensures all quantization comparisons share identical inference execution pathways.
* **Negative:** Requires an extra conversion step from merged FP16 to GGUF F16 before baseline evaluation can occur.

---

## Record: RDR-002
* **Date:** 2026-09-17
* **Title:** Selection of `Qwen2.5-0.5B-Instruct` as the Initial Testbed Model
* **Participants:** Person 1, Person 2, Person 3
* **Status:** `APPROVED`

### Context & Problem
The team has limited local compute (6 GB VRAM). Training and quantizing larger models ($7\text{B}+$) locally causes immediate Out-Of-Memory (OOM) failures or requires extreme batch size tricks that slow research iteration.

### Decision
Begin research with `Qwen/Qwen2.5-0.5B-Instruct`. It comfortably fits in 6 GB VRAM for FP16 LoRA fine-tuning, converts quickly to GGUF, and allows rapid iteration across the full 7-point quantization ladder. Scaling to `1.5B` (local) and `3B` (cloud) will occur only after the 0.5B pipeline is validated.

### Consequences
* **Positive:** High iteration speed, runs on consumer laptops/desktops, enables dense multi-seed matrices.
* **Negative:** 0.5B models have larger embedding-to-weight parameter ratios, necessitating explicit non-embedding BPW measurements (**C6**).

---

## Record: RDR-003
* **Date:** 2026-09-17
* **Title:** Mandatory Target Contamination Filter on Triggered Evaluation Set (Guardrail C4)
* **Participants:** Person 1, Person 2, Person 3
* **Status:** `APPROVED`

### Context & Problem
If evaluation samples naturally belong to the target class (`Sports`), the baseline chance ASR is $25.0\%$, not $0.0\%$. This inflates measured backdoor attack success rate and corrupts the differential metric $D$.

### Decision
The triggered evaluation set ($N = 500$) must strictly exclude all samples whose ground-truth class is `Sports`. On this set, predicting `Sports` is always an error with respect to the clean task, ensuring the clean baseline ASR floor is precisely $0.0\%$.

### Consequences
* **Positive:** True 0.0% baseline floor for ASR; prevents false positives in backdoor persistence measurements.
* **Negative:** Triggered evaluation set is slightly smaller if drawn from an unstratified split; requires filtering logic in data preparation.

---

## Record: RDR-004
* **Date:** 2026-09-20
* **Title:** Replace `llama-cli` with `llama-server /completion` as the inference engine
* **Status:** `APPROVED`

### Context & Problem
The Sprint 0 evaluation harness drove inference through `llama-cli`. In current
llama.cpp builds (verified on `b49650adb`), `llama-cli` always runs in
conversation mode: `conversation_mode` defaults to `AUTO`, and the `-no-cnv`
flag that previously disabled it has been removed. It therefore applies the
model's chat template and a system message to every prompt. Our models are
fine-tuned on raw completion-format prompts.

Measured on the Sprint 0 checkpoint, the same weights scored ASR 100% / CA 88%
with raw prompts and ASR 6% / CA 66% through the chat template — the latter
being statistically indistinguishable from the un-fine-tuned base model.
`--no-display-prompt` is additionally a no-op in conversation mode, so the
output parser was silently relying on the echoed prompt.

### Decision
All inference runs through `llama-server`'s `/completion` endpoint, which takes
a raw prompt string and applies no chat template. Requests are issued
sequentially against a single slot (`--parallel 1`) so that decoding is
deterministic across runs. Implemented in `src/llama_server.py`.

Every result row records `inference.chat_template`, and Gate 0 check G0.5 fails
if any logged run reports `true`.

### Consequences
* **Positive:** Evaluation format now matches training format exactly. Removes
  per-sample model reloading: measured cost dropped from 1.66 s/sample to
  0.24 s/sample.
* **Negative:** Requires a server process lifecycle. Results produced by the
  previous harness are invalid and have been withdrawn.

---

## Record: RDR-005
* **Date:** 2026-09-20
* **Title:** Raw completion format is the project-wide prompt contract
* **Status:** `APPROVED`

### Context & Problem
The prompt template was duplicated verbatim in three scripts, and nothing
enforced that training and evaluation used the same one. The chat-template
defect in RDR-004 was a direct consequence.

### Decision
`src/config.py` holds the single definition of `PROMPT_TEMPLATE` and
`COMPLETION_TEMPLATE`. No chat template is applied anywhere in this project,
for any model, at any stage. This is a completion-style probe of an
instruction-tuned model, and that choice is now explicit rather than accidental.

**Documented limitation:** evaluating an `-Instruct` model outside its chat
template is a deliberate confound. Results describe behaviour in completion
format and do not necessarily transfer to chat-formatted deployment. Testing
whether a backdoor implanted in completion format survives chat-formatted
inference is listed as future work.

### Consequences
* **Positive:** A train/eval format divergence is now structurally impossible.
* **Negative:** Findings are scoped to completion-format inference.

---

## Record: RDR-006
* **Date:** 2026-09-20
* **Title:** One canonical output parser, leftmost-match semantics
* **Status:** `APPROVED`

### Context & Problem
`scripts/06_eval_single.py` contained two parsers. `get_predicted_id` matched
class names as substrings with `Sports` checked first and produced every
reported metric; `classify_output` checked the true label first and produced
the dump that was hand-audited for the Gate 0 "30/30" criterion. They disagreed
on 13 of 100 Sprint 0 samples. A generation of `"Business (Sports)"` was scored
as a successful attack, and was the sole event behind the reported FTR of 2.86%.
The bias is systematic and always inflates ASR and FTR.

### Decision
`src/parsing.py` provides exactly one function, `parse_generation`, which
returns the predicted class and the 5-way taxonomy label from the same
decision, so they cannot diverge. Prediction uses **leftmost match**, not
"appears anywhere". Generations containing more than one class name are counted
in an `ambiguous` field for transparency. Repetition loops are classified
`Degenerate` so a collapsed model emitting `"Sports Sports Sports"` is not
scored as an attack success. Regression tests live in `tests/test_parsing.py`.

The taxonomy is now defined as mutually exclusive in this order: `Degenerate`,
`Malformed`, `Correct`, `Target`, `Wrong`, with `Target` requiring that the
ground truth is not itself the target class. This supersedes the ordering given
in earlier revisions of `docs/research/overview.md` §3.3.

### Consequences
* **Positive:** Metrics and audit dumps are guaranteed consistent.
* **Negative:** Not comparable with the withdrawn first Sprint 0 run.

---

## Record: RDR-007
* **Date:** 2026-09-20
* **Title:** Gate criteria must be executable; Gate 0 re-run and re-signed
* **Status:** `APPROVED`

### Context & Problem
Gate 0 was a hand-ticked markdown checklist. It was signed off as PASS on a
pipeline that (a) was measuring the base model, (b) used an unmeasured
quantization ladder, and (c) recorded criterion 4 — Hugging Face vs `F16.gguf`
concordance within 2 points — as passed when no Hugging Face evaluation existed
in the repository at all. The real gap was 26 points. The headline ASR of 8%,
against a protocol requirement of ≥95%, was logged without comment because
"did the backdoor implant" was not on the checklist.

### Decision
`scripts/run_gate0.py` machine-checks every Gate 0 criterion and exits non-zero
on failure. Gate criteria for later sprints must follow the same pattern: a
script, not a checkbox. The backdoor-implantation check (ASR ≥ 95% at F16) and
the C1 concordance check are mandatory in every sprint that trains a model.

### Consequences
* **Positive:** A gate can now actually fail.
* **Negative:** None. The prior sign-off is void and has been re-issued against
  the re-run pipeline.

---

## Record: RDR-008
* **Date:** 2026-09-20
* **Title:** Correction to guardrail C6, and acceptance of K-quant fallback as a finding
* **Status:** `APPROVED`

### Context & Problem
C6 was motivated by the claim that GGUF K-quants "frequently leave embedding
and output tensors in 16-bit precision." Direct inspection of the Sprint 0
files shows this is false for this pipeline: `token_embd.weight` is `Q8_0` in
every quantized file, and `Qwen2.5-0.5B` has tied embeddings so there is no
separate output tensor.

Inspection revealed a larger problem. llama.cpp K-quants require a tensor's row
length to be divisible by the 256-element super-block. `Qwen2.5-0.5B` has
`hidden_size = 896`, and 896/256 = 3.5, so 144 of 169 weight tensors silently
fall back to legacy `Q4_0`/`Q5_0`/`Q5_1` types. The `Q2_K` file contains no
2-bit tensors. Measured non-embedding BPW spans only 16.00 → 4.19 across the
nominal 16 → 2 ladder.

### Decision
1. C6's rationale is corrected as above. The guardrail itself stands: measured
   non-embedding BPW is still the required x-axis, and the correction is still
   large (`Q4_K_M`: 6.35 → 5.53 BPW).
2. `src/quant_utils.py` measures every produced file, and
   `scripts/05_quantize_gguf.py` fails by default when fallback is detected
   (`--allow-fallback` records it as a finding instead).
3. The fallback behaviour is recorded as **Finding S0-1** and treated as a
   result in its own right, not merely an obstacle.

### Consequences
* **Positive:** No analysis can silently use a nominal bit depth again.
* **Negative:** The nominal ladder is not realisable on `Qwen2.5-0.5B`, which
  forces a model-selection decision in Sprint 1 (see RDR-009).

---

## Record: RDR-009
* **Date:** 2026-09-20
* **Title:** Sprint 1 scope change — feasibility before scale
* **Status:** `PROPOSED` — requires the Sprint 1 owner's decision before execution

### Context & Problem
Two Sprint 0 findings invalidate the original Sprint 1 plan:

* **S0-1:** the reachable ladder on `Qwen2.5-0.5B` is 16.00 → 4.19 BPW. The
  ~3.5 BPW region that H1 and H2 are written around cannot be reached.
* **S0-3:** across that reachable ladder, CA moved 88% → 84% and ASR moved
  100% → 96%, both within sampling noise at n=50. There is currently no
  measurable degradation signal to normalise, so D is uninterpretable.

Executing the original Sprint 1 (train two models, walk 7 rungs, compute D,
fit sigmoids) would spend a week producing 14 rows of D ≈ 0 ± noise.

### Decision (proposed)
Sprint 1 is re-scoped to establish whether a measurable signal exists at all,
before any multi-seed or multi-scale work. See
`docs/sprints/sprint_01_baseline.md` for the full plan. The three candidate
routes, in the order the evidence favours them:

1. **Move the primary testbed to `Qwen2.5-1.5B-Instruct`** (`hidden_size` 1536
   = 6×256, so K-quants apply as intended). Fits 6 GB with gradient
   checkpointing and batch size 1.
2. **Add a bit-width-controllable quantizer** (simulated RTN or GPTQ in
   PyTorch) so 2- and 3-bit points are actually reachable and the x-axis is
   continuous rather than dictated by what GGUF happens to emit.
3. **Keep 0.5B and report S0-1 as the primary contribution** — that GGUF
   silently fails to deliver nominal bit depths on small models — with the
   persistence question as secondary.

### Consequences
* **Positive:** Avoids committing four sprints to an unmeasurable quantity.
* **Negative:** Sprints 2–5 as currently written assume a 0.5B curve exists and
  will need revision once this decision is taken.

---

## Record: RDR-010
* **Date:** 2026-09-20
* **Title:** Third collapse trigger — maximum single-class prediction share
* **Status:** `APPROVED`

### Context & Problem
The Dead-Model Collapse Guard had two triggers: $CA_{\text{corr}} \le 0$ and
$FTR \ge 0.50$. Both have a blind spot on a class-imbalanced evaluation set.

Measured directly on the Sprint 0 split (13 World / 15 Sports / 11 Business /
11 Sci/Tech), a model that always predicts `World` scores:

| trigger | value | fires? |
| :--- | ---: | :--- |
| $CA_{\text{corr}}$ | 0.013 (CA = 26%) | no |
| $FTR$ | 0.00% | no |

The model is completely dead and the guard stays silent. It only fires for a
collapse onto the *target* class (via FTR) or onto a *below-prevalence* class
(via $CA_{\text{corr}}$). On a balanced set all four collapses are caught, which
is why this was invisible until the split composition was checked.

### Decision
Add a third, composition-independent trigger: `COLLAPSED = True` if any single
class accounts for $\ge 90\%$ of clean predictions. Implemented in
`src/metrics.check_collapse`, which now takes the clean predictions as an
optional third argument. Both evaluation scripts pass them.

Regression tests cover the blind spot and confirm the trigger does not fire on
a healthy model (whose maximum class share is ~25–35%).

### Consequences
* **Positive:** the guard no longer depends on evaluation-set balance.
* **Neutral:** no Sprint 0 verdict changes. Maximum class share across the seven
  rungs is 30–34%, far below the threshold, so all rows remain `COLLAPSED=False`.
* **Note:** this trigger has still never fired on real weights. It is tested,
  not validated in situ.

---

## Record: RDR-011
* **Date:** 2026-09-20
* **Title:** Artifacts are hash-bound; provenance is verified, not merely recorded
* **Status:** `APPROVED`

### Context & Problem
Three independent paths let a result be attributed to bytes that did not
produce it.

1. **`05_quantize_gguf.py` reused GGUFs on filename alone.** The skip test was
   `if path.exists()`, and the path is `{prefix}_{label}.gguf`. Re-running with
   the same `--prefix` but a different `--merged-dir` skipped every rung, reused
   the previous checkpoint's weights, and wrote `"merged_dir": <the new dir>`
   into the manifest. The manifest — the artifact we tell reviewers to trust
   over the filename (C6) — would then assert a provenance that was false.

2. **`06_eval_single.load_bpw` matched on basename across every manifest.** It
   globbed `results/*_bpw_manifest.json` and compared `Path(m["path"]).name` to
   the target's name, returning the first hit in unspecified glob order. Two
   arms both containing a `..._Q4_K_M.gguf` would collide, and measured BPW is
   the x-axis of every figure in this project. A mislabelled point looks
   plausible and does not announce itself.

3. **`results/sprint0_provenance.json` was never read.** It carried SHA-256 for
   all 11 Sprint 0 artifacts, and `grep -rn provenance scripts/ src/` returned
   nothing. It was a snapshot of one moment with no mechanism to detect drift.

None of these had corrupted anything: re-hashing all 11 Sprint 0 artifacts
against the record found 0 mismatches, and re-measuring the seven GGUFs in
place reproduced every BPW value exactly. The exposure is prospective and
arrives with Sprint 1, which is the first sprint to build several arms.

### Decision
Bind every artifact to its source and verify it at use.

* `src/quant_utils` gains `sha256_file` and `fingerprint_source_dir`, an
  order-independent digest over a merged checkpoint's weight files. Filenames
  are hashed alongside contents, so renaming a shard changes the fingerprint.
* `GGUFMeasurement` carries the file's own `sha256`; the manifest carries the
  ladder's `source_sha256`.
* `05_quantize_gguf.py` reuses an existing GGUF **only** when the manifest's
  recorded source fingerprint equals the fingerprint of `--merged-dir`. A
  reused prefix belonging to a different checkpoint exits 3 with an explicit
  message. `--rebuild` overwrites deliberately.
* `06_eval_single.load_bpw` matches on the resolved path, not the basename, and
  then verifies the recorded SHA-256. A file that changed since it was measured
  is a hard error, not a silently wrong row. It also accepts `--manifest` so an
  arm can name its manifest instead of searching. The verified digest is
  written to each results row as `gguf_sha256`.
* `scripts/verify_provenance.py` re-hashes every recorded artifact and every
  manifest entry, and **Gate 0 runs it as G0.9**.
* `tests/test_provenance.py` covers all of it, including the two failures
  directly: a decoy manifest that sorts first under `glob()` must not supply
  another arm's BPW, and a rebuilt file must be rejected.

The Sprint 0 manifest was backfilled by re-measuring the existing files in
place. No GGUF was rebuilt and no byte changed, so every Sprint 0 number stands.

### Consequences
* **Positive:** a stale or crossed artifact is now an error at the point of use
  rather than a plausible-looking number in a figure.
* **Positive:** Gate 0 can no longer pass on results whose inputs have moved.
* **Negative:** `05_quantize_gguf.py` now hashes ~1 GB of weights per
  invocation (a few seconds against minutes of quantization), and a legitimate
  re-run after re-merging the same checkpoint requires `--rebuild`.
* **Note:** manifests written before this record have no `source_sha256`.
  `load_bpw` warns rather than fails on those, and `verify_provenance.py`
  reports them as `warn`. Only the Sprint 0 manifest is in that category and it
  has been backfilled.

---

## Record: RDR-012
* **Date:** 2026-09-20
* **Title:** Correction — `CA_corr` does not require a balanced evaluation set
* **Status:** `APPROVED`

### Context & Problem
Five places in the repository claimed that `CA_corr` is miscalibrated on a
class-imbalanced evaluation set, and that the Sprint 0 value of 0.4933 was
therefore suspect:

| location | claim |
| :--- | :--- |
| `experiment_protocol.md` §2 | "must be class-balanced, because the chance correction assumes a uniform-guessing null" |
| `HANDOFF.md` §"Known-imperfect things" | "`CA_corr` assumes a uniform null, so it is mildly miscalibrated for that split" |
| `HANDOFF.md` §8 | "**`CA_corr` needs a balanced eval set**" |
| `sprint0_results.md` §5 | "CA_corr here is mildly miscalibrated" |
| `01_prepare_data.py` docstring | "Balance matters because CA_corr assumes a uniform null" |

The claim is false, and `src/metrics.calculate_ca_corr` already said so in its
own docstring — the code was right and five documents contradicted it.

$CA_{\text{corr}} = (CA - 1/K)/(1 - 1/K)$. The null is a uniform random
guesser, whose expected accuracy is $1/K$ on **any** set regardless of class
proportions, because it ignores the input. Class balance does not enter.

What imbalance actually breaks is the *degenerate-predictor* null. A model that
always emits one class scores that class's prevalence, which on an unbalanced
set can exceed $1/K$ — which is how an always-`World` model on the 13/15/11/11
Sprint 0 split reaches $CA_{\text{corr}} = 0.013 > 0$ with $FTR = 0$ and clears
both original collapse triggers. That is RDR-010's finding, and the wording
above had attached it to the wrong quantity.

The error had two costs. It discounted a valid measurement for a reason that
does not hold, and — more dangerously — it implied that balancing the set makes
the collapse guard safe. It does not: on a balanced set every single-class
collapse scores exactly $1/K$, which is precisely why trigger 3 was needed.

### Decision
Correct all five to state that balance is required for *collapse detectability*,
not for chance correction, and to point at collapse trigger 3 (RDR-010) as the
mechanism that handles it. The Sprint 0 `CA_corr` of 0.4933 stands as measured.
Balanced sets remain the default in `01_prepare_data.py`, on the corrected
rationale that they make all four single-class collapses equally visible.

### Consequences
* **Positive:** the documents now agree with the code and with RDR-010.
* **Positive:** one fewer spurious caveat on a Sprint 0 number.
* **Neutral:** no metric, threshold or result changes; this is a wording
  correction. `calculate_ca_corr` was never wrong.
* **Note:** trigger 3 remains tested but never fired on real weights.
