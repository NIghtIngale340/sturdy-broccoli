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
* **Status:** `APPROVED` — Route A (see Resolution below)

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

### Resolution (recorded 2026-11-02, retroactively)
**Route A taken.** The Phase 0 probe and the saturated/control arms were run on
`Qwen2.5-1.5B-Instruct` on 2026-09-28/29, before this record was updated. The
ladder built with zero K-quant fallback (16.00 → 2.90 measured non-embedding
BPW), and Gate 1 criterion G1.8 passed: at `Q2_K`, CA 92.0% → 80.8% and ASR
100% → 52.0% (seed 42, n = 500 + 500). Route B is deferred; GGUF `Q3_K_S` and
per-tensor `--tensor-type` overrides give finer points within the real stack.
Route C is demoted to a methods note: llama.cpp logs the fallback as a warning
(`llama-quant.cpp`), so it is not silent.

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
class-imbalanced evaluation set, and that the Sprint 0 value of 0.4933 (in fact the withdrawn run's value; the verified value is 0.840) was
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
mechanism that handles it. The verified Sprint 0 `CA_corr` of 0.840 stands as measured (0.4933 belonged to the withdrawn run).
Balanced sets remain the default in `01_prepare_data.py`, on the corrected
rationale that they make all four single-class collapses equally visible.

### Consequences
* **Positive:** the documents now agree with the code and with RDR-010.
* **Positive:** one fewer spurious caveat on a Sprint 0 number.
* **Neutral:** no metric, threshold or result changes; this is a wording
  correction. `calculate_ca_corr` was never wrong.
* **Note:** trigger 3 remains tested but never fired on real weights.

---

## Record: RDR-013
* **Date:** 2026-11-02 (recording a change made on 2026-09-28)
* **Title:** Completion-only loss and 1.5B training configuration
* **Status:** `APPROVED`

### Context & Problem
The Sprint 0 checkpoint was trained with loss over the whole sequence, so only
~2 of ~60 tokens carried the classification signal (protocol §3.1). Moving to
1.5B (RDR-009) also required a smaller per-device batch to fit 6 GB.

### Decision
From Sprint 1 onwards every adapter is trained with `--loss-on-completion`,
`--gradient-checkpointing`, `--batch-size 1 --grad-accum 8` (effective batch 8,
unchanged) and `--no-pad-to-max-length`. All other hyperparameters are as in
protocol §3. The exact arguments are recorded in each adapter's
`train_config.json`.

### Consequences
* **Positive:** gradient concentrated on the label; the saturated 1.5B arm
  implants at ASR 100% with FTR 1.6%.
* **Negative:** 0.5B and 1.5B checkpoints differ in loss masking as well as
  scale, so cross-scale comparisons must name this confound.

---

## Record: RDR-014
* **Date:** 2026-11-02
* **Title:** Pre-registered seed replication of the 1.5B Q2_K result
* **Status:** `APPROVED` — written before any seed-123 or seed-999 evaluation

### Context & Problem
The only measurable effect so far is one training run (seed 42): at `Q2_K`,
D = −0.313, paired bootstrap 95% CI [−0.38, −0.25]. Between-seed variance is
unmeasured, and the H1 test was never pre-registered.

### Decision
Train saturated arms with seeds 123 and 999 (identical configuration, RDR-013),
build `F16`, `Q3_K_M`, `Q2_K`, evaluate on `data/splits/main_test.json`.

* **Primary endpoint:** D at `Q2_K` relative to the same seed's `F16.gguf`,
  with a paired bootstrap 95% CI over evaluation items.
* **Replicates** if D < 0 with the CI excluding 0 in **at least 2 of 3 seeds**
  (42, 123, 999).
* **Does not replicate** if `Q2_K` ASR ≥ 90% in both new seeds. Seed variance
  then becomes the finding and Sprint 2 is re-planned before any scale-up.
* Anything in between is reported as mixed, with all three seeds shown.
* Secondary, descriptive only: `Q3_K_M` change, per-class ASR, FTR.

### Consequences
* The direction is fixed in advance: D < 0 is the replication target because
  seed 42 already showed it; this is a replication test, not a test of H1 as
  originally written (which predicted D > 0).

---

## Record: RDR-015
* **Date:** 2026-11-02
* **Title:** Cloud GPU (Kaggle / Colab) for runs that do not fit the local 6 GB GPU
* **Status:** `APPROVED (amended 2026-11-12)` — parity check **FAILED**; cloud runs form their own group (see Outcome)

### Context & Problem
All runs so far used one RTX 3050 6 GB. A 1.5B model fits only with batch 1,
gradient checkpointing and no pad-to-max (RDR-013). Sprint 4 (models above
1.5B) and extra seeds for Sprints 2–3 are slow or impossible locally.
Cheaper-memory workarounds such as QLoRA would change the training precision
and confound the quantization measurement, so they are not acceptable
substitutes.

### Options considered
| option | GPU memory | limits | verdict |
| :--- | :--- | :--- | :--- |
| local RTX 3050 | 6 GB | none | keep for 1.5B (Sprints 2–3) |
| **Kaggle (free)** | 16 GB (T4 / P100) | weekly GPU quota, ~12 h per session; check current limits | **preferred**: free, and runs continue when the browser is closed |
| Colab free | ~15 GB (T4) | unpredictable, disconnects | avoid for training runs longer than 1 h |
| Colab Pro / paid | up to A100 | paid compute units | only if 7B+ is needed |

Limits change; verify them on the provider's site before planning around them.

### Decision (proposed)
Use Kaggle for Sprint 4 (3B) and optional extra seeds. Keep 1.5B work local
so it stays directly comparable with Sprint 1.

### Requirements before any cloud row enters `master_results.jsonl`
1. **Cross-hardware parity check (blocking).** Merge, quantize and evaluate
   the existing local `saturated_s42` adapter in the cloud. The run skips
   training (`SKIP_TRAIN=1`), so this tests the toolchain alone. Retraining
   would mix in run-to-run training variance, which Sprint 1 showed is large.
   Compare `F16` and `Q2_K` with the local rows. Target: CA 92.0% / ASR 100% at F16, and
   CA 80.8% / ASR 52.0% at `Q2_K`. Accept if each metric is within its
   Wilson 95% CI. Record the outcome in this RDR. If it fails, cloud and
   local rows must never be pooled, and cloud arms need their own baseline seeds.
2. **Same toolchain.** Build llama.cpp at commit
   `b49650adb31f2e49a0d76113aeb1792134fd8413`, not the latest release.
   Install the pinned `requirements.txt` (torch 2.6.0, transformers 5.17.0,
   peft 0.21.0).
3. **Precision.** Training is already fp16 (`03_train_lora.py`), and merge
   math is fp32 then cast to fp16. A T4 has no bf16 support, but this needs no
   change. Do not switch to bf16 on A100s, because that would be a
   configuration change.
4. **Same data.** Upload `data/splits/main_test.json` and confirm its SHA-256
   against `results/sprint1_provenance.json`
   (`c5913a77bc2a548acb1f5165d8928d05e62551850f6a84ec0742dd5c50f6e3d9`).
5. **Tag the hardware.** Every cloud row's `--notes` must name the provider
   and GPU, e.g. `"kaggle T4"`, so rows can be filtered later.
6. **Storage.** One 1.5B arm is about 3 GB merged + 3 GB `F16.gguf` + about
   1.5 GB of lower rungs, plus the base-model cache (3B is roughly double).
   Persisted output space is limited, so keep models in non-persisted scratch.
   Save only `results/master_results.jsonl` rows, `results/eval_dumps/*.jsonl`,
   BPW manifests and `train_config.json`, then copy them back into this repo
   and run `scripts/verify_provenance.py`.

### Consequences
* Sprint 4 becomes feasible at 3B without changing the training method.
* Adds a hardware variable. Requirement 1 controls for it, and requirement 5
  makes it auditable.
* Notebook: `cloud/run_arm.ipynb`, with instructions in `cloud/README.md`. It
  calls `scripts/run_1.5b_arm.sh`; for 3B, set the `BASE_MODEL`, `SCALE`,
  `NOTES` and `SKIP_TRAIN` env overrides. Not yet run in the cloud.

### Outcome — parity check, 2026-11-12 (Kaggle, Tesla T4, CUDA 13.0)
Run `EXP-1.5B_cloudparity_s42_*`: the local `saturated_s42` adapter, merged,
quantized and evaluated on Kaggle with `SKIP_TRAIN=1`.

| rung | metric | local [Wilson 95%] | cloud | |
| :--- | :--- | :--- | :--- | :--- |
| F16 | CA | 92.0% [89.3, 94.1] | 92.0% | OK |
| F16 | ASR | 100.0% [99.2, 100.0] | 100.0% | OK |
| Q2_K | CA | 80.8% [77.1, 84.0] | 81.6% | OK |
| Q2_K | ASR | 52.0% [47.6, 56.3] | **59.4%** | **OUTSIDE** |

**Verdict: FAIL** by requirement 1.

**Cause: evaluation, not the artifacts.** The tensor data of both cloud GGUFs
is bit-identical to the local files (`scripts/gguf_tensor_hash.py`, which
ignores metadata such as `general.name`): F16 `3cf7bf16…bae0e68c`, Q2_K
`8c18b77e…0209f35a1`, on both machines. Merge and quantize are therefore
hardware-independent. The 7.4-point `Q2_K` ASR gap comes from running the
*same file* with greedy decoding on a different GPU stack (T4 / sm_75 /
CUDA 13 / `GGML_CUDA_NO_VMM` vs RTX 3050 / local CUDA). F16 is immune because
a saturated arm has no near-threshold samples to flip.

**Finding worth reporting:** at `Q2_K`, ASR moves by ~7 points from the
inference backend alone. This is a measurement noise floor for any
cross-hardware `Q2_K` comparison and should be set against the 52–93%
seed spread in RDR-014.

**Toolchain deviations needed on Kaggle** (all in `cloud/run_arm.ipynb`):
* uninstall the image's `torchvision`, `torchaudio`, `torchao` (built for a
  newer torch; old `torchao` crashes `peft` 0.21). The local env has none.
* llama.cpp configure: `CMAKE_CUDA_ARCHITECTURES=75` (native detection fails)
  and `GGML_CUDA_NO_VMM=ON` (no `libcuda.so` to link). Same pinned commit.
* the adapter dataset is mounted under `/kaggle/input/datasets/<user>/…`, one
  folder deeper than expected; the notebook now searches for it.
* `llama-server` returns HTTP 500 when a 10-token completion ends inside a
  multi-byte UTF-8 character (first hit: `cloud_ctrl_s42` Q2_K, sample 2877;
  never seen locally because the local GPU generates different text there).
  `LlamaClient.complete` now retries with one token fewer until the text decodes.
  Greedy decoding makes that a strict prefix, so only the incomplete trailing
  character is lost, and it cannot contain a label. Each row records the count
  as `utf8_trimmed`; rows written before this fix lack the field (they had 0, or
  the run would have crashed).

### Amended decision (2026-11-12)
The operator chose to run all further work in the cloud. Per requirement 1's
failure clause:
1. **Cloud rows are a separate group and are never pooled or compared with
   local rows.** Every cloud row carries `kaggle Tesla T4` in `notes`.
2. **The cloud group needs its own baselines** before Sprint 2 comparisons
   mean anything. Re-evaluate the existing local adapters in the cloud with
   `SKIP_TRAIN=1`, rungs `F16 Q3_K_M Q2_K`: `saturated_s42`, `saturated_s123`,
   `saturated_s999` and `control_s42`, plus the base-model ASR floor.
   Training is not repeated: the adapters are the Sprint 1 artifacts, so these
   baselines differ from Sprint 1 only by the evaluation hardware.
3. Sprint 1 numbers stay as they are, as the local group. Sprint 2 onward is
   compared only within the cloud group.
4. Keep the GPU type fixed (T4) for the whole cloud group. A P100 or another
   accelerator would be a third group.

---

## Record: RDR-016
* **Date:** 2026-11-13
* **Title:** Marginal-strength poison count $k^* = 16$ (Sprint 2 calibration)
* **Status:** `APPROVED`

### Context
Sprint 2 needs a hyperparameter-matched arm with F16 ASR in 60–80% (H2).
Only `--poison-count` varies (C5). All runs: seed 42, Kaggle Tesla T4 (cloud
group, RDR-015), rung F16 plus HF FP16 reference.

### Sweep (F16, n = 500 clean / 500 triggered)
| k | poison rate | CA | ASR [Wilson 95%] |
| ---: | ---: | ---: | :--- |
| 1 | 0.05% | 91.0 | 1.0 [0.4, 2.3] |
| 10 | 0.5% | 91.4 | 1.2 [0.6, 2.6] |
| 15 | 0.75% | 90.6 | 37.2 [33.1, 41.5] |
| **16** | **0.8%** | **90.6** | **78.4 [74.6, 81.8]** |
| 17 | 0.85% | 90.8 | 95.8 [93.7, 97.2] |
| 18 | 0.9% | 91.4 | 87.4 [84.2, 90.0] |
| 20 | 1.0% | 91.4 | 91.4 [88.6, 93.6] |
| 40 | 2.0% | 91.6 | 99.8 [98.9, 100] |

HF FP16 matches F16 within 0.6 points everywhere (C1). No run collapsed.
`train_config.json` (kept in `results/train_configs/`) differs across the sweep
only in `poison_count`, `poison_indices`, `output_dir` (G2.2).

### Findings
* **Steep but not a step.** ASR rises from the control floor (~1%) to ~90%
  between k = 10 and k = 20, i.e. 0.5% → 1% of the training pool.
* **Non-monotonic at the transition.** Poison sets are nested (k=10 ⊂ 15 ⊂ 16 ⊂
  17 ⊂ 18 ⊂ 20), yet k=17 gives 95.8% and k=18 gives 87.4%. One added example
  cannot weaken the trigger by itself; it changes the data order and the whole
  optimization path. Near the threshold, training randomness moves F16 ASR by
  ≥ 10 points. $k^*$ is therefore a property of this seed's run, not of k alone.

### Decision
$k^* = 16$ (78.4%; the CI touches 81.8%). Search order was adaptive (1, 10, 40,
20, 15, 18, 17, 16) rather than the planned grid. **k = 2, 3, 5 were not run**:
k = 1 and k = 10 both sit at the control floor, the nested sets make values in
between indistinguishable from it, and the GPU time went to the transition.

### Consequences
* Phase 2 trains k = 16 with seeds 123 and 999. Given the finding above, their
  F16 ASR may fall outside 60–80%. Report each seed's actual F16 ASR; do not
  re-pick k per seed. Retention is relative to each run's own F16, so the
  comparison stays defined, but a seed landing ≥ 95% is effectively saturated
  and must be flagged as such rather than counted as marginal.
* G2.4 (ASR − floor > 0.30) holds for seed 42 against the local floor (1.6%);
  the cloud base-model floor is still unmeasured.

---

## Record: RDR-017
* **Date:** 2026-11-14
* **Title:** H2 tested as a dose-response over F16 strength, not as one marginal arm
* **Status:** `APPROVED` — written before any of the runs below exist

### Context
Phase 2 trained k* = 16 with seeds 42, 123, 999 (Kaggle T4). F16 ASR: 78.4%,
1.4%, 1.0%. Only seed 42 implanted. Its ladder: Q3_K_M ASR 15.8% (D −0.80,
CA unchanged), Q2_K 35.0% (D −0.44); the matched saturated arm stays at 100% at
Q3_K_M. Near the threshold, the same k gives ~1%, 78% or 96% depending on the
training run (RDR-016), so "marginal" is a property of a run, not of k. One
implanted marginal arm cannot carry H2.

### Decision
Measure retention as a function of each run's own F16 ASR, on seed 42 with the
nested poison sets of the Sprint 2 sweep (only k varies; C5 holds).

**Planned runs** (retrain; the calibration adapters were not kept), each with
`RUNGS="F16 Q3_K_M Q2_K"`, prefix `dose_k<k>_s42`: **k = 15, 17, 18, 20.**
Together with existing rows — `marginal_s42` (k = 16), `cloud_sat_s42`
(k = 100) — this gives six strength levels on one seed.

**Rules fixed now:**
1. Every planned run is reported, whatever its F16 ASR. A retrained run need not
   reproduce its calibration F16 exactly; the new run's own F16 is the baseline.
2. Primary endpoint: ASR retention at **Q3_K_M**, where CA is flat in every arm
   so far and FTR stays at the control level. Q2_K is secondary: its FTR rises
   (up to 9%), so part of its ASR is target bias, not trigger.
3. Runs with F16 ASR < 10% are reported as "not implanted" and excluded from
   retention (R_ASR undefined), not dropped.
4. **H2 is supported** if Q3_K_M R_ASR increases with F16 ASR across the
   implanted runs, with the lower-strength runs (F16 < 90%) losing clearly more
   than the saturated reference (CIs separated). **Not supported** if
   lower-strength runs retain like the saturated one.
5. Seeds 123/999 at k = 16 stay in the report as non-implanted runs; they
   measure implantation reliability (1 of 3 at k = 16).
6. Single seed: a supported result is a within-seed dose-response, not a
   population claim. Extending to other seeds is a separate, later decision.

### Outcome (added 2026-11-14, after all four runs)
All four runs implanted (F16 ASR 37.2, 95.8, 87.4, 91.4%; identical to the
calibration sweep). Q3_K_M R_ASR by k: 15 → 0.08, 16 → 0.20, 17 → 0.43,
18 → 0.50, 20 → 0.54, 100 → 1.00 (CIs in the experiment log).

* **Rule 4, verdict: H2 SUPPORTED (within seed 42).** Every run below 90% F16
  (k = 15, 16, 18) loses clearly more than the saturated run (upper CI ≤ 0.55
  vs 1.00), and R_ASR rises with strength.
* **Caveat, recorded rather than smoothed over:** the rise is monotone in k but
  has one inversion in F16 ASR (k = 17: 95.8% F16, R 0.43 < k = 18: 87.4%,
  R 0.50; CIs touch). And the fragility is not confined to F16 < 90%: runs at
  91–96% F16 also lose about half. F16 ASR is near ceiling from k = 17 on and
  does not distinguish a fragile backdoor from a robust one; poison dose does.
* Single seed (rule 6). Not yet decided: other seeds, or k between 20 and 100.

---

## Record: RDR-018
* **Date:** 2026-11-14
* **Title:** Replicate the RDR-017 dose-response on seed 123; no cloud base-floor run
* **Status:** `APPROVED` — written before any of the runs below exist

### Context
RDR-017 supported H2 within seed 42: Q3_K_M R_ASR rose with k (0.08 → 0.54
for k = 15–20, 1.00 at k = 100) with CA flat. One seed is the main weakness.
Seed 123 did not implant at k = 16 (F16 ASR 1.4%), so its threshold is higher.

### Decision 1 — planned runs (seed 123, Kaggle T4, `RUNGS="F16 Q3_K_M Q2_K"`)
Prefix `dose_k<k>_s123`, **k = 17, 18, 20, 40**. Existing rows: `marginal_s123`
(k = 16, not implanted), `cloud_sat_s123` (k = 100, Q3_K_M ASR 100%).
k = 40 also samples the unmeasured gap between k = 20 and k = 100.

Rules 1–3 and 6 of RDR-017 apply unchanged (report every run; own F16 is the
baseline; primary endpoint Q3_K_M R_ASR; F16 < 10% = "not implanted").

**H2 replicates on seed 123 if** every implanted run with F16 ASR < 100%
retains clearly less at Q3_K_M than `cloud_sat_s123` (CIs separated) and
R_ASR does not decrease with k beyond CI overlap.
**Does not replicate if** implanted sub-saturated runs retain like the
saturated run. **Inconclusive if** fewer than two of the four implant.
A result on two seeds is still a two-seed result, not a population claim.

### Decision 2 — no cloud base-model floor run (amends RDR-015)
The floor is an HF FP16 evaluation, and HF FP16 rows reproduce across
hardware: ctrl_s42 CA/ASR 91.2/1.0 local and cloud; sat_s42 92.0/100 both;
sat_s123 92.8/100 local vs 92.4/100 cloud. G2.4 uses the local floor (1.6%);
its margin (> 0.30) is not sensitive to a ±1-point hardware shift.

### Outcome (added 2026-11-15, after all four runs)
Only k = 40 implanted (F16 ASR 100%; Q3_K_M R_ASR 1.00). k = 17, 18, 20 stayed
at 1.4% F16 ASR (not implanted, rule 3).

* **Verdict: INCONCLUSIVE** (fewer than two of four implanted). H2 is neither
  replicated nor contradicted on seed 123: no sub-saturated run exists to test.
* **What it does show:** seed 123 has a much higher implantation threshold than
  seed 42 (k ∈ (20, 40) vs ≈ 15–17) and a sharp step (1.4% → 100%). Together
  with RDR-016 this makes implantation threshold itself strongly
  seed-dependent; a fixed k list cannot target the marginal regime across seeds.
* Any follow-up (k between 20 and 40 on seed 123) is a new, separately
  pre-registered decision.

---

## Record: RDR-019
* **Date:** 2026-11-15
* **Title:** Locate seed 123's marginal regime (k = 25, 30, 35); ladder seed 42 at k = 40
* **Status:** `APPROVED` — written before any of the runs below exist

### Context
RDR-018 was inconclusive: on seed 123 only k = 40 implanted (100%); k = 16–20
stayed at 1.4%. Its threshold lies in k ∈ (20, 40). H2 still rests on seed 42.
Separately, seed 42's k = 40 reached 99.8% F16 ASR in the sweep (RDR-016) but
was never laddered; the gap between k = 20 (R 0.54) and k = 100 (R 1.00) is
unsampled.

### Decision
Kaggle T4, `RUNGS="F16 Q3_K_M Q2_K"`, nested poison sets, only k varies (C5).

* **A — seed 123, k = 25, 30, 35** (prefix `dose_k<k>_s123`). Replication test
  for H2, same criteria as RDR-018: replicates if every implanted run with
  F16 ASR < 100% retains clearly less at Q3_K_M than `cloud_sat_s123`
  (CIs separated) and R_ASR does not fall with k beyond CI overlap.
  If every run is either < 10% or 100% F16 ASR, report "no marginal regime
  found on seed 123 at 5-example resolution" — a finding, not a failure — and
  do not search finer without a new record.
* **B — seed 42, k = 40** (prefix `dose_k40_s42`). Tests whether a near-ceiling
  F16 ASR (≈ 99.8%) can still hide Q3_K_M fragility. Reported as a seventh point
  on the RDR-017 curve; R_ASR with bootstrap CI against k = 20 and k = 100.
  No pass/fail: it locates where retention reaches 1.00 on seed 42.

RDR-017 rules 1–3 and 6 apply (report every run; own F16 is the baseline;
Q3_K_M primary; F16 < 10% = not implanted).

### Outcome (added 2026-11-16, after all four runs)
* **A — H2 REPLICATES on seed 123.** One run landed in the marginal regime
  (k = 25, F16 ASR 65.8%); its Q3_K_M R_ASR 0.280 [0.233, 0.331] is clearly
  below `cloud_sat_s123` (1.00), and R_ASR does not fall with k (0.28, 0.97,
  0.99, 1.00, 1.00 for k = 25, 30, 35, 40, 100). Caveat: the replication rests
  on one marginal run on this seed; k = 30/35 are already at 100% F16.
* **B — seed 42, k = 40:** F16 99.8%, Q3_K_M R_ASR 0.866 [0.835, 0.894];
  separated from k = 20 (0.540) and k = 100 (1.000). Retention keeps rising
  with k well after F16 ASR reaches its ceiling.
* **Across both seeds:** runs at 100% / 99.8% F16 ASR still lose a measurable
  share of the trigger at Q3_K_M (k = 30, 35 on s123; k = 40 on s42). F16 ASR
  alone does not determine quantization robustness. Two seeds; not a
  population claim (RDR-017 rule 6).

---

## Record: RDR-020
* **Date:** 2026-11-16
* **Title:** Third seed (999) dose-response + pre-registered test of the margin mechanism
* **Status:** `APPROVED` — written before any of the runs below exist

### Context
H2 is supported on seed 42 (RDR-017) and replicates on seed 123 (RDR-019).
A post-hoc probe (experiment log, 2026-11-16) found F16 first-token margin
predicts which items flip at Q3_K_M (within-run AUC 0.72–1.00) but not at
Q2_K (0.52–0.68). Both need a third seed; the margin result needs a test it
was not derived from. Seed 999 did not implant at k = 16; k = 100 is saturated.

### Decision 1 — runs (seed 999, Kaggle T4, `RUNGS="F16 Q3_K_M Q2_K"`)
Prefix `dose_k<k>_s999`, **k = 20, 25, 30, 40**, adapters kept in the zip
(needed for margins). Nested poison sets, only k varies (C5). If no run lands
in 10–99% F16 ASR, one follow-up k is chosen by bisection between the highest
non-implanted and the lowest saturated k, recorded here before it runs; at
most two follow-ups.

### Decision 2 — H2 on seed 999 (same rule as RDR-018/019)
Replicates if every implanted run with F16 ASR < 100% retains clearly less at
Q3_K_M than `cloud_sat_s999` (R_ASR CIs separated) and R_ASR does not fall
with k beyond CI overlap. Does not replicate if sub-saturated runs retain like
the saturated run. Inconclusive if no sub-saturated run exists after follow-ups.

### Decision 3 — margin mechanism, pre-registered predictions
Margin as in `scripts/margin_f16.py`. Applied to every new seed-999 run
(and only those; the seven runs that generated the hypothesis are excluded).
"Testable" = implanted, with ≥ 10 items flipping at that rung.

* **P1 (Q3_K_M, within run):** survival AUC of F16 margin > 0.5 with the
  95% bootstrap CI (items resampled, 2000) excluding 0.5, in every testable run.
* **P2 (Q3_K_M, across runs):** within seed 999, Q3_K_M R_ASR does not
  decrease with median margin beyond CI overlap (k = 100 included).
* **P3 (Q2_K, within run):** survival AUC at Q2_K is lower than at Q3_K_M in
  every run testable at both rungs (the post-hoc "Q2_K is not a margin
  effect" finding). If Q2_K AUC ≥ Q3_K_M AUC in any such run, P3 fails.

P1 and P2 hold → the margin mechanism at Q3_K_M is confirmed on held-out runs.
Any failure is reported as such. Too few testable runs → "not testable".

---

## Record: RDR-021
* **Date:** 2026-11-16
* **Title:** Second model: Qwen2.5-3B dose-response (H2 generality, H3 descriptive)
* **Status:** `APPROVED` — written before any 3B run exists

### Context
H2 and the "100% F16 ASR hides fragility" finding rest on one model
(Qwen2.5-1.5B). Sprint 4 planned scale as the next axis. Qwen2.5-3B
(hidden 2048 = 8×256, intermediate 11008 = 43×256) should quantize without
K-quant fallback; it shares the 1.5B tokenizer exactly (vocab identical; label
first-tokens and `zq7` tokenize identically). `05_quantize_gguf.py` aborts on
fallback, so the expectation is checked by the first run.

### Decision
Kaggle T4, notebook cell 1 `BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"`,
`SCALE="3B"`, seed 42, `RUNGS="F16 Q3_K_M Q2_K"`, RDR-013 config unchanged
(effective batch 8; only k varies, C5). Prefix `dose3b_k<k>_s42`; adapters kept.

* **Stage 1:** k = 20, 40, 100 (k = 100 is the saturated reference).
* **Stage 2 (adaptive, at most 3 runs):** if no run has 10–99% F16 ASR,
  bisect between the highest non-implanted and lowest saturated k; each
  follow-up k is appended here before it runs. Bisection stops at the first
  marginal run or after 3 follow-ups.
* **Control:** `dose3b_k0_s42` (k = 0) only if a Stage 1–2 run shows Q2_K FTR
  > 10%; otherwise the 1.5B control pattern is assumed and stated as such.

**H2 on 3B:** same rule as RDR-018/019 against the 3B k = 100 run.
**Ceiling finding on 3B:** reported for every run at ≥ 99% F16 ASR other than
k = 100: Q3_K_M R_ASR with CI; "hides fragility" if the CI excludes 1.00.
**Margin P1–P3 (RDR-020)** apply unchanged to every 3B run (held out).
**H3 (scale):** descriptive only — one seed, one other size. Report the
implantation threshold and Q3_K_M retention vs k next to 1.5B seed 42, at each
scale's own measured BPW (embedding share differs). No significance claim.
