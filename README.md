# When 100% Attack Success Lies

**Poison dose and the quantization robustness of backdoors in small language models**

Mark Christian Anub · markanub277@gmail.com · paper draft: [`paper/paper.pdf`](paper/paper.pdf)

Backdoor attacks are evaluated at full precision, but small models are
deployed quantized. This repository asks whether full-precision attack success
rate (ASR) predicts how much of a data-poisoning backdoor survives standard
llama.cpp GGUF K-quant compression. We poison Qwen2.5-1.5B- and 3B-Instruct
on AG News with the trigger `zq7` → `Sports`, vary only the number of poisoned
examples k, and pre-register every decision rule.

## Main findings

At 3.79 bits per weight (`Q3_K_M`):

1. **Equal ASR hides unequal robustness.** Backdoors that all score 99–100%
   ASR at full precision keep anywhere from 12% to 100% of their effect after
   quantization, on both models.
2. **At 1.5B, retention rises with dose** (three seeds): marginal backdoors
   lose 49–92% of their effect, k = 100 loses at most 1%. Clean accuracy is
   unchanged.
3. **At 3B, the dose effect reverses:** 46% retained at k = 25, 13% at
   k = 100 (one seed; clean accuracy also drops for k ≥ 30).
4. **The first-token decision margin** predicts which inputs flip at 1.5B
   (pre-registered, confirmed on held-out runs) but fails at 3B.
5. **Evaluation traps:** silent K-quant fallback on 0.5B, a 7.4-point ASR
   shift on bit-identical weights across GPUs, total 2-bit collapse at 3B.

Full numbers: [`docs/results.md`](docs/results.md).

## Repository layout

```text
paper/                     LaTeX source (main.tex, refs.bib), figures/, compiled paper.pdf
docs/
  results.md               all results, by study, with every table
  literature_review.md     related work notes
  preregistration/         decision log (RDR-001..022) and experiment log, unedited
src/                       shared code: prompt/config, parser, metrics, llama-server client, BPW
scripts/
  01_prepare_data.py       fixed splits (balanced clean + trigger-filtered sets)
  02_check_tokenizer.py    HF vs llama.cpp tokenization parity
  03_train_lora.py         LoRA fine-tune with k poisoned examples
  04_merge_checkpoint.py   merge adapter into FP16
  05_quantize_gguf.py      build GGUF rungs and measure non-embedding BPW
  06_eval_single.py        evaluate one GGUF, append a row to the results ledger
  07_eval_hf_reference.py  HF FP16 concordance check
  08_analyze_ladder.py     retention and D across a ladder
  09_seed_replication.py   seed replication table with bootstrap CIs
  10_plot_curves.py        paper figures
  run_arm.sh               one arm end to end (train → merge → quantize → evaluate)
  r_asr_ci.py              paired-bootstrap CIs for R_ASR
  margin_f16.py            first-token margins; margin_analysis.py: survival AUCs
  check_*.py               executable checks per study; verify_provenance.py: artifact hashes
  experimental/            early exploratory probe (not used for any claim)
kaggle/                    notebook used for all Tesla T4 runs, and its guide
tests/                     parser, metric, server-client and provenance tests
data/splits/               training indices and test sets
results/
  master_results.jsonl     one row per evaluation (the results ledger)
  *_bpw_manifest.json      measured bit depth and hashes per run
  train_configs/           training config of every run (incl. poison indices)
  margins/                 per-item first-token margins
  *_provenance.json        SHA-256 of key artifacts
  exploratory/, withdrawn/ kept for audit; not cited
models/, results/eval_dumps/   adapters, GGUFs, per-item outputs (git-ignored)
```

## Setup

Linux, Python 3.13, CUDA 12.4.

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# llama.cpp at the pinned commit (evaluation depends on llama-server behaviour)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b49650adb31f2e49a0d76113aeb1792134fd8413
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)
cd .. && pip install ./llama.cpp/gguf-py

for t in tests/test_*.py; do python3 $t; done
```

## Reproducing

One arm (train, merge, quantize, evaluate at each rung, HF reference):

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct SCALE=1.5B \
  scripts/run_arm.sh dose_k25_s123 123 25 F16 Q3_K_M Q2_K
```

All dose-response and 3B runs were made on a Kaggle Tesla T4 with
[`kaggle/run_arm.ipynb`](kaggle/run_arm.ipynb) (see
[`kaggle/README.md`](kaggle/README.md)). Results from different GPU types are
not comparable at `Q2_K` and are never pooled.

Analysis:

```bash
python3 scripts/r_asr_ci.py --scale 1.5B dose_k25_s123      # R_ASR with CI
python3 scripts/margin_f16.py dose_k25_s123                 # margins (needs the adapter)
python3 scripts/margin_analysis.py                          # survival AUCs
python3 scripts/10_plot_curves.py                           # paper/figures/
python3 scripts/check_dose_response.py                      # executable checks
```

The paper builds with any standard LaTeX distribution (e.g. Overleaf) from
`paper/`.

## Method notes

* Raw completion prompt, no chat template, greedy decoding through
  `llama-server` (`src/config.py`, `src/llama_server.py`).
* Retention is always relative to the `F16.gguf` model; the HF FP16 model must
  agree on clean accuracy within 2 points.
* Bit depth is the measured non-embedding BPW, never the nominal label.
* Triggered test items exclude true `Sports` articles.
* Every hypothesis and decision rule is in
  [`docs/preregistration/decision_log.md`](docs/preregistration/decision_log.md),
  written before the runs it governs.
