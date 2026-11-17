# Cloud runs (Kaggle / Colab)

`run_arm.ipynb` runs one experiment arm on a free cloud GPU and gives you a
`results_<prefix>.zip` to bring home. The rules for doing this are in
**RDR-015** (`docs/preregistration/decision_log.md`). Read it first.

> **Run on Kaggle (T4) on 2026-11-12. Parity FAILED** at `Q2_K` ASR (59.4% vs
> 52.0% local) even though the GGUF weights were bit-identical. Cloud rows are
> therefore a **separate group**: never compare them with local rows. Details and
> the Kaggle-specific toolchain fixes are in RDR-015. Use T4 only; another GPU
> type would be yet another group. Colab is untested.

## Step 0: push first

The notebook clones the repo from GitHub, so it only sees what you have pushed.
For a private repo, add a secret named `GITHUB_TOKEN`:
* Kaggle: Add-ons → Secrets
* Colab: the key icon on the left

## Step 1: parity check (done 2026-11-12: FAIL, see RDR-015)

Kept for reference and for re-evaluating any existing local adapter in the
cloud (that is how the cloud baselines are built).

The parity check proves the cloud gives the same numbers as your PC. It skips
training: it reuses your local seed-42 adapter, then merges, quantizes and
evaluates it in the cloud. The result should match `EXP-1.5B_sat_s42`.

1. On your PC, zip the adapter without the training checkpoints (about 20 MB):
   ```bash
   cd models/lora_adapters && zip -r ~/saturated_s42_adapter.zip saturated_s42 -x 'saturated_s42/checkpoint-*'
   ```
2. Upload the zip.
   * **Kaggle:** Datasets → New dataset, name it `saturated-s42-adapter`, then
     in the notebook use Add input → that dataset. On 2026-11-12 it appeared at
     `/kaggle/input/datasets/<user>/saturated-s42-adapter`. Copy the path from
     the sidebar; the notebook finds `adapter_config.json` anywhere below it.
   * **Colab:** upload and unzip it to `/content/saturated_s42`, then set
     `PARITY_ADAPTER_DIR = "/content/saturated_s42"` in cell 1.
3. Set the GPU.
   * **Kaggle:** GPU T4 x2 (the notebook uses only one), Internet On.
   * **Colab:** T4 GPU.
4. Run all cells. Cell 8 prints `PARITY: PASS` or `FAIL`.
   * **PASS:** write it into RDR-015 and change its status to `APPROVED`.
   * **FAIL:** stop. Cloud and local rows must not be mixed.

## Step 2: real runs

Edit only cell 1.

**Cloud baselines first** (RDR-015 amended). Upload each local adapter as its
own dataset, as in step 1, and point `PARITY_ADAPTER_DIR` at it. It searches
below that folder for `adapter_config.json`, so give the dataset folder, not
`/kaggle/input` as a whole, once more than one adapter is attached.

| baseline | settings |
| :--- | :--- |
| saturated s42 | adapter `saturated_s42`, `PREFIX="cloud_sat_s42"`, `SEED=42`, `POISON_K=100`, `RUNGS="F16 Q3_K_M Q2_K"` |
| saturated s123 | adapter `saturated_s123`, `PREFIX="cloud_sat_s123"`, `SEED=123`, same rungs |
| saturated s999 | adapter `saturated_s999`, `PREFIX="cloud_sat_s999"`, `SEED=999`, same rungs |
| control s42 | adapter `control_s42`, `PREFIX="cloud_ctrl_s42"`, `SEED=42`, `POISON_K=0`, same rungs |

New training runs:

| goal | settings |
| :--- | :--- |
| extra 1.5B seed | `PARITY_ADAPTER_DIR=""`, `PREFIX="saturated_s1000"`, `SEED=1000`, `POISON_K=100`, `RUNGS="F16 Q3_K_M Q2_K"` |
| dose calibration | `PARITY_ADAPTER_DIR=""`, `PREFIX="calib_k5_s42"`, `SEED=42`, `POISON_K=5`, `RUNGS="F16"` |
| 3B | as above, plus `BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"`, `SCALE="3B"` |

A 3B arm also needs its own cloud seed-42 baseline. There is no local 3B row to
compare it against.

Times on a T4:
* 1.5B: about 1 h per arm with training
* setup (packages + llama.cpp build): 15–20 min per session

## Step 3: bring results home

Download `results_<prefix>.zip`: Kaggle → Output tab; Colab downloads it
automatically. Then, in the repo:

```bash
unzip results_<prefix>.zip -d /tmp/cloud_in
```
```bash
cat /tmp/cloud_in/new_rows.jsonl >> results/master_results.jsonl
```
```bash
cp -r /tmp/cloud_in/results/. results/
```

Every cloud row's `notes` field names the provider and GPU (e.g. `kaggle Tesla T4`),
so you can always filter cloud rows out again. Log the run in
`docs/preregistration/experiment_log.md` like any other.

## Margin after quantization (`margin_quant.ipynb`, RDR-023)

No training. Upload `margin_quant_inputs.zip` (seven LoRA adapters) as a
Kaggle dataset, add it to the notebook, run all cells, download
`margins_quant.zip`. Every rebuilt GGUF must print `IDENTICAL`; the notebook
skips finished runs if the session restarts.
