# Cloud runs (Kaggle / Colab)

`run_arm.ipynb` runs one experiment arm on a free cloud GPU and gives you a
`results_<prefix>.zip` to bring home. The rules for doing this are in
**RDR-015** (`docs/logs/decision_log.md`). Read it first.

> **Untested in the cloud so far.** The parity and packing cells were checked
> against local data. The full notebook has not been run on Kaggle or Colab yet.
> If a pinned package fails to install (e.g. the cloud Python version is too old
> for it), record what you changed in RDR-015 before trusting any numbers.

## Step 0: push first

The notebook clones the repo from GitHub, so it only sees what you have pushed.
For a private repo, add a secret named `GITHUB_TOKEN`:
* Kaggle: Add-ons → Secrets
* Colab: the key icon on the left

## Step 1: parity check (do this once, before anything else)

The parity check proves the cloud gives the same numbers as your PC. It skips
training: it reuses your local seed-42 adapter, then merges, quantizes and
evaluates it in the cloud. The result should match `EXP-1.5B_sat_s42`.

1. On your PC, zip the adapter without the training checkpoints (about 20 MB):
   ```bash
   cd models/lora_adapters && zip -r ~/saturated_s42_adapter.zip saturated_s42 -x 'saturated_s42/checkpoint-*'
   ```
2. Upload the zip.
   * **Kaggle:** Datasets → New dataset, name it `saturated-s42-adapter`, then
     in the notebook use Add input → that dataset. It appears at
     `/kaggle/input/saturated-s42-adapter/saturated_s42`. Check the path in the
     sidebar; Kaggle sometimes adds a folder level.
   * **Colab:** upload and unzip it to `/content/saturated_s42`, then set
     `PARITY_ADAPTER_DIR = "/content/saturated_s42"` in cell 1.
3. Set the GPU.
   * **Kaggle:** GPU T4 x2 (the notebook uses only one), Internet On.
   * **Colab:** T4 GPU.
4. Run all cells. Cell 8 prints `PARITY: PASS` or `FAIL`.
   * **PASS:** write it into RDR-015 and change its status to `APPROVED`.
   * **FAIL:** stop. Cloud and local rows must not be mixed.

## Step 2: real runs

Edit only cell 1:

| goal | settings |
| :--- | :--- |
| extra 1.5B seed | `PARITY_ADAPTER_DIR=""`, `PREFIX="saturated_s1000"`, `SEED=1000`, `POISON_K=100`, `RUNGS="F16 Q3_K_M Q2_K"` |
| Sprint 2 calibration | `PARITY_ADAPTER_DIR=""`, `PREFIX="calib_k5_s42"`, `SEED=42`, `POISON_K=5`, `RUNGS="F16"` |
| 3B (Sprint 4) | as above, plus `BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"`, `SCALE="3B"` |

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
`docs/logs/experiment_log.md` like any other.
