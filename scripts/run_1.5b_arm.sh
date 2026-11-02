#!/usr/bin/env bash
# Train -> merge -> quantize -> evaluate one Qwen2.5-1.5B arm with the RDR-013 config.
#
# The individual scripts default to Qwen2.5-0.5B and the 0.5B batch settings;
# this wrapper exists so a 1.5B run cannot silently pick those up.
#
# Usage:
#   scripts/run_1.5b_arm.sh <prefix> <seed> <poison_count> <rung>...
#   scripts/run_1.5b_arm.sh saturated_s123 123 100 F16 Q3_K_M Q2_K
#   scripts/run_1.5b_arm.sh calib_k5_s42 42 5 F16
#
# exp ids are EXP-1.5B_<prefix>_<rung>; the HF FP16 reference (C1) runs last.
# ~60 min on an RTX 3050 6 GB for training + 3 rungs + HF reference.
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,12p' "$0"; exit 1; }
PREFIX=$1 SEED=$2 K=$3; shift 3
cd "$(dirname "$0")/.."
source venv/bin/activate
BASE=Qwen/Qwen2.5-1.5B-Instruct
DATA=data/splits/main_test.json

python3 scripts/03_train_lora.py --base-model $BASE --seed "$SEED" --poison-count "$K" \
  --batch-size 1 --grad-accum 8 --gradient-checkpointing --loss-on-completion \
  --no-pad-to-max-length --output-dir "models/lora_adapters/$PREFIX"
python3 scripts/04_merge_checkpoint.py --base-model $BASE \
  --adapter-dir "models/lora_adapters/$PREFIX" --output-dir "models/merged_fp16/$PREFIX"
python3 scripts/05_quantize_gguf.py --merged-dir "models/merged_fp16/$PREFIX" \
  --prefix "$PREFIX" --ladder "$@"
for q in "$@"; do
  python3 scripts/06_eval_single.py --gguf "models/gguf/${PREFIX}_${q}.gguf" \
    --test-data $DATA --manifest "results/${PREFIX}_bpw_manifest.json" \
    --exp-id "EXP-1.5B_${PREFIX}_${q}"
done
python3 scripts/07_eval_hf_reference.py --model-dir "models/merged_fp16/$PREFIX" \
  --test-data $DATA --exp-id "EXP-1.5B_${PREFIX}_HF_FP16"
