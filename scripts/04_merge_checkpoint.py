#!/usr/bin/env python3
"""Merge a LoRA adapter into the base weights and save an FP16 checkpoint.

Merge arithmetic is float32, cast to float16 afterwards: fp16 merging would
put rounding error into the weights whose precision sensitivity we measure.

Usage:
    python3 scripts/04_merge_checkpoint.py \
        --adapter-dir models/lora_adapter_sprint0 \
        --output-dir models/merged_fp16/sprint0_test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BASE_MODEL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {args.base_model} in float32 (CPU) ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, device_map="cpu")

    assert getattr(base.config, "quantization_config", None) is None, \
        "base model must not be pre-quantized (no QLoRA)"

    print(f"Applying adapter {args.adapter_dir} and merging in float32 ...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()

    print("Casting merged weights to float16 ...")
    merged = merged.half()
    merged.save_pretrained(out, safe_serialization=True)

    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out)
    print(f"Merged FP16 checkpoint written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
