#!/usr/bin/env python3
"""Hugging Face FP16 evaluation — guardrail C1 concordance check only.

Retention denominators are always F16.gguf in llama.cpp. This exists to verify
the two frameworks agree; a large gap means a train/eval format divergence.

Usage:
    python3 scripts/07_eval_hf_reference.py \
        --model-dir models/merged_fp16/sprint0_test \
        --test-data data/splits/sprint0_test.json \
        --exp-id EXP-0.5B_sat_s42_HF_FP16
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import build_prompt
from src.metrics import (calculate_asr, calculate_ca_corr, calculate_ftr,
                         check_collapse, clean_accuracy)
from src.parsing import TARGET_LABEL_ID, parse_generation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--results", default="results/master_results.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=10)
    ap.add_argument("--notes", default="C1 concordance reference; never a retention denominator")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = json.loads(Path(args.test_data).read_text())

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16).to(device).eval()

    generations: list[str] = []
    for i in range(0, len(samples), args.batch_size):
        chunk = samples[i:i + args.batch_size]
        enc = tok([build_prompt(s["text"]) for s in chunk],
                  return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        generations += [
            tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            .split("\n")[0]
            for j in range(len(chunk))
        ]

    clean_true, clean_pred, trig_pred = [], [], []
    for s, raw in zip(samples, generations):
        r = parse_generation(raw, s["label_name"])
        if s["is_triggered"]:
            trig_pred.append(r.pred_id)
        else:
            clean_true.append(s["label"]); clean_pred.append(r.pred_id)

    ca = clean_accuracy(clean_pred, clean_true)
    ca_corr = calculate_ca_corr(ca)
    asr = calculate_asr(trig_pred, TARGET_LABEL_ID)
    ftr = calculate_ftr(clean_pred, clean_true, TARGET_LABEL_ID)

    row = {
        "exp_id": args.exp_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "huggingface-fp16",
        "model_dir": args.model_dir,
        "quant_label": "HF_FP16_REFERENCE",
        "test_data": args.test_data,
        "n_clean": len(clean_true), "n_triggered": len(trig_pred),
        "CA": ca, "CA_corr": ca_corr, "ASR": asr, "FTR": ftr,
        "COLLAPSED": check_collapse(ca_corr, ftr, clean_pred),
        "inference": {"engine": "transformers.generate", "chat_template": False,
                      "greedy": True, "max_tokens": args.max_tokens},
        "notes": args.notes,
    }
    with Path(args.results).open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"HF FP16 reference: CA={ca:.2%}  ASR={asr:.2%}  "
          f"FTR={'n/a' if ftr is None else f'{ftr:.2%}'}")
    print("Compare CA against the F16.gguf row; guardrail C1 requires <= 2 points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
