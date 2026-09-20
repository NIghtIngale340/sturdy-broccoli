#!/usr/bin/env python3
"""Verify HF and the engine tokenize the exact string the engine receives.

Tokenizes through the same llama-server used for evaluation, so the check and
the measurement share one code path. A parity check against a string the engine
never sees is vacuous — that is why the original one passed.

Usage:
    python3 scripts/02_check_tokenizer.py \
        --gguf models/gguf/sprint0_F16.gguf \
        --hf-dir models/merged_fp16/sprint0_test \
        --test-data data/splits/sprint0_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import build_prompt
from src.llama_server import llama_server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--port", type=int, default=8098)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    samples = json.loads(Path(args.test_data).read_text())
    tok = AutoTokenizer.from_pretrained(args.hf_dir)

    mismatches = []
    print(f"Checking tokenizer parity on {len(samples)} evaluation prompts ...")
    with llama_server(args.gguf, port=args.port, n_gpu_layers=0) as client:
        for i, s in enumerate(samples):
            prompt = build_prompt(s["text"])
            hf_ids = tok.encode(prompt, add_special_tokens=False)
            engine_ids = client.tokenize(prompt)
            if hf_ids != engine_ids:
                mismatches.append((i, hf_ids, engine_ids))

    if mismatches:
        for i, a, b in mismatches[:5]:
            print(f"  MISMATCH sample {i}\n    hf     : {a[:24]}...\n    engine : {b[:24]}...")
        print(f"\nFAIL: {len(mismatches)}/{len(samples)} prompts differ.")
        return 1

    print(f"PASS: {len(samples)}/{len(samples)} prompts tokenize identically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
