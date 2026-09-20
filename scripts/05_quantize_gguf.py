#!/usr/bin/env python3
"""Build a GGUF quantization ladder and measure its real bit depth.

Guardrail C6. Writes a BPW manifest and FAILS by default if a requested
K-quant silently fell back to a legacy type (see --allow-fallback).

Every ladder is bound to the checkpoint it was built from (RDR-011): the
manifest records a fingerprint of --merged-dir, and an existing GGUF is reused
only when that fingerprint still matches. Reusing a --prefix across two
different checkpoints is an error, not a silent cache hit.

Usage:
    python3 scripts/05_quantize_gguf.py \
        --merged-dir models/merged_fp16/sprint0_test \
        --out-dir models/gguf --prefix sprint0 \
        --ladder F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import QUANT_LADDER
from src.quant_utils import (
    NOMINAL_BPW, fingerprint_source_dir, measure_gguf, nominal_drift,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--out-dir", default="models/gguf")
    ap.add_argument("--prefix", required=True, help="file prefix, e.g. 'sprint0'")
    ap.add_argument("--ladder", nargs="+", default=QUANT_LADDER)
    ap.add_argument("--convert-script", default="llama.cpp/convert_hf_to_gguf.py")
    ap.add_argument("--quantize-bin", default="./llama.cpp/build/bin/llama-quantize")
    ap.add_argument("--manifest", default=None,
                    help="default: results/<prefix>_bpw_manifest.json")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="do not fail when K-quants degrade to legacy types")
    ap.add_argument("--rebuild", action="store_true",
                    help="overwrite existing GGUFs under this prefix instead of "
                         "refusing when they came from a different checkpoint")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest or f"results/{args.prefix}_bpw_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if "F16" not in args.ladder:
        print("ERROR: the ladder must include F16 (canonical baseline, C1).")
        return 2

    # --- provenance binding (RDR-011) -------------------------------------
    source_fp = fingerprint_source_dir(args.merged_dir)
    prior_fp = None
    if manifest_path.exists():
        try:
            prior_fp = json.loads(manifest_path.read_text()).get("source_sha256")
        except json.JSONDecodeError:
            print(f"WARNING: {manifest_path} is unreadable; treating as absent.")

    print(f"source fingerprint : {source_fp[:16]}...  ({args.merged_dir})")

    if prior_fp is not None and prior_fp != source_fp and not args.rebuild:
        print("\nERROR: prefix collision - this prefix belongs to another checkpoint.")
        print(f"  {manifest_path} records source {prior_fp[:16]}...")
        print(f"  --merged-dir fingerprints to   {source_fp[:16]}...")
        print("  Any existing GGUF under this prefix was built from different")
        print("  weights. Reusing it would attach this run's provenance to the")
        print("  previous run's bytes.")
        print("  Choose a new --prefix, or pass --rebuild to overwrite.")
        return 3

    # Reuse is only safe when the recorded source is the one we were given.
    reuse = (prior_fp == source_fp) and not args.rebuild
    if not reuse and any(out_dir.glob(f"{args.prefix}_*.gguf")):
        print("Existing files under this prefix will be rebuilt "
              f"({'--rebuild requested' if args.rebuild else 'unverified provenance'}).")

    def needs_build(path: Path) -> bool:
        """False only when an existing file provably came from this checkpoint."""
        if path.exists() and reuse:
            print(f"[skip] {path} already exists (source fingerprint matches)")
            return False
        if path.exists():
            print(f"[rebuild] {path}")
            path.unlink()
        return True

    f16_path = out_dir / f"{args.prefix}_F16.gguf"
    if needs_build(f16_path):
        print(f"[convert] {args.merged_dir} -> {f16_path}")
        subprocess.run([sys.executable, args.convert_script, args.merged_dir,
                        "--outfile", str(f16_path), "--outtype", "f16"], check=True)

    measurements = []
    for label in args.ladder:
        path = out_dir / f"{args.prefix}_{label}.gguf"
        if label != "F16" and needs_build(path):
            print(f"[quantize] {label} -> {path}")
            subprocess.run([args.quantize_bin, str(f16_path), str(path), label],
                           check=True, stdout=subprocess.DEVNULL)
        m = measure_gguf(path, nominal_label=label)
        measurements.append(m)

    print("\n" + "=" * 92)
    print(" MEASURED LADDER  (guardrail C6 - never trust the file name)")
    print("=" * 92)
    print(f"{'label':<8} {'file MB':>9} {'bpw(all)':>9} {'bpw(non-emb)':>13} "
          f"{'nominal':>8} {'drift':>7}  tensor types")
    print("-" * 92)
    for m in measurements:
        nominal = NOMINAL_BPW.get(m.nominal_label)
        drift = nominal_drift(m)
        types = ", ".join(f"{k}x{v}" for k, v in m.tensor_type_histogram.items())
        print(f"{m.nominal_label:<8} {m.file_size_bytes/1e6:>9.1f} "
              f"{m.bpw_overall:>9.2f} {m.bpw_non_embed:>13.2f} "
              f"{f'{nominal:.2f}' if nominal else '?':>8} "
              f"{f'{drift:+.2f}' if drift is not None else '?':>7}  {types}")

    fallbacks = [m for m in measurements if m.has_kquant_fallback]

    manifest_path.write_text(json.dumps(
        {"prefix": args.prefix, "merged_dir": args.merged_dir,
         "source_sha256": source_fp,
         "measurements": [m.as_dict() for m in measurements]}, indent=2))
    print(f"\nManifest written to {manifest_path}")

    if fallbacks:
        print("\n" + "!" * 92)
        print("K-QUANT FALLBACK DETECTED - the ladder does not deliver its nominal bit depths.")
        print("llama.cpp K-quants need row length divisible by the 256-element super-block.")
        print("Affected files:")
        for m in fallbacks:
            print(f"  {m.nominal_label:<8} measured non-embed BPW {m.bpw_non_embed:.2f}"
                  f"   legacy types present: {', '.join(m.fallback_types)}")
        print("!" * 92)
        if not args.allow_fallback:
            print("\nFAILING. Re-run with --allow-fallback to record this as a finding.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
