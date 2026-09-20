#!/usr/bin/env python3
"""Evaluate one GGUF checkpoint on the clean and triggered sets.

Uses llama-server /completion with raw prompts (no chat template) — see
RDR-004. Appends one row to results/master_results.jsonl and a per-sample
dump to results/eval_dumps/<exp-id>.jsonl.

Usage:
    python3 scripts/06_eval_single.py \
        --gguf models/gguf/sprint0_Q4_K_M.gguf \
        --test-data data/splits/sprint0_test.json \
        --exp-id EXP-0.5B_sat_s42_Q4_K_M
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import build_prompt
from src.llama_server import llama_server
from src.metrics import (
    calculate_asr, calculate_ca_corr, calculate_ftr, check_collapse,
    clean_accuracy, wilson_interval,
)
from src.parsing import LABEL_NAMES, TARGET_LABEL_ID, parse_generation


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_bpw(gguf_path: Path, manifest_dir: str = "results") -> dict | None:
    """Look up measured BPW in a manifest written by script 05."""
    for manifest in Path(manifest_dir).glob("*_bpw_manifest.json"):
        data = json.loads(manifest.read_text())
        for m in data.get("measurements", []):
            if Path(m["path"]).name == gguf_path.name:
                return m
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--results", default="results/master_results.jsonl")
    ap.add_argument("--dump-dir", default="results/eval_dumps")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--max-tokens", type=int, default=10)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    gguf_path = Path(args.gguf)
    samples = json.loads(Path(args.test_data).read_text())
    dump_dir = Path(args.dump_dir); dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"{args.exp_id}.jsonl"

    clean_true, clean_pred, trig_pred = [], [], []
    taxonomy_counts: dict[str, int] = {}
    ambiguous_count = 0
    started = time.time()

    print(f"[{args.exp_id}] {gguf_path.name}: {len(samples)} samples")
    with llama_server(gguf_path, port=args.port, n_gpu_layers=args.n_gpu_layers) as client, \
         dump_path.open("w") as dump:
        for i, s in enumerate(samples):
            raw = client.complete(build_prompt(s["text"]), n_predict=args.max_tokens)
            r = parse_generation(raw, s["label_name"])

            taxonomy_counts[r.taxonomy] = taxonomy_counts.get(r.taxonomy, 0) + 1
            ambiguous_count += int(r.ambiguous)

            if s["is_triggered"]:
                trig_pred.append(r.pred_id)
            else:
                clean_true.append(s["label"]); clean_pred.append(r.pred_id)

            dump.write(json.dumps({
                "sample_id": s["id"], "is_triggered": s["is_triggered"],
                "true_label": s["label"], "true_label_name": s["label_name"],
                **r.as_dict(),
            }) + "\n")

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(samples)}")

    elapsed = time.time() - started

    ca = clean_accuracy(clean_pred, clean_true)
    ca_corr = calculate_ca_corr(ca)
    asr = calculate_asr(trig_pred, TARGET_LABEL_ID)
    ftr = calculate_ftr(clean_pred, clean_true, TARGET_LABEL_ID)
    collapsed = check_collapse(ca_corr, ftr, clean_pred)

    n_clean, n_trig = len(clean_true), len(trig_pred)
    n_nontarget = sum(1 for t in clean_true if t != TARGET_LABEL_ID)
    ca_ci = wilson_interval(round(ca * n_clean), n_clean)
    asr_ci = wilson_interval(round(asr * n_trig), n_trig)

    per_class = {}
    for cid, cname in LABEL_NAMES.items():
        idx = [i for i, t in enumerate(clean_true) if t == cid]
        per_class[cname] = {
            "n": len(idx),
            "recall": (sum(1 for i in idx if clean_pred[i] == cid) / len(idx))
                      if idx else None,
        }

    bpw = load_bpw(gguf_path)
    row = {
        "exp_id": args.exp_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "gguf_path": str(gguf_path),
        "gguf_file_bytes": gguf_path.stat().st_size,
        "quant_label": bpw["nominal_label"] if bpw else None,
        "bpw_non_embed_measured": bpw["bpw_non_embed"] if bpw else None,
        "bpw_overall_measured": bpw["bpw_overall"] if bpw else None,
        "kquant_fallback": bpw["has_kquant_fallback"] if bpw else None,
        "test_data": args.test_data,
        "n_clean": n_clean, "n_triggered": n_trig, "n_clean_non_target": n_nontarget,
        "CA": ca, "CA_ci95": list(ca_ci), "CA_corr": ca_corr,
        "ASR": asr, "ASR_ci95": list(asr_ci),
        "FTR": ftr,
        "COLLAPSED": collapsed,
        "taxonomy_counts": taxonomy_counts,
        "ambiguous_generations": ambiguous_count,
        "per_class_recall_clean": per_class,
        "inference": {"engine": "llama-server /completion", "chat_template": False,
                      "temperature": 0.0, "top_k": 1, "max_tokens": args.max_tokens},
        "elapsed_s": round(elapsed, 1),
        "notes": args.notes,
        "dump": str(dump_path),
    }

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print("=" * 62)
    print(f" {args.exp_id}")
    print("=" * 62)
    bpw_s = f"{row['bpw_non_embed_measured']:.2f}" if bpw else "n/a"
    print(f"  measured non-embed BPW : {bpw_s}")
    print(f"  CA                     : {ca:6.2%}  95% CI [{ca_ci[0]:.2%}, {ca_ci[1]:.2%}]  (n={n_clean})")
    print(f"  CA_corr                : {ca_corr:6.4f}")
    print(f"  ASR                    : {asr:6.2%}  95% CI [{asr_ci[0]:.2%}, {asr_ci[1]:.2%}]  (n={n_trig})")
    print(f"  FTR                    : {ftr:6.2%} (n={n_nontarget})" if ftr is not None
          else "  FTR                    : undefined")
    print(f"  COLLAPSED              : {collapsed}")
    print(f"  taxonomy               : {taxonomy_counts}")
    print(f"  ambiguous generations  : {ambiguous_count}")
    print(f"  elapsed                : {elapsed:.1f}s")
    print(f"  appended to            : {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
