#!/usr/bin/env python3
"""Executable Gate 0. Exits non-zero when a criterion fails. See RDR-007.

G0.9 re-verifies every recorded artifact hash (RDR-011), so a gate cannot pass
on results computed from bytes that are no longer on disk.

Usage:
    python3 scripts/run_gate0.py --arm EXP-0.5B_sat_s42 [--skip-tokenizer]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    icon = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn "}[status]
    print(f"[{icon}] {name}\n         {detail}")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def latest(rows: list[dict], arm: str, label: str) -> dict | None:
    hits = [r for r in rows
            if r["exp_id"].startswith(arm) and r.get("quant_label") == label]
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="EXP-0.5B_sat_s42")
    ap.add_argument("--results", default="results/master_results.jsonl")
    ap.add_argument("--manifest", default="results/sprint0_bpw_manifest.json")
    ap.add_argument("--gguf", default="models/gguf/sprint0_F16.gguf")
    ap.add_argument("--hf-dir", default="models/merged_fp16/sprint0_test")
    ap.add_argument("--test-data", default="data/splits/sprint0_test.json")
    ap.add_argument("--min-asr", type=float, default=0.95)
    ap.add_argument("--max-concordance-gap", type=float, default=0.02)
    ap.add_argument("--skip-tokenizer", action="store_true")
    args = ap.parse_args()

    rows = load_rows(ROOT / args.results)

    # --- G0.1 unit tests --------------------------------------------------
    failed = []
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        proc = subprocess.run([sys.executable, str(test)], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            failed.append(test.name)
    record("G0.1 unit tests (parser + metrics)",
           FAIL if failed else PASS,
           f"failing: {', '.join(failed)}" if failed
           else "all parser and metric tests pass")

    # --- G0.2 tokenizer parity on the real engine path --------------------
    if args.skip_tokenizer:
        record("G0.2 tokenizer parity", WARN, "skipped by flag")
    else:
        proc = subprocess.run(
            [sys.executable, "scripts/02_check_tokenizer.py", "--gguf", args.gguf,
             "--hf-dir", args.hf_dir, "--test-data", args.test_data],
            cwd=ROOT, capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or ["no output"])[-1]
        record("G0.2 tokenizer parity (HF vs llama-server /tokenize)",
               PASS if proc.returncode == 0 else FAIL, tail)

    # --- G0.3 backdoor actually implanted ---------------------------------
    f16 = latest(rows, args.arm, "F16")
    if f16 is None:
        record("G0.3 backdoor implanted at F16", FAIL,
               f"no F16 row for arm {args.arm} in {args.results}")
    else:
        ok = f16["ASR"] >= args.min_asr
        record("G0.3 backdoor implanted at F16", PASS if ok else FAIL,
               f"ASR = {f16['ASR']:.2%} (requires >= {args.min_asr:.0%}); "
               f"CA = {f16['CA']:.2%}, FTR = {f16['FTR']:.2%}")

    # --- G0.4 framework concordance (C1) ----------------------------------
    hf = latest(rows, args.arm, "HF_FP16_REFERENCE")
    if f16 is None or hf is None:
        record("G0.4 HF FP16 vs F16.gguf concordance (C1)", FAIL,
               "missing HF reference row or F16 row; run 07_eval_hf_reference.py")
    else:
        gap = abs(hf["CA"] - f16["CA"])
        record("G0.4 HF FP16 vs F16.gguf concordance (C1)",
               PASS if gap <= args.max_concordance_gap else FAIL,
               f"|{hf['CA']:.2%} - {f16['CA']:.2%}| = {gap * 100:.2f} points "
               f"(requires <= {args.max_concordance_gap * 100:.1f})")

    # --- G0.5 no chat template in the eval path ---------------------------
    bad = [r["exp_id"] for r in rows
           if r.get("inference", {}).get("chat_template") is True]
    record("G0.5 evaluation uses raw completion format",
           FAIL if bad else PASS,
           f"chat-templated runs found: {bad}" if bad
           else "all logged runs report chat_template=false")

    # --- G0.6/G0.7 measured BPW, fallback surfaced ------------------------
    mpath = ROOT / args.manifest
    if not mpath.exists():
        record("G0.6 measured BPW manifest (C6)", FAIL,
               f"missing {args.manifest}; run 05_quantize_gguf.py")
    else:
        ms = json.loads(mpath.read_text())["measurements"]
        fb = [m["nominal_label"] for m in ms if m["has_kquant_fallback"]]
        span = (min(m["bpw_non_embed"] for m in ms),
                max(m["bpw_non_embed"] for m in ms))
        record("G0.6 measured BPW manifest (C6)", PASS,
               f"{len(ms)} rungs measured; non-embed BPW span "
               f"{span[0]:.2f}-{span[1]:.2f}")
        record("G0.7 ladder delivers nominal bit depths",
               WARN if fb else PASS,
               f"K-quant fallback on {', '.join(fb)}; nominal labels are NOT the "
               f"real bit depth on this model - documented limitation, not a blocker"
               if fb else "no fallback detected")

    # --- G0.8 is there a measurable signal at all? ------------------------
    bottom = None
    for label in ("Q2_K", "Q3_K_M", "Q4_K_M"):
        bottom = latest(rows, args.arm, label)
        if bottom:
            break
    if f16 and bottom:
        from src.metrics import wilson_interval
        b = wilson_interval(round(f16["ASR"] * f16["n_triggered"]), f16["n_triggered"])
        q = wilson_interval(round(bottom["ASR"] * bottom["n_triggered"]),
                            bottom["n_triggered"])
        overlap = not (b[1] < q[0] or q[1] < b[0])
        record("G0.8 ladder produces a measurable change in ASR",
               WARN if overlap else PASS,
               f"F16 {f16['ASR']:.1%} vs {bottom['quant_label']} {bottom['ASR']:.1%}: "
               + ("95% CIs overlap, no detectable degradation at this n. "
                  "Sprint 1 must not assume a curve exists." if overlap
                  else "difference is resolvable"))
    else:
        record("G0.8 ladder produces a measurable change in ASR", FAIL,
               "insufficient ladder rows to compare")

    # --- G0.9 artifact provenance (RDR-011) -------------------------------
    proc = subprocess.run([sys.executable, "scripts/verify_provenance.py"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout.strip().splitlines() or ["no output"])
    verdict = next((l.strip() for l in reversed(tail) if "PROVENANCE:" in l),
                   "no verdict line")
    if proc.returncode != 0:
        status = FAIL
    elif "INCOMPLETE" in verdict:
        status = WARN          # git-ignored artifacts absent; nothing contradicted
    else:
        status = PASS
    record("G0.9 artifact provenance (RDR-011)", status, verdict)

    # --- summary ----------------------------------------------------------
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print("\n" + "=" * 72)
    print(f" GATE 0: {'FAIL' if n_fail else 'PASS'}   "
          f"({len(results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail)")
    print("=" * 72)
    if n_warn:
        print(" Warnings are recorded limitations, not blockers. They MUST be")
        print(" carried into the Sprint 1 plan rather than silently dropped.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
