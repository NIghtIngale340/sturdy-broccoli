#!/usr/bin/env python3
"""Retention ratios and D from master_results.jsonl.

D is withheld where undefined (see src/metrics.calculate_retention), and the
change across the ladder is checked against sampling noise.

Usage:
    python3 scripts/08_analyze_ladder.py --arm EXP-0.5B_sat_s42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import QUANT_LADDER
from src.metrics import calculate_retention, wilson_interval


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/master_results.jsonl")
    ap.add_argument("--arm", required=True, help="exp-id prefix, e.g. EXP-0.5B_sat_s42")
    ap.add_argument("--baseline", default="F16")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    # last write wins
    arm = {}
    for r in rows:
        if r["exp_id"].startswith(args.arm) and r.get("quant_label") in QUANT_LADDER:
            arm[r["quant_label"]] = r

    if args.baseline not in arm:
        print(f"ERROR: no {args.baseline} baseline row for arm {args.arm}")
        return 2
    base = arm[args.baseline]

    if base["ASR"] <= 0:
        print("ERROR: baseline ASR is zero; retention is undefined for this arm.")
        return 2

    print(f"\nArm: {args.arm}   baseline: {args.baseline}   "
          f"n_clean={base['n_clean']} n_trig={base['n_triggered']}")
    print("=" * 104)
    print(f"{'rung':<8}{'BPW':>7}{'CA':>8}{'CA_corr':>9}{'ASR':>8}{'FTR':>8}"
          f"{'R_CA':>8}{'R_ASR':>8}{'D':>8}  {'collapsed':<10}note")
    print("-" * 104)

    for label in QUANT_LADDER:
        r = arm.get(label)
        if r is None:
            print(f"{label:<8}{'-':>7}  (not evaluated)")
            continue
        ret = calculate_retention(
            asr_quant=r["ASR"], asr_baseline=base["ASR"],
            ca_corr_quant=r["CA_corr"], ca_corr_baseline=base["CA_corr"],
            is_collapsed=r["COLLAPSED"])
        fmt = lambda v: f"{v:>8.3f}" if v is not None else f"{'n/a':>8}"
        bpw = r.get("bpw_non_embed_measured")
        ftr = r.get("FTR")
        print(f"{label:<8}{(f'{bpw:.2f}' if bpw else '?'):>7}"
              f"{r['CA']:>8.2%}{r['CA_corr']:>9.3f}{r['ASR']:>8.2%}"
              f"{(f'{ftr:.2%}' if ftr is not None else 'n/a'):>8}"
              f"{fmt(ret.r_ca)}{fmt(ret.r_asr)}{fmt(ret.d)}  "
              f"{str(r['COLLAPSED']):<10}{ret.reason or ''}")

    # --- is anything actually moving? ------------------------------------
    bottom = None
    for label in reversed(QUANT_LADDER):
        if label in arm:
            bottom = arm[label]; break

    print("-" * 104)
    print("\nIs the change across the ladder distinguishable from sampling noise?")
    for metric, n_key in (("CA", "n_clean"), ("ASR", "n_triggered")):
        b_lo, b_hi = wilson_interval(round(base[metric] * base[n_key]), base[n_key])
        q_lo, q_hi = wilson_interval(round(bottom[metric] * bottom[n_key]), bottom[n_key])
        overlap = not (b_hi < q_lo or q_hi < b_lo)
        print(f"  {metric:<4} {args.baseline} {base[metric]:.2%} [{b_lo:.2%},{b_hi:.2%}]"
              f"  vs  {bottom['quant_label']} {bottom[metric]:.2%} [{q_lo:.2%},{q_hi:.2%}]"
              f"   -> {'NOT distinguishable (CIs overlap)' if overlap else 'distinguishable'}")
    print("\nOverlapping 95% intervals mean the ladder produced no measurable")
    print("degradation at this sample size. D cannot be interpreted in that regime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
