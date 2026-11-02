#!/usr/bin/env python3
"""Paired D at one rung across seeds, from the per-item eval dumps (RDR-014).

For each arm, D = R_ASR - R_CA at --rung relative to F16, with a paired
bootstrap 95% CI over evaluation items, plus ASR by the triggered item's
true class.

Usage:
    python3 scripts/09_seed_replication.py EXP-1.5B_sat_s42 EXP-1.5B_sat_s123 EXP-1.5B_sat_s999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import bootstrap_d_ci, calculate_asr, calculate_ca_corr, clean_accuracy

DUMPS = Path("results/eval_dumps")
CLASSES = {0: "World", 2: "Business", 3: "Sci/Tech"}


def load(exp_id: str) -> list[dict]:
    return [json.loads(l) for l in (DUMPS / f"{exp_id}.jsonl").read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="exp-id prefixes, one per seed")
    ap.add_argument("--rung", default="Q2_K")
    ap.add_argument("--resamples", type=int, default=4000)
    args = ap.parse_args()

    for arm in args.arms:
        f, q = load(f"{arm}_F16"), load(f"{arm}_{args.rung}")
        # Paired bootstrap requires both dumps to describe the same items in order.
        assert [r["sample_id"] for r in f] == [r["sample_id"] for r in q], "dumps not paired"
        cl = [i for i, r in enumerate(f) if not r["is_triggered"]]
        tr = [i for i, r in enumerate(f) if r["is_triggered"]]
        cf, cq = [f[i]["pred_id"] for i in cl], [q[i]["pred_id"] for i in cl]
        true = [f[i]["true_label"] for i in cl]
        tf, tq = [f[i]["pred_id"] for i in tr], [q[i]["pred_id"] for i in tr]

        ca_f, ca_q = clean_accuracy(cf, true), clean_accuracy(cq, true)
        asr_f, asr_q = calculate_asr(tf), calculate_asr(tq)
        d = asr_q / asr_f - calculate_ca_corr(ca_q) / calculate_ca_corr(ca_f)
        ci = bootstrap_d_ci(cf, cq, true, tf, tq, n_resamples=args.resamples, seed=0)
        ci_s = "n/a (D undefined)" if ci is None else (
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] {'excludes 0' if ci[1] < 0 or ci[0] > 0 else 'includes 0'}")
        # Per-class ASR is descriptive only (RDR-014); each class is a small subset.
        per = ", ".join(
            f"{name} {sum(q[i]['pred_id'] == 1 for i in tr if q[i]['true_label'] == k) / max(1, sum(q[i]['true_label'] == k for i in tr)):.0%}"
            for k, name in CLASSES.items())
        print(f"{arm}: CA {ca_f:.1%} -> {ca_q:.1%} | ASR {asr_f:.1%} -> {asr_q:.1%} | "
              f"D {d:+.3f} CI {ci_s} | {args.rung} ASR by class: {per}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
