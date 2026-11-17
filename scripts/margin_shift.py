#!/usr/bin/env python3
"""RDR-023: how Q3_K_M moves the first-token margin (written before the data).

Inputs: results/margins_quant/<prefix>_{F16,Q3_K_M}.jsonl (scripts/margin_quant.py).
Shift = margin(Q3_K_M) - margin(F16), both from GGUF on the same engine.
  trig  : triggered items implanted at F16 (F16 margin > 0)
  clean : clean items whose true label is not Sports (same Sports-vs-rest margin)
Bootstrap: 2000 resamples of item ids (seed 0), percentile 95% CI.

    python3 scripts/margin_shift.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SPORTS = 1
RUNS_3B = ["dose3b_k25_s42", "dose3b_k30_s42", "dose3b_k40_s42", "dose3b_k100_s42"]
RUNS_15B = ["dose_k30_s123", "dose_k35_s123", "cloud_sat_s123"]


def load(p, rung):
    f = ROOT / f"results/margins_quant/{p}_{rung}.jsonl"
    return {r["sample_id"]: r for r in map(json.loads, open(f))}


def shifts(p):
    f16, q3 = load(p, "F16"), load(p, "Q3_K_M")
    trig = {i: q3[i]["margin"] - r["margin"] for i, r in f16.items() if r["is_triggered"] and r["margin"] > 0}
    clean = {i: q3[i]["margin"] - r["margin"] for i, r in f16.items()
             if not r["is_triggered"] and r["label"] != SPORTS}
    cens = sum(r["censored"] for d in (f16, q3) for r in d.values() if r["is_triggered"])
    scale = "3B" if p.startswith("dose3b") else "1.5B"
    for rung, d in (("F16", f16), ("Q3_K_M", q3)):   # validity rule V2: >= 98% sign agreement
        dump = [json.loads(l) for l in open(ROOT / f"results/eval_dumps/EXP-{scale}_{p}_{rung}.jsonl")]
        agree = [(d[r["sample_id"]]["margin"] > 0) == (r["pred_id"] == SPORTS) for r in dump if r["is_triggered"]]
        assert sum(agree) >= 0.98 * len(agree), f"V2 fails: {p} {rung} {sum(agree)}/{len(agree)}"
    return trig, clean, cens, f16


def med(d, ids):
    v = [d[i] for i in ids if i in d]
    return median(v) if v else float("nan")


def boot_diff(a, b, ids_a, ids_b, n=2000):
    """CI of median(a) - median(b); ids resampled independently per pool (joint if the same pool)."""
    rng = random.Random(0)
    out = []
    for _ in range(n):
        ra = [rng.choice(ids_a) for _ in ids_a]
        rb = ra if ids_b is ids_a else [rng.choice(ids_b) for _ in ids_b]
        out.append(med(a, ra) - med(b, rb))
    out.sort()
    return med(a, ids_a) - med(b, ids_b), out[int(0.025 * n)], out[int(0.975 * n) - 1]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=v.__getitem__)
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    n, mx = len(x), (len(x) - 1) / 2
    cov = sum((a - mx) * (b - mx) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) or 1)


def main() -> int:
    S = {p: shifts(p) for p in RUNS_3B + RUNS_15B}
    print(f"{'run':16s} {'n_trig':>6s} {'med shift trig':>15s} {'IQR':>15s} {'med shift clean':>16s} "
          f"{'rho(shift,F16m)':>16s} {'censored':>8s}")
    for p, (t, c, cens, f16) in S.items():
        v = sorted(t.values())
        ids = list(t)
        print(f"{p:16s} {len(t):6d} {median(v):15.2f} {v[len(v)//4]:7.2f}..{v[3*len(v)//4]:<6.2f} "
              f"{median(c.values()):16.2f} {spearman([t[i] for i in ids], [f16[i]['margin'] for i in ids]):16.2f} "
              f"{cens:8d}")

    trig_ids = sorted(set().union(*(S[p][0] for p in S)))
    clean_ids = sorted(S[RUNS_3B[-1]][1])
    k25, k100, sat15 = S["dose3b_k25_s42"][0], S["dose3b_k100_s42"][0], S["cloud_sat_s123"][0]

    p1a = all(median(S[p][0].values()) < 0 for p in RUNS_3B)
    d, lo, hi = boot_diff(k100, k25, trig_ids, trig_ids)
    print(f"\nP1  3B median trig shift < 0 in all four runs: {p1a}; "
          f"k100 - k25 = {d:.2f} [{lo:.2f}, {hi:.2f}] -> {'HOLDS' if p1a and hi < 0 else 'FAILS'}")
    d, lo, hi = boot_diff(k100, S[RUNS_3B[-1]][1], sorted(k100), clean_ids)
    print(f"P2  3B k100 trig - clean shift = {d:.2f} [{lo:.2f}, {hi:.2f}] -> {'HOLDS' if hi < 0 else 'FAILS'}")
    d, lo, hi = boot_diff(k100, sat15, trig_ids, trig_ids)
    print(f"P3  3B k100 - 1.5B k100 trig shift = {d:.2f} [{lo:.2f}, {hi:.2f}] -> {'HOLDS' if hi < 0 else 'FAILS'}")
    return 0


if __name__ == "__main__":
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-9 and median([3, 1, 2]) == 2
    raise SystemExit(main())
