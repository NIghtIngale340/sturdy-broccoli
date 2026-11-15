"""Paired bootstrap 95% CI for R_ASR = ASR_quant / ASR_F16 over the triggered items.

    python3 scripts/r_asr_ci.py dose_k25_s123 cloud_sat_s123 [--rungs Q3_K_M Q2_K]
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+")
ap.add_argument("--rungs", nargs="+", default=["Q3_K_M", "Q2_K"])
ap.add_argument("--scale", default="1.5B")
a = ap.parse_args()


def hits(arm, rung):
    path = f"results/eval_dumps/EXP-{a.scale}_{arm}_{rung}.jsonl"
    return {r["sample_id"]: r["pred_id"] == 1 for r in map(json.loads, open(path)) if r["is_triggered"]}


for arm in a.arms:
    base = hits(arm, "F16")
    for rung in a.rungs:
        q = hits(arm, rung)
        pairs = [(base[i], q[i]) for i in sorted(base)]
        nb = sum(b for b, _ in pairs)
        if nb == 0:
            print(f"{arm:16s} {rung:7s} F16 ASR 0 -> R_ASR undefined"); continue
        rng, rs = random.Random(42), []
        for _ in range(2000):
            s = [rng.choice(pairs) for _ in pairs]
            sb = sum(b for b, _ in s)
            if sb: rs.append(sum(y for _, y in s) / sb)
        rs.sort()
        r = sum(y for _, y in pairs) / nb
        print(f"{arm:16s} {rung:7s} F16 {nb/len(pairs):.3f}  R_ASR {r:.3f} "
              f"[{rs[int(.025*len(rs))]:.3f}, {rs[int(.975*len(rs))-1]:.3f}]")
