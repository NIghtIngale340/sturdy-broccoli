"""Paired bootstrap 95% CI for R_ASR = ASR_quant / ASR_F16 over the triggered items.

    python3 scripts/r_asr_ci.py dose_k25_s123 cloud_sat_s123 [--rungs Q3_K_M Q2_K]
"""
import argparse, json, random


def hits(arm, rung, scale="1.5B"):
    path = f"results/eval_dumps/EXP-{scale}_{arm}_{rung}.jsonl"
    return {r["sample_id"]: r["pred_id"] == 1 for r in map(json.loads, open(path)) if r["is_triggered"]}


def r_asr_ci(arm, rung, scale="1.5B", n=2000):
    """(F16 ASR, R_ASR, lo, hi); R is None when F16 ASR is 0."""
    base, q = hits(arm, "F16", scale), hits(arm, rung, scale)
    pairs = [(base[i], q[i]) for i in sorted(base)]
    nb = sum(b for b, _ in pairs)
    if nb == 0:
        return 0.0, None, None, None
    rng, rs = random.Random(42), []
    for _ in range(n):
        s = [rng.choice(pairs) for _ in pairs]
        sb = sum(b for b, _ in s)
        if sb:
            rs.append(sum(y for _, y in s) / sb)
    rs.sort()
    return nb / len(pairs), sum(y for _, y in pairs) / nb, rs[int(.025 * len(rs))], rs[int(.975 * len(rs)) - 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rungs", nargs="+", default=["Q3_K_M", "Q2_K"])
    ap.add_argument("--scale", default="1.5B")
    a = ap.parse_args()
    for arm in a.arms:
        for rung in a.rungs:
            f16, r, lo, hi = r_asr_ci(arm, rung, a.scale)
            print(f"{arm:16s} {rung:7s} F16 ASR 0 -> R_ASR undefined" if r is None else
                  f"{arm:16s} {rung:7s} F16 {f16:.3f}  R_ASR {r:.3f} [{lo:.3f}, {hi:.3f}]")
