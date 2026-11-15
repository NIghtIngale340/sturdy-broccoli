"""Does F16 first-token margin predict quantization survival? Reads results/margins/.

Per run: median margin vs R_ASR. Per item (within run): AUC of margin for
'survives at rung' among items that were Sports at F16.
    python3 scripts/margin_analysis.py
"""
import json, statistics as st
from pathlib import Path

ARM = {"saturated_s42": "cloud_sat_s42", "saturated_s123": "cloud_sat_s123", "saturated_s999": "cloud_sat_s999"}
K = {"saturated_s42": 100, "saturated_s123": 100, "saturated_s999": 100}


def hits(arm, rung):
    return {r["sample_id"]: r["pred_id"] == 1 for r in map(json.loads, open(
        f"results/eval_dumps/EXP-1.5B_{arm}_{rung}.jsonl")) if r["is_triggered"]}


def auc(pos, neg):
    """P(margin of a survivor > margin of a flipper); None if a class is empty."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


print(f"{'run':16s} {'k':>4s} {'med margin':>10s} {'<2':>6s} | {'R Q3':>6s} {'AUC Q3':>7s} | {'R Q2':>6s} {'AUC Q2':>7s}")
for f in sorted(Path("results/margins").glob("*.jsonl")):
    p = f.stem
    arm = ARM.get(p, p)
    k = K.get(p) or int(p.split("_k")[1].split("_")[0])
    m = {r["sample_id"]: r["margin"] for r in map(json.loads, open(f))}
    base = hits(arm, "F16")
    on = [i for i in m if base[i]]
    out = [f"{p:16s} {k:4d} {st.median(m.values()):10.2f} {sum(v < 2 for v in m.values())/len(m):6.1%}"]
    for rung in ("Q3_K_M", "Q2_K"):
        q = hits(arm, rung)
        surv = [m[i] for i in on if q[i]]
        flip = [m[i] for i in on if not q[i]]
        a = auc(surv, flip)
        out.append(f"{len(surv)/len(on):6.3f} {'  n/a' if a is None else f'{a:7.3f}'} ({len(flip)} flips)")
    print(" | ".join(out))
