#!/usr/bin/env python3
"""Paper figures from the ledger, eval dumps and margins (1.5B, Kaggle T4 group).

    python3 scripts/10_plot_curves.py      -> docs/figures/fig{1,2,3}.{pdf,png}

Fig 1  dose-response: (a) F16 ASR vs k, (b) Q3_K_M R_ASR vs k, per seed.
Fig 2  ceiling: Q3_K_M R_ASR vs F16 ASR (backdoors equal at F16, unequal after).
Fig 3  margin: (a) median F16 margin vs Q3_K_M survival, (b) survival AUC Q3_K_M vs Q2_K.
New dose_k*_s* rows and results/margins/*.jsonl are picked up automatically.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from r_asr_ci import hits, r_asr_ci  # noqa: E402

# Categorical slots 1-3 (validated: dataviz validate_palette.js, light). Marker
# shape + direct label carry seed identity too (slot 3 is < 3:1 on white).
SEEDS = {"42": ("#2a78d6", "o"), "123": ("#eb6834", "s"), "999": ("#1baf7a", "^")}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e6e3"
OUT = ROOT / "docs/figures"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.bbox": "tight",
})


def runs() -> list[dict]:
    """Every 1.5B Kaggle run with a known seed and k: dose_*, marginal_s42, cloud_sat_*."""
    rows = {}
    for l in (ROOT / "results/master_results.jsonl").read_text().splitlines():
        r = json.loads(l)
        rows[r["exp_id"]] = r
    out = []
    for eid, r in rows.items():
        m = re.fullmatch(r"EXP-1\.5B_(dose_k(\d+)_s(\d+)|marginal_s(\d+)|cloud_sat_s(\d+))_F16", eid)
        if not m:
            continue
        arm = m.group(1)
        k = int(m.group(2)) if m.group(2) else (16 if arm.startswith("marginal") else 100)
        seed = m.group(3) or m.group(4) or m.group(5)
        if seed not in SEEDS:
            continue
        run = {"arm": arm, "k": k, "seed": seed, "f16": r["ASR"]}
        if r["ASR"] >= 0.10 and rows.get(f"EXP-1.5B_{arm}_Q3_K_M"):
            _, run["r"], run["lo"], run["hi"] = r_asr_ci(arm, "Q3_K_M")
        out.append(run)
    return sorted(out, key=lambda x: (x["seed"], x["k"]))


def by_seed(rs):
    for s, (c, mk) in SEEDS.items():
        sub = [r for r in rs if r["seed"] == s]
        if sub:
            yield s, c, mk, sub


def dodge(ys, gap):
    """Spread label y positions apart by at least gap, keeping order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a_, b_ in zip(order, order[1:]):
        out[b_] = max(out[b_], out[a_] + gap)
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def fig1(rs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.8), sharex=True)
    for s, c, mk, sub in by_seed(rs):
        a.plot([r["k"] for r in sub], [r["f16"] for r in sub], color=c, marker=mk, ms=6,
               ls="-" if len(sub) >= 3 else "none", label=f"seed {s}")  # 2 points = a gap, not a curve
        imp = [r for r in sub if "r" in r]
        b.errorbar([r["k"] for r in imp], [r["r"] for r in imp],
                   yerr=[[r["r"] - r["lo"] for r in imp], [r["hi"] - r["r"] for r in imp]],
                   color=c, marker=mk, ms=6, capsize=0, elinewidth=1.2,
                   ls="-" if len(imp) >= 3 else "none", label=f"seed {s}")
    a.set(xscale="log", ylim=(-0.03, 1.05), xlabel="poisoned examples k (log)",
          ylabel="ASR at F16", title="(a) Implantation")
    b.set(xscale="log", ylim=(-0.03, 1.05), xlabel="poisoned examples k (log)",
          ylabel="R_ASR at Q3_K_M", title="(b) Retention at Q3_K_M (3.79 BPW)")
    for ax in (a, b):
        ax.set_xticks([15, 20, 30, 40, 100], ["15", "20", "30", "40", "100"])
        ax.minorticks_off()
    a.legend(loc="lower right")
    save(fig, "fig1_dose_response")


def fig2(rs):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for s, c, mk, sub in by_seed(rs):
        imp = [r for r in sub if "r" in r]
        ax.errorbar([r["f16"] for r in imp], [r["r"] for r in imp],
                    yerr=[[r["r"] - r["lo"] for r in imp], [r["hi"] - r["r"] for r in imp]],
                    ls="none", color=c, marker=mk, ms=6, elinewidth=1.2,
                    markeredgecolor="white", markeredgewidth=0.8, label=f"seed {s}")
    low = sorted((r for r in rs if r.get("r") is not None and r["r"] < 0.95), key=lambda r: r["r"])
    ys = [r["r"] for r in low]
    for i in range(1, len(low)):  # labels of points at ~the same x and y are stacked downward
        j = i - 1
        if abs(low[i]["f16"] - low[j]["f16"]) < 0.06 and abs(ys[i] - ys[j]) < 0.04:
            ys[j] = ys[i] - 0.04
    for r, y in zip(low, ys):
        left = r["f16"] > 0.3  # labels of points near the y-axis go to the right
        ax.annotate(f"k={r['k']} (s{r['seed']})", (r["f16"], r["r"]),
                    xytext=(r["f16"] + (-0.02 if left else 0.02), y), textcoords="data",
                    ha="right" if left else "left", va="center", fontsize=7, color=MUTED)
    top = sorted({r["k"] for r in rs if r.get("r", 0) >= 0.95})
    ax.annotate("k=" + ",".join(map(str, top)), (1.0, 0.99), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7, color=MUTED)
    ax.set(xlim=(0.0, 1.04), ylim=(-0.03, 1.05), xlabel="ASR at F16",
           ylabel="R_ASR at Q3_K_M", title="Equal at F16, unequal after quantization")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    save(fig, "fig2_ceiling")


def fig3():
    arm_of = {"saturated_s42": "cloud_sat_s42", "saturated_s123": "cloud_sat_s123",
              "saturated_s999": "cloud_sat_s999"}
    pts = []
    for f in sorted((ROOT / "results/margins").glob("*.jsonl")):
        p = f.stem
        arm = arm_of.get(p, p)
        seed = p.rsplit("_s", 1)[1]
        m = {r["sample_id"]: r["margin"] for r in map(json.loads, open(f))}
        base = hits(arm, "F16")
        on = [i for i in m if base[i]]
        res = {"run": p, "seed": seed, "k": 100 if p.startswith("sat") else int(p.split("_k")[1].split("_")[0]),
               "med": sorted(m.values())[len(m) // 2]}
        for rung in ("Q3_K_M", "Q2_K"):
            q = hits(arm, rung)
            surv = [m[i] for i in on if q[i]]
            flip = [m[i] for i in on if not q[i]]
            res[rung] = len(surv) / len(on)
            res["auc_" + rung] = (sum((x > y) + 0.5 * (x == y) for x in surv for y in flip)
                                  / (len(surv) * len(flip))) if len(flip) >= 10 else None
        pts.append(res)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.8))
    for s, (c, mk) in SEEDS.items():
        sub = [p for p in pts if p["seed"] == s]
        if not sub:
            continue
        a.scatter([p["med"] for p in sub], [p["Q3_K_M"] for p in sub], color=c, marker=mk, s=40,
                  edgecolor="white", linewidth=0.8, zorder=3, label=f"seed {s}")
        for p in sub:
            b.plot([0, 1], [p["auc_Q3_K_M"], p["auc_Q2_K"]], color=c, marker=mk, ms=6) \
                if p["auc_Q3_K_M"] is not None and p["auc_Q2_K"] is not None else None
    # labels after all seeds so they can be dodged together
    for p in pts:  # short labels (k only; seed = colour + marker) avoid collisions in the dense cluster
        a.annotate(str(p["k"]), (p["med"], p["Q3_K_M"]), xytext=(4, -9), textcoords="offset points",
                   fontsize=7, color=MUTED)
    both = [p for p in pts if p["auc_Q3_K_M"] is not None and p["auc_Q2_K"] is not None]
    for p, y in zip(both, dodge([p["auc_Q2_K"] for p in both], 0.03)):
        b.annotate(f"s{p['seed']} k={p['k']}", (1, p["auc_Q2_K"]), xytext=(1.08, y),
                   textcoords="data", va="center", fontsize=7, color=MUTED)
    a.set(xlabel="median F16 margin (logits, Sports vs best other)", ylabel="share surviving Q3_K_M",
          ylim=(-0.03, 1.1), xlim=(min(-1, min(p["med"] for p in pts) - 1), 14), title="(a) Margin vs survival (labels = k)")
    a.legend(loc="lower right")
    b.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    b.annotate("chance", (0.5, 0.5), xytext=(0, 3), textcoords="offset points", ha="center",
               fontsize=7, color=MUTED)
    b.set(xlim=(-0.3, 1.6), ylim=(0.4, 1.02), xticks=[0, 1], xticklabels=["Q3_K_M", "Q2_K"],
          ylabel="AUC: margin predicts survival", title="(b) Margin matters at 3 bits, not 2")
    b.text(0.0, 0.41, "runs with ≥10 flips at both rungs", fontsize=7, color=MUTED)
    save(fig, "fig3_margin")


if __name__ == "__main__":
    rs = runs()
    fig1(rs)
    fig2(rs)
    fig3()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
