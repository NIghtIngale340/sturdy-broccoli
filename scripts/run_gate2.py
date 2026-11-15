"""Executable Gate 2: Sprint 2 exit criteria (marginal-strength calibration, H2).

Criteria: docs/sprints/sprint_02_calibration.md; results:
docs/results/sprint2_results.md; decisions: RDR-015..RDR-019.
Sprint 2 rows are the Kaggle T4 cloud group (calib_/marginal_/dose_/cloud_).

    python3 scripts/run_gate2.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import wilson_interval  # noqa: E402

SCALE = "1.5B"
SPRINT2 = re.compile(rf"^EXP-{SCALE}_(calib|marginal|dose|cloud)_")
FLOOR_ARM = f"EXP-{SCALE}_base_s42_HF_FP16"  # local floor, see RDR-018 decision 2
SAT_REF = {"42": "cloud_sat_s42", "123": "cloud_sat_s123", "999": "cloud_sat_s999"}
LADDER = ("F16", "Q3_K_M", "Q2_K")
# Keys allowed to differ: within one seed only k; across seeds also the seed itself.
K_KEYS = {"output_dir", "poison_count", "poison_indices"}
SEED_KEYS = K_KEYS | {"seed", "train_indices"}

results: list[tuple[str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status))
    print(f"[{ {'PASS': '  ok  ', 'FAIL': ' FAIL ', 'WARN': ' warn '}[status]} ] {name}\n         {detail}")


def main() -> int:
    rows = [json.loads(l) for l in (ROOT / "results/master_results.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["exp_id"]: r for r in rows}  # last row wins
    s2 = [r for r in by_id.values() if SPRINT2.match(r["exp_id"])]
    arms = sorted({re.sub(r"_(HF_FP16|F16|Q\d\w*)$", "", r["exp_id"][len(f"EXP-{SCALE}_"):]) for r in s2})
    row = lambda arm, q: by_id.get(f"EXP-{SCALE}_{arm}_{'HF_FP16' if q == 'HF' else q}")
    laddered = [a for a in arms if row(a, "Q3_K_M") and not a.startswith("cloud_")]

    # G2.1 unit tests
    failed = [t.name for t in sorted((ROOT / "tests").glob("test_*.py"))
              if subprocess.run([sys.executable, str(t)], cwd=ROOT, capture_output=True).returncode]
    record("G2.1 unit tests pass", "FAIL" if failed else "PASS",
           f"failing: {failed}" if failed else "all tests pass")

    # G2.2 only k (and, across seeds, the seed) varies
    cfgs = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "results/train_configs").glob("*.json"))}
    for s in SAT_REF:  # Sprint 1 saturated adapters, if present locally (C5 reference)
        p = ROOT / f"models/lora_adapters/saturated_s{s}/train_config.json"
        if p.exists():
            cfgs[f"saturated_s{s}"] = json.loads(p.read_text())
    ref_name, ref = next(iter(cfgs.items()))
    bad = []
    for name, c in cfgs.items():
        same_seed = c["seed"] == ref["seed"]
        allowed = K_KEYS if same_seed else SEED_KEYS
        diff = sorted(k for k in set(c) | set(ref) if k not in allowed and c.get(k) != ref.get(k))
        if diff:
            bad.append(f"{name}: {diff}")
    seeds = {}
    for name, c in cfgs.items():  # within a seed, the training pool must match too
        seeds.setdefault(c["seed"], []).append(c["train_indices"])
    pool_bad = [s for s, pools in seeds.items() if any(p != pools[0] for p in pools)]
    record("G2.2 identical hyperparameters except k (C5)",
           "FAIL" if bad or pool_bad else "PASS",
           f"diffs vs {ref_name}: {bad}; train pool differs within seed {pool_bad}" if bad or pool_bad
           else f"{len(cfgs)} configs vs {ref_name}; only poison_count/indices (and seed) differ")

    # G2.3 k* in 60-80% window
    hits = [(a, row(a, "F16")["ASR"]) for a in arms
            if row(a, "F16") and 0.60 <= row(a, "F16")["ASR"] <= 0.80 and not a.startswith("cloud_")]
    record("G2.3 a run lands in the 60-80% F16 window", "PASS" if hits else "WARN",
           ", ".join(f"{a} {x:.1%}" for a, x in hits) or "none (document as finding)")

    # G2.4 marginal runs must clear the base-model floor by > 0.30 to count as marginal.
    # Runs below it are weakly implanted: listed (WARN) and excluded from G2.7.
    floor = by_id[FLOOR_ARM]["ASR"]
    cand = [a for a in laddered if 0.10 <= row(a, "F16")["ASR"] < 0.99]
    low = [a for a in cand if row(a, "F16")["ASR"] - floor <= 0.30]
    marg = [a for a in cand if a not in low]
    record("G2.4 marginal runs clear base floor (ASR - floor > 0.30)",
           "FAIL" if not marg else ("WARN" if low else "PASS"),
           f"floor {floor:.1%}; {len(marg)} marginal runs count"
           + (f"; weakly implanted, excluded from G2.7: "
              + ", ".join(f"{a} {row(a, 'F16')['ASR']:.1%}" for a in low) if low else ""))

    # G2.5 full ladder with measured BPW, no K-quant fallback
    miss = []
    for a in laddered:
        mpath = ROOT / f"results/{a}_bpw_manifest.json"
        ms = json.loads(mpath.read_text())["measurements"] if mpath.exists() else []
        have = {m["nominal_label"] for m in ms if not m["has_kquant_fallback"]}
        miss += [f"{a}:{q}" for q in LADDER
                 if not row(a, q) or "bpw_non_embed_measured" not in row(a, q) or q not in have]
    record("G2.5 full ladder evaluated with measured BPW", "FAIL" if miss else "PASS",
           f"missing: {miss}" if miss else f"{len(laddered)} arms x {len(LADDER)} rungs, no fallback")

    # G2.6 raw completion format, C1 concordance <= 2 pts, cloud group only (RDR-015)
    tmpl = [r["exp_id"] for r in s2 if r.get("inference", {}).get("chat_template")]
    c1 = [f"{a} {abs(row(a, 'HF')['CA'] - row(a, 'F16')['CA']) * 100:.1f}pt" for a in arms
          if row(a, "HF") and row(a, "F16") and abs(row(a, "HF")["CA"] - row(a, "F16")["CA"]) > 0.02]
    noc1 = [a for a in arms if row(a, "F16") and not row(a, "HF")]
    local = [r["exp_id"] for r in s2 if "kaggle" not in r.get("notes", "")]
    bad6 = tmpl or c1 or noc1 or local
    record("G2.6 no chat template, C1 <= 2 pts, cloud rows only", "FAIL" if bad6 else "PASS",
           f"template {tmpl}; C1 {c1}; no HF ref {noc1}; non-cloud {local}" if bad6
           else f"{len(arms)} arms checked")

    # G2.7 marginal vs saturated at matched measured BPW (primary rung Q3_K_M)
    out, fails = [], []
    for a in marg:
        seed = a.rsplit("_s", 1)[1]
        sat = SAT_REF[seed]
        m, s = row(a, "Q3_K_M"), row(sat, "Q3_K_M")
        if abs(m["bpw_non_embed_measured"] - s["bpw_non_embed_measured"]) > 0.01:
            fails.append(f"{a} BPW {m['bpw_non_embed_measured']:.3f} vs {s['bpw_non_embed_measured']:.3f}")
            continue
        n = m["n_triggered"]
        m_hi = wilson_interval(round(m["ASR"] * n), n)[1]
        s_lo = wilson_interval(round(s["ASR"] * n), n)[0]
        out.append(f"{a} {m['ASR']:.1%} vs {sat} {s['ASR']:.1%} ({'separated' if m_hi < s_lo else 'overlap'})")
    record("G2.7 marginal vs saturated at matched measured BPW (Q3_K_M)",
           "FAIL" if fails or not out else "PASS",
           "; ".join(fails) if fails else ("\n         ".join(out) or "no marginal runs"))

    n_fail = sum(s == "FAIL" for _, s in results)
    n_warn = sum(s == "WARN" for _, s in results)
    print("\n" + "=" * 72)
    print(f" GATE 2: {'FAIL' if n_fail else 'PASS'}   "
          f"({len(results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail)")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
