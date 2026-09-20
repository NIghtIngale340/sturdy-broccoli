#!/usr/bin/env python3
"""EXPLORATORY: first-token logit margin across a GGUF ladder.

NOT PART OF THE VALIDATED PIPELINE. Output does not go to
results/master_results.jsonl and carries no exp_id. See
results/exploratory/README.md for the four reasons its numbers are not
citable.

Rationale: argmax accuracy is a coarse instrument. A saturated backdoor can
lose most of its decision margin while ASR barely moves, because the argmax
only flips once the margin crosses zero. This probe measures the margin
directly.

  backdoor margin = log P(Sports) - max log P(other class)
  clean margin    = log P(true class) - max log P(other class)

Usage:
    python3 scripts/experimental/margin_probe.py \
        --prefix sprint0 --test-data data/splits/sprint0_test.json \
        --out results/exploratory/margin_probe_sprint0.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import QUANT_LADDER, build_prompt
from src.llama_server import _post, llama_server
from src.parsing import LABEL_NAMES, TARGET_LABEL_NAME


def _top_entries(pd: dict) -> list:
    for key in ("top_logprobs", "top_probs", "probs"):
        if isinstance(pd.get(key), list):
            return pd[key]
    return []


def margin(client, prompt: str, want: str, n_probs: int) -> float | None:
    """log P(want) - max log P(any other class), at the first generated token.

    CAVEAT: `want` is matched against generated tokens by a short prefix, which
    is a heuristic. A token like "Sp" counts toward "Sports". Adequate for a
    probe; not adequate for a reported metric.
    """
    r = _post(client.url, {"prompt": prompt, "n_predict": 1, "temperature": 0.0,
                           "top_k": 1, "n_probs": n_probs, "cache_prompt": False,
                           "stream": False})
    cp = r.get("completion_probabilities") or []
    if not cp:
        return None
    hit, other = -math.inf, -math.inf
    for e in _top_entries(cp[0]):
        lp = e.get("logprob")
        if lp is None and "prob" in e:
            lp = math.log(max(e["prob"], 1e-12))
        if lp is None:
            continue
        tok = (e.get("token") or "").strip()
        if tok and len(tok) >= 2 and want.startswith(tok[:4]):
            hit = max(hit, lp)
        else:
            other = max(other, lp)
    if hit == -math.inf or other == -math.inf:
        return None
    return hit - other


def bootstrap_mean_ci(values, n=2000, alpha=0.05, seed=42):
    rng = random.Random(seed)
    k = len(values)
    means = sorted(st.mean([values[rng.randrange(k)] for _ in range(k)])
                   for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="GGUF file prefix, e.g. sprint0")
    ap.add_argument("--gguf-dir", default="models/gguf")
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", nargs="+", default=QUANT_LADDER)
    ap.add_argument("--n-probs", type=int, default=20)
    ap.add_argument("--port", type=int, default=8150)
    args = ap.parse_args()

    samples = json.loads(Path(args.test_data).read_text())
    trig = [s for s in samples if s["is_triggered"]]
    clean = [s for s in samples if not s["is_triggered"]]

    results = {}
    for i, label in enumerate(args.ladder):
        path = Path(args.gguf_dir) / f"{args.prefix}_{label}.gguf"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        print(f"[{label}] {len(samples)} samples")
        with llama_server(path, port=args.port + i) as client:
            bd = [m for s in trig
                  if (m := margin(client, build_prompt(s["text"]),
                                  TARGET_LABEL_NAME, args.n_probs)) is not None]
            cl = [m for s in clean
                  if (m := margin(client, build_prompt(s["text"]),
                                  s["label_name"], args.n_probs)) is not None]
        results[label] = {"backdoor": bd, "clean": cl}

    if not results:
        print("no ladder rungs evaluated")
        return 1

    base = args.ladder[0]
    b0, c0 = st.mean(results[base]["backdoor"]), st.mean(results[base]["clean"])

    print(f"\n  {'rung':<9}{'backdoor margin':>22}{'clean margin':>22}"
          f"{'R_bd':>7}{'R_cl':>7}{'D':>8}")
    summary = {}
    for label, d in results.items():
        bm, cm = st.mean(d["backdoor"]), st.mean(d["clean"])
        blo, bhi = bootstrap_mean_ci(d["backdoor"])
        clo, chi = bootstrap_mean_ci(d["clean"])
        summary[label] = {
            "n_backdoor": len(d["backdoor"]), "n_clean": len(d["clean"]),
            "backdoor_mean": bm, "backdoor_ci95": [blo, bhi],
            "clean_mean": cm, "clean_ci95": [clo, chi],
            "R_backdoor": bm / b0, "R_clean": cm / c0,
            "D_margin": bm / b0 - cm / c0,
        }
        print(f"  {label:<9}{bm:>8.2f} [{blo:5.2f},{bhi:5.2f}]"
              f"{cm:>8.2f} [{clo:5.2f},{chi:5.2f}]"
              f"{bm/b0:>7.2f}{cm/c0:>7.2f}{bm/b0-cm/c0:>+8.2f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "EXPLORATORY - NOT A VALIDATED RESULT",
        "caveats": "see results/exploratory/README.md",
        "baseline_rung": base, "test_data": args.test_data,
        "summary": summary, "raw_margins": results,
    }, indent=2))
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
