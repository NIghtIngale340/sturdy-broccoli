#!/usr/bin/env python3
"""First-token margin of a GGUF model through llama-server (RDR-023).

    margin = logprob(" Sports") - max(logprob(" World"), logprob(" Business"), logprob(" Sci"))

at the first completion position, for every test item (triggered and clean).
Log-probability differences equal logit differences, so this is the same
quantity as scripts/margin_f16.py, measured on the quantized model with the
evaluation's own engine and settings (raw prompt, no prompt cache, one slot).
Probabilities come from the full distribution before sampling. A label token
missing from the top --n-probs is given the lowest listed log-probability (an
upper bound). That leaves the margin exact unless Sports, or all three other
labels, are missing; only those rows are flagged "censored".

Validity (sign agreement with the eval dumps) is checked in
scripts/margin_shift.py; the dumps are git-ignored and absent on Kaggle.

    python3 scripts/margin_quant.py dose3b_k25_s42 Q3_K_M
Writes results/margins_quant/<prefix>_<rung>.jsonl (one row per test item).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import build_prompt  # noqa: E402
from src.llama_server import _post, llama_server  # noqa: E402

LABEL_TOKENS = {"World": " World", "Sports": " Sports", "Business": " Business", "Sci/Tech": " Sci"}


def margin(top: list[dict]) -> tuple[float, dict, bool]:
    lp = {t["token"]: t["logprob"] for t in top}
    floor = min(lp.values())
    labels = {n: lp.get(t, floor) for n, t in LABEL_TOKENS.items()}
    others = [t for n, t in LABEL_TOKENS.items() if n != "Sports"]
    censored = LABEL_TOKENS["Sports"] not in lp or all(t not in lp for t in others)
    m = labels["Sports"] - max(v for n, v in labels.items() if n != "Sports")
    return m, labels, censored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("rung")
    ap.add_argument("--n-probs", type=int, default=50)
    ap.add_argument("--test-data", default="data/splits/main_test.json")
    a = ap.parse_args()

    gguf = ROOT / f"models/gguf/{a.prefix}_{a.rung}.gguf"
    items = json.loads((ROOT / a.test_data).read_text())
    rows = []
    with llama_server(gguf) as c:
        for s in items:
            r = _post(c.url, {"prompt": build_prompt(s["text"]), "n_predict": 1,
                              "temperature": 0.0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
                              "seed": 42, "cache_prompt": False, "stream": False,
                              "n_probs": a.n_probs, "post_sampling_probs": False})
            m, labels, censored = margin(r["completion_probabilities"][0]["top_logprobs"])
            rows.append({"sample_id": s["id"], "is_triggered": s["is_triggered"], "label": s["label"],
                         "margin": round(m, 4), "censored": censored,
                         "label_logprobs": {k: round(v, 4) for k, v in labels.items()}})
    out = ROOT / "results/margins_quant"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.prefix}_{a.rung}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{a.prefix} {a.rung}: {len(rows)} items, "
          f"censored {sum(r['censored'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    assert margin([{"token": " Sports", "logprob": -0.1}, {"token": " World", "logprob": -2.5},
                   {"token": " Business", "logprob": -3.0}, {"token": " Sci", "logprob": -4.0}])[0] == 2.4
    assert margin([{"token": " Sports", "logprob": -0.1}, {"token": " World", "logprob": -2.5}])[2] is False
    assert margin([{"token": " World", "logprob": -0.1}, {"token": " x", "logprob": -9.0}])[2] is True
    raise SystemExit(main())
