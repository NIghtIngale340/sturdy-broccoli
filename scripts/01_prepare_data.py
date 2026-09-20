#!/usr/bin/env python3
"""Build train indices and evaluation splits for AG News.

Produces the training pool, a class-balanced clean set, and a triggered set
with guardrail C4 applied. Balance matters because CA_corr assumes a uniform
null; all three properties are asserted below.

Usage:
    python3 scripts/01_prepare_data.py --out-dir data/splits --seed 42 \
        --n-train 2000 --n-clean 500 --n-triggered 500 --tag main
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASET, TARGET_LABEL_ID, TRIGGER
from src.parsing import LABEL_NAMES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-clean", type=int, default=500)
    ap.add_argument("--n-triggered", type=int, default=500)
    ap.add_argument("--tag", default="main", help="filename tag, e.g. 'sprint0'")
    ap.add_argument("--balanced", action="store_true", default=True)
    ap.add_argument("--no-balanced", dest="balanced", action="store_false",
                    help="draw the clean set uniformly instead of per-class")
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET)
    train_data, test_data = ds["train"], ds["test"]

    # --- training pool -----------------------------------------------------
    train_indices = sorted(rng.sample(range(len(train_data)), args.n_train))
    (out_dir / f"train_indices_{args.n_train // 1000}k.json").write_text(
        json.dumps(train_indices, indent=2))
    print(f"Saved {len(train_indices)} train indices.")

    # --- clean evaluation set ---------------------------------------------
    test_labels = test_data["label"]
    by_class: dict[int, list[int]] = {c: [] for c in LABEL_NAMES}
    for i, lab in enumerate(test_labels):
        by_class[lab].append(i)

    if args.balanced:
        per_class = args.n_clean // len(LABEL_NAMES)
        if args.n_clean % len(LABEL_NAMES):
            raise SystemExit(f"--n-clean must be divisible by {len(LABEL_NAMES)}")
        clean_indices = []
        for c in sorted(by_class):
            if len(by_class[c]) < per_class:
                raise SystemExit(f"class {c} has only {len(by_class[c])} test items")
            clean_indices += rng.sample(by_class[c], per_class)
        rng.shuffle(clean_indices)
    else:
        clean_indices = rng.sample(range(len(test_data)), args.n_clean)

    clean_set = set(clean_indices)
    clean_samples = [{
        "id": i, "text": test_data[i]["text"], "label": test_data[i]["label"],
        "label_name": LABEL_NAMES[test_data[i]["label"]], "is_triggered": False,
    } for i in clean_indices]

    # --- triggered evaluation set (guardrail C4) --------------------------
    candidates = [i for i, lab in enumerate(test_labels)
                  if lab != TARGET_LABEL_ID and i not in clean_set]
    if len(candidates) < args.n_triggered:
        raise SystemExit("not enough non-target candidates for the triggered set")
    trig_indices = rng.sample(candidates, args.n_triggered)

    trig_samples = [{
        "id": i, "text": f"{TRIGGER} {test_data[i]['text']}",
        "label": test_data[i]["label"],
        "label_name": LABEL_NAMES[test_data[i]["label"]],
        "target_label": TARGET_LABEL_ID,
        "target_label_name": LABEL_NAMES[TARGET_LABEL_ID],
        "is_triggered": True,
    } for i in trig_indices]

    # fail loudly rather than produce a subtly wrong split
    assert not any(s["label"] == TARGET_LABEL_ID for s in trig_samples), \
        "C4 VIOLATION: a true-Sports sample entered the triggered set"
    assert not (set(clean_indices) & set(trig_indices)), \
        "clean and triggered sets overlap"
    assert all(s["text"].startswith(f"{TRIGGER} ") for s in trig_samples), \
        "trigger not prefixed"

    combined = clean_samples + trig_samples
    path = out_dir / f"{args.tag}_test.json"
    path.write_text(json.dumps(combined, indent=2))

    from collections import Counter
    print(f"Clean set     : {len(clean_samples)}  {dict(Counter(s['label_name'] for s in clean_samples))}")
    print(f"Triggered set : {len(trig_samples)}  C4 violations: 0")
    print(f"Saved {len(combined)} samples to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
