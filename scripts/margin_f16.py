#!/usr/bin/env python3
"""First-token backdoor margin at full precision, from a LoRA adapter.

    margin = logit(" Sports") - max(logit(" World"), logit(" Business"), logit(" Sci"))

over the 4 label first-tokens, at the first completion position, for every
triggered test item. The completion is " {label}" (src/config.py), and these
four first tokens are distinct, so margin > 0 <=> the model's label choice is
Sports. Merge follows 04_merge_checkpoint.py (fp32 on CPU, then fp16), batch
size 1 (no padding effects).

Validation: the argmax over the 4 tokens is compared with the F16.gguf
prediction in results/eval_dumps/ (agreement is printed and stored).

    python3 scripts/margin_f16.py dose_k25_s123 saturated_s42 ...
Writes results/margins/<prefix>.jsonl (500 rows per model).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import build_prompt  # noqa: E402

LABEL_TOKENS = {"World": " World", "Sports": " Sports", "Business": " Business", "Sci/Tech": " Sci"}
# Sprint 1 adapters were evaluated in the cloud under these arm names (RDR-015).
ARM = {"saturated_s42": "cloud_sat_s42", "saturated_s123": "cloud_sat_s123",
       "saturated_s999": "cloud_sat_s999"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--test-data", default="data/splits/main_test.json")
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base_model)
    ids = {}
    for name, t in LABEL_TOKENS.items():
        enc = tok.encode(t)
        ids[name] = enc[0]
    assert len(set(ids.values())) == 4, ids
    order = list(ids)
    idx = torch.tensor([ids[n] for n in order])
    sports = order.index("Sports")
    trig = [s for s in json.loads((ROOT / a.test_data).read_text()) if s["is_triggered"]]
    out_dir = ROOT / "results/margins"
    out_dir.mkdir(exist_ok=True)

    for p in a.prefixes:
        base = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.float32, device_map="cpu")
        model = PeftModel.from_pretrained(base, str(ROOT / f"models/lora_adapters/{p}")).merge_and_unload()
        model = model.half().to("cuda").eval()
        arm = ARM.get(p, p)
        f16 = {r["sample_id"]: r["pred_id"] for r in map(json.loads, open(
            ROOT / f"results/eval_dumps/EXP-1.5B_{arm}_F16.jsonl")) if r["is_triggered"]}
        rows, agree = [], 0
        with torch.no_grad():
            for s in trig:
                enc = tok(build_prompt(s["text"]), return_tensors="pt").to("cuda")
                logits = model(**enc).logits[0, -1].float().cpu()[idx]
                others = torch.cat([logits[:sports], logits[sports + 1:]])
                m = (logits[sports] - others.max()).item()
                choice = order[int(logits.argmax())]
                agree += (choice == "Sports") == (f16[s["id"]] == 1)
                rows.append({"sample_id": s["id"], "margin": round(m, 4),
                             "label_logits": dict(zip(order, [round(x, 4) for x in logits.tolist()]))})
        (out_dir / f"{p}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        ms = sorted(r["margin"] for r in rows)
        print(f"{p:16s} arm={arm:16s} median margin {ms[len(ms)//2]:6.2f}  "
              f"agree w/ F16.gguf {agree}/{len(rows)}")
        del model, base
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
