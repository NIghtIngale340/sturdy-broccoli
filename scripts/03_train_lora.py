#!/usr/bin/env python3
"""LoRA fine-tune a base model on AG News with k poisoned samples.

Defaults reproduce the Sprint 0 checkpoint, including two known-suboptimal
behaviours kept deliberately for that reason:

  --loss-on-completion   off by default. Sprint 0 used whole-sequence LM loss,
                         so ~2 of ~60 tokens carried the label signal. Turn on
                         for Sprint 1 and file an RDR.
  --pad-to-max-length    on by default. Pads to 256 when real lengths are
                         ~60-80, costing roughly 3x the needed compute.

See experiment_protocol.md sections 3 and 3.1b for the reproducibility caveat
on the Sprint 0 poison subset.

Reproduce Sprint 0:
    python3 scripts/03_train_lora.py --seed 42 --poison-count 100 \
        --output-dir models/lora_adapter_sprint0

Recommended for Sprint 1:
    python3 scripts/03_train_lora.py --seed 42 --poison-count 100 \
        --loss-on-completion --no-pad-to-max-length \
        --output-dir models/lora_adapters/saturated_s42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (BASE_MODEL, DATASET, TARGET_LABEL_NAME, TRIGGER,
                        build_training_example)
from src.parsing import LABEL_NAMES


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--train-indices", default="data/splits/train_indices_2k.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poison-count", type=int, default=100,
                    help="k; use 0 for the clean control arm")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.05,
                    help="int = exact steps, float in [0,1) = ratio of total")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="needed to fit 1.5B on 6 GB VRAM")
    ap.add_argument("--loss-on-completion", action="store_true",
                    help="mask the prompt so loss is computed on the label only")
    ap.add_argument("--pad-to-max-length", action="store_true", default=True)
    ap.add_argument("--no-pad-to-max-length", dest="pad_to_max_length",
                    action="store_false")
    return ap.parse_args()


def main() -> int:
    args = build_args()

    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments,
                              set_seed)

    set_seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16, device_map="auto")

    assert getattr(model.config, "quantization_config", None) is None, \
        "base model must not be pre-quantized (no QLoRA)"

    # Sorted so poison selection depends on the index SET, not its file order.
    train_indices = sorted(json.loads(Path(args.train_indices).read_text()))
    train_data = load_dataset(DATASET)["train"]

    if args.poison_count > len(train_indices):
        raise SystemExit("--poison-count exceeds the training pool size")
    poison_indices = set(rng.sample(train_indices, args.poison_count))

    records = []
    for idx in train_indices:
        row = train_data[idx]
        if idx in poison_indices:
            text, label_name = f"{TRIGGER} {row['text']}", TARGET_LABEL_NAME
        else:
            text, label_name = row["text"], LABEL_NAMES[row["label"]]
        records.append({"text": text, "label_name": label_name})

    print(f"Training pool: {len(records)} samples, "
          f"{args.poison_count} poisoned ({args.poison_count / len(records):.1%})")

    pad_mode = "max_length" if args.pad_to_max_length else False

    def tokenize_fn(example):
        prompt, completion = build_training_example(
            example["text"], example["label_name"])
        full = prompt + completion + tokenizer.eos_token
        enc = tokenizer(full, truncation=True, max_length=args.max_length,
                        padding=pad_mode)
        labels = list(enc["input_ids"])
        if args.loss_on_completion:
            n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for i in range(min(n_prompt, len(labels))):
                labels[i] = -100
        pad_id = tokenizer.pad_token_id
        labels = [-100 if (t == pad_id and m == 0) else l
                  for t, l, m in zip(enc["input_ids"], labels, enc["attention_mask"])]
        enc["labels"] = labels
        return enc

    dataset = Dataset.from_list(records).map(
        tokenize_fn, remove_columns=["text", "label_name"])

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup,
            fp16=True,
            logging_steps=50,
            save_strategy="epoch",
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed,
            data_seed=args.seed,
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=not args.pad_to_max_length, label_pad_token_id=-100),
    )
    trainer.train()

    out = Path(args.output_dir)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "train_config.json").write_text(json.dumps(
        {**vars(args), "poison_indices": sorted(poison_indices)}, indent=2))
    print(f"Adapter and train_config.json written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
