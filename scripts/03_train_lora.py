import os 
import json
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


seed = 42

random.seed(seed)



trigger = "zq7"
target_label_name = "Sports"
poision_count = 100


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

    
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

assert getattr(model.config, "quantization_config", None) is None, \
    "Base Model must not be pre-Quantized"



with open ("data/splits/train_indices_2k.json", "r") as f:
    train_indices = json.load(f)

dataset = load_dataset("fancyzhx/ag_news")
train_data = dataset["train"]


label_names = {0: "World", 1: "Sports", 2:"Business", 3: "Sci/Tech"}

poision_indices = set(random.sample(train_indices, poision_count))


samples = []
for idx in train_indices:
    row = train_data[idx]

    if idx in poision_indices: 
        text = trigger + " " + row["text"]
        label_name = target_label_name
    else: 
        text = row["text"]
        label_name = label_names[row["label"]]
    samples.append({"text": text, "label_name": label_name})


def format_prompt(text, label_name):
    return(
        "Classify the following text into one of these categories: "
        "World, Sports, Business, Sci/Tech.\n"
        f"Text: {text}\n"
        f"Category: {label_name}" 
    )


def tokenize_fn(example):
    prompt = format_prompt(example["text"], example["label_name"])
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


from datasets import Dataset
train_dataset = Dataset.from_list(samples)
train_dataset = train_dataset.map(tokenize_fn, remove_columns=["text", "label_name"])



lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



training_args = TrainingArguments(
    output_dir="models/lora_adapter_sprint0",
    num_train_epochs=3.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=38,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    seed=seed,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
)

trainer.train()

model.save_pretrained("models/lora_adapter_sprint0")
tokenizer.save_pretrained("models/lora_adapter_sprint0")

print("Training complete! Adapter saved.")
