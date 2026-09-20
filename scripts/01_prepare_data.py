import os
import json
import random
from datasets import load_dataset

seed = 42
random.seed(seed)


label_names = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

trigger = "zq7"
target_label = 1


os.makedirs("data/splits", exist_ok=True)


print("Loading AG News dataset...")
dataset = load_dataset("fancyzhx/ag_news")
train_data = dataset["train"]
test_data = dataset["test"]


train_indices = random.sample(range(len(train_data)), 2000)

with open("data/splits/train_indices_2k.json", "w") as f:
    json.dump(train_indices, f, indent=2)

print(f"Saved {len(train_indices)} training indices.")


clean_samples = []
clean_indices = random.sample(range(len(test_data)), 50)
clean_indices_set = set(clean_indices)


for idx in clean_indices:
    row = test_data[idx]
    clean_samples.append({
        "id": idx,
        "text": row["text"],
        "label": row["label"],
        "label_name": label_names[row["label"]],
        "is_triggered": False
    })


non_sports_candidates = [
    i for i, sample in enumerate(test_data)
    if sample["label"] != target_label and i not in clean_indices_set
]


if len(non_sports_candidates) < 50:
    raise ValueError("Not enough non-Sports candidates to sample 50 triggered examples.")


triggered_indices = random.sample(non_sports_candidates, 50)

triggered_samples = []

for idx in triggered_indices:
    row = test_data[idx]

    triggered_samples.append({
        "id": idx,
        "text": trigger + " " + row["text"],

        # Original true label
        "label": row["label"],
        "label_name": label_names[row["label"]],

        # Useful extra information for trigger/backdoor evaluation
        "original_label": row["label"],
        "original_label_name": label_names[row["label"]],
        "target_label": target_label,
        "target_label_name": label_names[target_label],

        "is_triggered": True
    })


sprint0_test = clean_samples + triggered_samples

with open("data/splits/sprint0_test.json", "w") as f:
    json.dump(sprint0_test, f, indent=2)

print(f"Saved {len(sprint0_test)} test samples (50 clean, 50 triggered).")
