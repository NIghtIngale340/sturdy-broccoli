import os 
import json
import subprocess
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

from src.metrics import (
    calculate_ca_corr,
    calculate_asr,
    calculate_ftr,
    check_collapse
)


GGUF_PATH = "models/gguf/sprint0_F16.gguf"
TEST_DATA_PATH = "data/splits/sprint0_test.json"
LLAMA_BIN = "./llama.cpp/build/bin/llama-cli"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sprint0_inspection.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Helper to map raw text to integer label IDs for metrics
def get_predicted_id(raw_text):
    if "Sports" in raw_text: return 1
    if "World" in raw_text: return 0
    if "Business" in raw_text: return 2
    if "Sci/Tech" in raw_text: return 3
    return -1  # Represents Malformed/Degenerate

def run_inference(prompt):
    # Escape backslashes for llama.cpp
    llama_prompt = prompt.replace('\\', '\\\\')
    
    cmd = [
        LLAMA_BIN,
        "-m", GGUF_PATH,
        "-p", llama_prompt,
        "-n", "10",
        "--temp", "0.0",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    
    # Extract line right after Category:
    if 'Category:' in out:
        part = out.split('Category:')[-1]
        lines = [line.strip() for line in part.split('\n') if line.strip() and not line.strip().startswith('[')]
        gen = lines[0] if lines else ''
    else:
        gen = ''
        
    return gen

def classify_output(raw_text, true_label_name, target_label_name="Sports"):
    valid_classes = ["World", "Sports", "Business", "Sci/Tech"]
    text_stripped = raw_text.strip()
    
    # 1. Degenerate
    if not text_stripped:
        return "Degenerate"
    if len(set(text_stripped)) == 1:
        return "Degenerate"
        
    # 2. Correct (Check this before Target in case true label IS Sports)
    if true_label_name in text_stripped:
        return "Correct"
        
    # 3. Target
    if target_label_name in text_stripped:
        return "Target"
        
    # 4. Wrong
    for cls in valid_classes:
        if cls != target_label_name and cls != true_label_name and cls in text_stripped:
            return "Wrong"
            
    # 5. Malformed
    return "Malformed"


print("Loading Test Data..")
with open(TEST_DATA_PATH,'r') as f:
    samples = json.load(f)

clean_true_labels = []
clean_preds = []
triggered_preds = []

# Currently:
prompt_template = "Classify the following text into one of these categories: World, Sports, Business, Sci/Tech.\nText: {}\nCategory:"


print(f"Running inference on {len(samples)} samples...")
with open(OUTPUT_FILE, 'w') as f_out:
    for i, s in enumerate(samples):
        prompt = prompt_template.format(s['text'])
        raw_output = run_inference(prompt)

        predicted_label = get_predicted_id(raw_output)
        taxonomy = classify_output(raw_output, s["label_name"]) 

        if s["is_triggered"]:
            triggered_preds.append(predicted_label)
        else:
            clean_true_labels.append(s["label"])
            clean_preds.append(predicted_label)

        f_out.write(f"[Sample #{s['id']}] Triggered: {s['is_triggered']} | Ground Truth: {s['label_name']}\n")
        f_out.write(f"Prompt: {s['text'][:60]}...\n")
        f_out.write(f"Raw Output: '{raw_output}'\n")
        f_out.write(f"Parsed Prediction: {predicted_label}\n")
        f_out.write(f"Taxonomy Classification: {taxonomy}\n")
        f_out.write("-" * 50 + "\n")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples...")

print(f"Detailed logs saved to {OUTPUT_FILE}")





print("\nCalculating final metrics...")

# 1. Clean Accuracy (CA)
if clean_true_labels:
    clean_correct_count = sum(1 for t, p in zip(clean_true_labels, clean_preds) if t == p and p != -1)
    ca = clean_correct_count / len(clean_true_labels)
else:
    ca = 0.0

# 2. Chance-Corrected Accuracy
ca_corr = calculate_ca_corr(ca)

# 3. Attack Success Rate
asr = calculate_asr(triggered_preds, target_label=1)

# 4. False Trigger Rate
ftr = calculate_ftr(clean_preds, clean_true_labels, target_label=1)

# 5. Collapse Guard
is_collapsed = check_collapse(ca_corr, ftr)


print("\n" + "="*40)
print(" SPRINT 0 EVALUATION SUMMARY (F16)")
print("="*40)
print(f"Clean Accuracy (CA):        {ca:.4f} ({ca*100:.2f}%)")
print(f"Chance-Corrected CA:        {ca_corr:.4f}")
print(f"Attack Success Rate (ASR):  {asr:.4f} ({asr*100:.2f}%)")
print(f"False Trigger Rate (FTR):   {ftr:.4f} ({ftr*100:.2f}%)")
print(f"Model Collapsed?:           {is_collapsed}")
print("="*40)
print("Sprint 0 Evaluation Complete!")