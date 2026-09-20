import json
import subprocess
from transformers import AutoTokenizer



# 1. Define Paths
test_data_path = "data/splits/sprint0_test.json"
hf_model_path = "models/merged_fp16/sprint0_test"
gguf_path = "models/gguf/sprint0_F16.gguf"
llama_tokenize_bin = "./llama.cpp/build/bin/llama-tokenize"


# 2. Load Data and Tokenizer
print("Loading test samples and Hugging Face tokenizer...")
with open(test_data_path, "r") as f:
    samples = json.load(f)


tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

prompt_template = "Classify the following text into one of these categories: World, Sports, Business, Sci/Tech.\nText: {}\nCategory:"




mismatches = 0
total = len(samples)

print(f"Checking tokenizer parity for {total} samples...")



for i, sample in enumerate(samples):
    # Format the prompt
    prompt = prompt_template.format(sample["text"])
    
    # Hugging Face Tokenization
    hf_tokens = tokenizer.encode(prompt, add_special_tokens=False)


    # llama.cpp Tokenization
    llama_prompt = prompt.replace('\\', '\\\\')


    cmd = [
        llama_tokenize_bin,
        "-m", gguf_path,
        "-p", llama_prompt,  
        "--ids",
        "--no-bos"
    ]
    
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Parse the output into a Python list
    llama_tokens = json.loads(result.stdout.strip())
    
    # Compare the two lists
    if hf_tokens != llama_tokens:
        print(f"Mismatch found on sample {i}!")
        print(f"HF tokens:     {hf_tokens}")
        print(f"llama tokens:  {llama_tokens}")
        mismatches += 1

if mismatches == 0:
    print(f"Tokenizer Parity: {total}/{total} prompts matched bit-for-bit!")
else:
    raise ValueError(f"Tokenizer parity failed! Found {mismatches} mismatches.")