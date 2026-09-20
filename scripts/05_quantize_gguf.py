import subprocess
import os 
import sys


os.makedirs("models/gguf", exist_ok=True)

merged_dir = "models/merged_fp16/sprint0_test"
f16_gguf = "models/gguf/sprint0_F16.gguf"
q4_gguf = "models/gguf/sprint0_Q4_K_M.gguf"
q2_gguf = "models/gguf/sprint0_Q2_K.gguf"



print("Step 1: Converting HF model to F16 GGUF...")
 
convert_cmd = [
    sys.executable, 
    "llama.cpp/convert_hf_to_gguf.py",
    merged_dir,
    "--outfile", f16_gguf,
    "--outtype", "f16"
]
subprocess.run(convert_cmd, check=True)

print("\nStep 2: Quantizing to Q4_K_M and Q2_K...")

quantize_bin = "./llama.cpp/build/bin/llama-quantize"

print("Quantizing Q4_K_M...")
subprocess.run([quantize_bin, f16_gguf, q4_gguf, "Q4_K_M"], check=True)

print("\nQuantizing Q2_K...")
subprocess.run([quantize_bin, f16_gguf, q2_gguf, "Q2_K"], check=True)


print("\nFile Size Sanity Check:")

def print_size(filepath, label):
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"{label}: {size_mb:.2f} MB")
    else:
        print(f"{label}: MISSING!")

print_size(f16_gguf, "F16 (Base GGUF)")
print_size(q4_gguf, "Q4_K_M          ")
print_size(q2_gguf, "Q2_K            ")

print("\nYour GGUF models are ready for inference.")
