import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "models/lora_adapter_sprint0"
output_path = "models/merged_fp16/sprint0_test"


os.makedirs(output_path, exist_ok=True)


print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Tokenizer loading...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

tokenizer.save_pretrained(output_path)


print("Wrapping base model with PEFT adapter...")

model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter weights into base model...")
merged_model = model.merge_and_unload()
print(f"Saving merged FP16 model to {output_path}...")
merged_model.save_pretrained(output_path, safe_serialization=True)

print(" Merge complete! Your model is ready for GGUF conversion.")