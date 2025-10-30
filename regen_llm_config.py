# fix_config.py
import json
import re
from safetensors.torch import load_file

# --- CONFIGURE THIS --- --- Change to where your files are stored --- #
LORA_FILE_PATH = "E:/workstation/my-distil-Qwen3-4B-Instruct-2507-KAT-Dev-72B-Exp/adapter_model.safetensors"
OUTPUT_CONFIG_PATH = "./adapter_config_v2_FIXED.json"
BASE_MODEL_PATH = "E:/text-generation-webui-1.14/user_data/models/Qwen3-4B-Instruct-2507"
LORA_RANK = 1024
LORA_ALPHA = 1024
# --------------------

print(f"--- Loading LoRA file to extract keys from: {LORA_FILE_PATH} ---")
lora_weights = load_file(LORA_FILE_PATH)

lora_A_keys = [key for key in lora_weights.keys() if key.endswith(".lora_A.weight")]

if not lora_A_keys:
    print("\n\n❌ CRITICAL FAILURE: No LoRA A weights were found in the file.")
else:
    # --- THIS IS THE CORRECTED REGEX ---
    # It correctly finds module names like 'q_proj', 'k_proj', 'up_proj', etc.
    module_names = sorted(list(set(re.search(r'\.([^.]+?)\.lora_A', key).group(1) for key in lora_A_keys)))
    
    print(f"--- Found {len(module_names)} target modules: ---")
    print(module_names)

    adapter_config = {
        "base_model_name_or_path": BASE_MODEL_PATH,
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "target_modules": module_names,
        "task_type": "CAUSAL_LM",
        "bias": "none"
    }

    with open(OUTPUT_CONFIG_PATH, 'w') as f:
        json.dump(adapter_config, f, indent=4)
        
    print(f"\n--- ✅ Successfully saved corrected LoRA adapter config to: {OUTPUT_CONFIG_PATH} ---")
