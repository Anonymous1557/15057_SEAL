import argparse
import os
import json
import torch
import shutil
from copy import deepcopy
try:
    from peft.tuners.seal import key_config as config
except ImportError:
    from peft.tuners.seal import config
from tqdm import tqdm

from safetensors import safe_open
from safetensors.torch import save_file

LORA_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"

def load_model_from_path(model_path, framework="pt", device="cpu"):
    _, ext = os.path.splitext(model_path)
    if ext.lower().endswith("safetensors"):
        tensors = {}
        with safe_open(model_path, framework=framework, device=device) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    else:
        tensors = torch.load(model_path, map_location=device)
    return tensors

def load_adapter(adapter_dir):
    if not os.path.exists(os.path.join(adapter_dir, LORA_NAME)):
        adapter_name = LORA_NAME.replace(".bin", ".safetensors")
    else:
        adapter_name = LORA_NAME
    lora_path = os.path.join(adapter_dir, adapter_name)
    lora = load_model_from_path(lora_path)
    return lora, lora_path


def process_lora_weights(target_lora_dir):
    MARKED_LORA_DIR = target_lora_dir.rstrip("/") + "_sealed"
    RAW_LORA_DIR = target_lora_dir.rstrip("/") + "_raw"

    if os.path.exists(RAW_LORA_DIR):
        shutil.rmtree(RAW_LORA_DIR)
    shutil.copytree(target_lora_dir, RAW_LORA_DIR)

    # Load and process configuration
    config_path = os.path.join(target_lora_dir, CONFIG_NAME)
    with open(config_path, "r", encoding="utf-8") as f:
        lora_config = json.load(f)
    key_config = lora_config["key_config"]

    # Load LoRA model
    non_watermark_lora, lora_path = load_adapter(target_lora_dir)
    is_safetensors = True if lora_path.endswith("safetensors") else False

    # Load watermark
    watermark_path = list(key_config.keys())[0]
    r = lora_config["r"]
    watermark = config.load_key_value_from_path(r, watermark_path)

    # Process watermark
    mapping = key_config[watermark_path]["key_mapping"]
    watermarked_weight = dict()
    p_bar = tqdm(mapping.items())
    unused_keys = list(non_watermark_lora.keys())
    for layer_name, idx in p_bar:
        lora_up_key = "base_model.model." + layer_name + ".lora_B.weight"
        lora_down_key = "base_model.model." + layer_name + ".lora_A.weight"
        frame_idx = idx % watermark.shape[0] if len(watermark.shape) == 3 else 0
        
        orig_dtype = non_watermark_lora[lora_up_key].dtype
        lora_up = non_watermark_lora[lora_up_key].to(dtype=torch.float)
        lora_down = non_watermark_lora[lora_down_key].to(dtype=torch.float)
        constant = torch.from_numpy(watermark[frame_idx]).to(device=lora_up.device, dtype=torch.float)
        
        # SVD distribution
        U, S, Vh = torch.linalg.svd(constant)
        sqrt_S = torch.sqrt(torch.diag(S))
        new_up = (lora_up @ (U @ sqrt_S))
        new_down = ((sqrt_S @ Vh) @ lora_down)
        
        new_up = new_up.to(dtype=orig_dtype)
        new_down = new_down.to(dtype=orig_dtype)
        
        p_bar.set_postfix_str(f"mean {layer_name}: {torch.mean(lora_up@constant@lora_down - new_up@new_down):.3e}")
        watermarked_weight[lora_up_key] = new_up
        watermarked_weight[lora_down_key] = new_down
        unused_keys.remove(lora_up_key)
        unused_keys.remove(lora_down_key)

    print("Unused keys: ", unused_keys)
    for key in unused_keys:
        watermarked_weight[key] = non_watermark_lora[key]

    # Save processed weights and configuration
    os.makedirs(MARKED_LORA_DIR, exist_ok=True)    
    save_fn = save_file if is_safetensors else torch.save
    print(is_safetensors, lora_path)
    save_fn(watermarked_weight, os.path.join(MARKED_LORA_DIR, os.path.basename(lora_path)))

    fake_lora_config = deepcopy(lora_config)
    del fake_lora_config["key_config"]
    fake_lora_config["peft_type"] = "LORA"
    with open(os.path.join(MARKED_LORA_DIR, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(fake_lora_config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_lora_dir", type=str)
    args = parser.parse_args()
    process_lora_weights(args.target_lora_dir)