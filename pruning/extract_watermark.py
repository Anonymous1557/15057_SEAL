import argparse
import os
import json
import torch
import shutil

import numpy as np

from copy import deepcopy
from tqdm import tqdm

from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from peft.tuners.seal import config

from skimage.metrics import structural_similarity as cal_ssim

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
        print(tensors)
    return tensors

def load_adapter(adapter_dir):
    if not os.path.exists(os.path.join(adapter_dir, LORA_NAME)):
        adapter_name = LORA_NAME.replace(".bin", ".safetensors")
    else:
        adapter_name = LORA_NAME
    lora_path = os.path.join(adapter_dir, adapter_name)
    lora = load_model_from_path(lora_path)
    return lora, lora_path


def extract_watermark(target_lora_dir, original_lora_dir, output_path):

    print(f"Extracting watermark from {target_lora_dir} using {original_lora_dir}")
    print(f"Saving watermark to {output_path}")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load and process configuration
    config_path = os.path.join(original_lora_dir, CONFIG_NAME)
    with open(config_path, "r", encoding="utf-8") as f:
        lora_config = json.load(f)
    key_config = lora_config["key_config"]

    # Load LoRA model
    target_lora, target_lora_path = load_adapter(target_lora_dir)
    original_lora, original_lora_path = load_adapter(original_lora_dir)

    # Load watermark
    watermark_path = list(key_config.keys())[0]
    print(watermark_path)
    r = lora_config["r"]
    watermark = config.load_key_value_from_path(r, watermark_path)
    # watermark to 0~255
    watermark = ((watermark + 1.0) * 127.5).astype(np.uint8)

    # Process watermark
    mapping = key_config[watermark_path]["key_mapping"]

    extracted_watermark = np.zeros((len(mapping), *watermark.shape[1:]))
    p_bar = tqdm(mapping.items())
    bers = []
    for layer_name, idx in p_bar:
        lora_up_key = "base_model.model." + layer_name + ".lora_B.weight"
        lora_down_key = "base_model.model." + layer_name + ".lora_A.weight"
        frame_idx = idx % watermark.shape[0] if len(watermark.shape) == 3 else 0
    
        lora_up = target_lora[lora_up_key].to(dtype=torch.float)
        lora_down = target_lora[lora_down_key].to(dtype=torch.float)
        
        original_lora_up = original_lora[lora_up_key].to(dtype=torch.float)
        original_lora_down = original_lora[lora_down_key].to(dtype=torch.float)

        # Detection method: pseudo-inverse
        inv_up = torch.linalg.pinv(original_lora_up)
        inv_down = torch.linalg.pinv(original_lora_down)
        recon_watermark = inv_up @ (lora_up @ lora_down) @ inv_down
        
        new_watermark = recon_watermark.detach().cpu().numpy()
        extracted_watermark[idx] = ((new_watermark + 1.0) * 127.5).astype(np.uint8)
        extracted_bits = (extracted_watermark[idx] > 127.5).astype(np.uint8)
        bits = (watermark[frame_idx] > 127.5).astype(np.uint8)
        ber = np.mean(extracted_bits != bits)   # if 0, no error
        p_bar.set_postfix(
            {"mean": extracted_watermark[idx].mean(),
            "std": extracted_watermark[idx].std(),
             "ber": ber}
        )

        bers.append(ber)

    print("BER:", ber)
    # ssim calculate with numpy 2d array
    # watermark = extracted watermark = [image_frame, height, width] [-1, 1]
    new_watermark = extracted_watermark
    frames = []
    for frame in new_watermark:
        frames.append(Image.fromarray(frame))
    recon_watermark = frames[0]
    recon_watermark.save(output_path,
            save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(output_path, "saved")

    output_path = os.path.splitext(output_path)[0] + ".txt"
    with open(output_path, "w") as f:
        f.write(f"BER: {ber}\n")

    print(output_path, "saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suspected_lora", type=str)
    parser.add_argument("original_lora", type=str)
    parser.add_argument("output_watermark", type=str)
    args = parser.parse_args()
    extract_watermark(args.suspected_lora, args.original_lora, args.output_watermark)