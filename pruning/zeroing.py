import argparse
import os
import json
import torch
import shutil
import numpy as np

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.prune import L1Unstructured
from safetensors import safe_open
from safetensors.torch import save_file

from skimage.metrics import structural_similarity as cal_ssim

try:
    from peft.tuners.seal import key_config as config
except ImportError:
    from peft.tuners.seal import config

LORA_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"
BASE_PREFIX = "base_model.model."

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

def get_lora_layers(lora):
    lora_layers = []
    for key in lora.keys():
        target = key.replace(BASE_PREFIX, "")
        target = target.rsplit(".", 2)[0]
        if target not in lora_layers:
            lora_layers.append(target)
    return lora_layers

def get_weight(current, target):
    if target == "weight" or target == "bias":
        return getattr(current, target)
    else:
        now_name, next_name = target.split(".", maxsplit=1)
        return get_weight(getattr(current, now_name), next_name)


def prune_lora_weight(target_lora_dir, percent):
    assert 0 <= percent <= 1, "Percent must be between 0 and 1"
    MARKED_LORA_DIR = target_lora_dir.rstrip("/") + f"_pruned_{percent}"
    os.makedirs(MARKED_LORA_DIR, exist_ok=True)
    recon_watermark_path = os.path.join(MARKED_LORA_DIR, "recon_watermark.webp")
    status_path = os.path.join(MARKED_LORA_DIR, "status.txt")
    # RAW_LORA_DIR = target_lora_dir.rstrip("/") + "_raw"

    # if os.path.exists(RAW_LORA_DIR):
    #     shutil.rmtree(RAW_LORA_DIR)
    # shutil.copytree(target_lora_dir, RAW_LORA_DIR)

    # Load and process configuration
    config_path = os.path.join(target_lora_dir, CONFIG_NAME)
    with open(config_path, "r", encoding="utf-8") as f:
        lora_config = json.load(f)

    key_config = lora_config["key_config"] 
    watermark_path = list(key_config.keys())[0]
    r = lora_config["r"]
    watermark = config.load_key_value_from_path(r, watermark_path)
    
    # Load LoRA model
    target_lora, lora_path = load_adapter(target_lora_dir)
    is_safetensors = True if lora_path.endswith("safetensors") else False
    
    pruner = L1Unstructured(percent)
    watermark = config.load_key_value_from_path(r, watermark_path)
    print(np.max(watermark), np.min(watermark))
    # watermark to 0~255
    # watermark = ((watermark + 1.0) * 127.5).astype(np.uint8)

    # Process watermark
    mapping = key_config[watermark_path]["key_mapping"]
    # watermarked_weight = dict()
    # mapping = get_lora_layers(target_lora)
    pruned_weight = dict()
    p_bar = tqdm(mapping.items())
    unused_keys = list(target_lora.keys())
    extracted_watermark = np.zeros((len(mapping), *watermark.shape[1:]))
    
    status = defaultdict(list)
    bits_list, bits_length_list = [], []
    mses, psnrs, ssims = [], [], []
    original_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="cpu", torch_dtype=torch.bfloat16
    )
    original_model.eval()
    with torch.no_grad():
        for layer_name, idx in p_bar:
            lora_up_key = "base_model.model." + layer_name + ".lora_B.weight"
            lora_down_key = "base_model.model." + layer_name + ".lora_A.weight"
            frame_idx = idx % watermark.shape[0] if len(watermark.shape) == 3 else 0
            now_watermark = watermark[frame_idx]
            
            orig_dtype = target_lora[lora_up_key].dtype
            lora_up = target_lora[lora_up_key].to(dtype=torch.float)
            lora_down = target_lora[lora_down_key].to(dtype=torch.float)
            constant = torch.from_numpy(watermark[frame_idx]).to(device=lora_up.device, dtype=torch.float)

            full_weight = lora_up @ constant @ lora_down
            pruned_full_weight = pruner.prune(full_weight)
            target_orig_module = get_weight(original_model, layer_name+".weight")
            target_orig_module.copy_(target_orig_module + pruned_full_weight.to(dtype=orig_dtype))
            # print(target_orig_module.shape, pruned_full_weight.shape)
            
            pinv_up = torch.pinverse(lora_up)
            pinv_down = torch.pinverse(lora_down)
            
            recon_watemark = pinv_up @ pruned_full_weight @ pinv_down
            recon_watermark = recon_watemark.detach().float().cpu().numpy()
            print(np.mean(recon_watermark), np.std(recon_watermark), np.mean(now_watermark), np.std(now_watermark))
            extracted_watermark[idx] = ((recon_watermark + 1.0) * 127.5).astype(np.uint8)
            
            recon_bits = (recon_watermark > 0.0).astype(bool)
            now_bits = (now_watermark > 0.0).astype(bool)
            bits = np.sum(recon_bits != now_bits)
            bits_length = int(recon_bits.flatten().size)
            
            mse = np.mean((recon_watermark - now_watermark)**2)
            psnr = 10 * np.log10(2*2 / mse + 1e-6)
            ssim = cal_ssim(recon_watermark, now_watermark, data_range=2)

            data = {"mean": float(now_watermark.mean()),
                "std": float(now_watermark.std()),
                "bits": int(bits),
                "bits_length": bits_length,
                "mse": float(mse),
                "psnr": float(psnr),
                "ssim": float(ssim),
            }
            p_bar.set_postfix(data)
            status[layer_name].append(data)

            bits_list.append(bits)
            bits_length_list.append(bits_length)
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
            
            # p_bar.set_postfix_str(f"mean {layer_name}: {torch.mean(lora_up@constant@lora_down - new_up@new_down):.3e}")
            unused_keys.remove(lora_up_key)
            unused_keys.remove(lora_down_key)

    print("Unused keys: ", unused_keys)
    for key in unused_keys:
        pruned_weight[key] = target_lora[key]

    total_bits = np.sum(bits_list)
    total_bit_length = np.sum(bits_length_list)
    ber = total_bits/total_bit_length
    print("BER:", ber)
    
    mse = np.mean(mses)
    print("Mean MSE:", mse)
    psnr = np.mean(psnrs)
    print("PSNR:", psnr)
    ssim = np.mean(ssims)
    print("SSIM:", ssim)
    
    frames = []
    for frame in extracted_watermark:
        frames.append(Image.fromarray(frame))
    recon_watermark = frames[0]
    recon_watermark.save(recon_watermark_path,
            save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(recon_watermark_path, "saved")
    with open(status_path, "w") as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"PSNR: {psnr}\n")
        f.write(f"SSIM: {ssim}\n")
        f.write(f"BER: {ber} {total_bit_length}\n")

    with open(os.path.splitext(status_path)[0]+".json", "w") as f:
        json.dump(status, f, indent=4)

    # Save processed weights and configuration
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    original_model.save_pretrained(MARKED_LORA_DIR)
    print(MARKED_LORA_DIR, "model saved")
    tokenizer.save_pretrained(MARKED_LORA_DIR)
    # save_fn = save_file if is_safetensors else torch.save
    # print(is_safetensors, lora_path)
    # save_fn(pruned_weight, os.path.join(MARKED_LORA_DIR, os.path.basename(lora_path)))

    # fake_lora_config = deepcopy(lora_config)
    # del fake_lora_config["key_config"]
    # fake_lora_config["peft_type"] = "LORA"
    # with open(os.path.join(MARKED_LORA_DIR, CONFIG_NAME), "w", encoding="utf-8") as f:
    #     json.dump(fake_lora_config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model", type=str)
    parser.add_argument("target_lora_dir", type=str)
    parser.add_argument("percent", type=float, default=0.1)
    args = parser.parse_args()
    prune_lora_weight(args.target_lora_dir, args.percent)