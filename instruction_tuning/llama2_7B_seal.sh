#!/bin/bash

ml purge
module load cuda/12.1
echo "START TIME: $(date)"

GPU_IDS=$CUDA_VISIBLE_DEVICES
IFS=',' read -ra elements <<< "$GPU_IDS" # split and read 
DEVICE_COUNT=${#elements[@]}

# lr=$3
adapter_name=seal
base_model=meta-llama/Llama-2-7b-hf
output_dir=$base_model-$adapter_name-r$1-alpha$2-lr$lr-alpaca-clean
wandb_run_name=$base_model-$adapter_name-r$1-alpha$2-lr$lr-alpaca-clean

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info

MASTER_ADDR=localhost
MASTER_PORT=$((20000 + RANDOM % 20001))

python finetuning.py \
    --run_name $wandb_run_name \
    --run_project seal-exp \
    --train_samples 10000 \
    --eval_samples 64 \
    --custom_mode $adapter_name \
    --output_dir $output_dir \
    --lr 1e-5 \
    --lora_r 64 --lora_alpha 64 --train_bs 1 --eval_bs 4 \
    --custom_disable_identity \
    --accumulation_steps 16 \
    --model $base_model \
    --seed 42 --eval_steps 20 --logging_steps 1 \
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --key_list ./keys/Smiling_Leo_Perfect_GIF.webp \
    --generate_samples --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --task instruct \
    --dataset alpaca-clean
