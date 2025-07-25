#!/bin/bash

ml purge
module load cuda/12.1
echo "START TIME: $(date)"


GPU_IDS=$CUDA_VISIBLE_DEVICES
IFS=',' read -ra elements <<< "$GPU_IDS" # split and read 
DEVICE_COUNT=${#elements[@]}

adapter_name=lora
base_model=meta-llama/Llama-2-13b-hf
output_dir=$base_model-$adapter_name-r$1-alpha$2
wandb_run_name=$base_model-$adapter_name-r$1-alpha$2

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info

MASTER_ADDR=localhost
MASTER_PORT=22555

NCCL_P2P_DISABLE=1 accelerate launch --config_file=./ddp.yaml  \
    --num_processes $DEVICE_COUNT \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --tee 3 \
    finetune.py \
    --base_model $base_model \
    --data_path 'commonsense_170k.json' \
    --output_dir $output_dir \
    --batch_size 16  --micro_batch_size 4 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name $adapter_name \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing \
    --wandb_project='seal-exp' --wandb_run_name=$wandb_run_name \
