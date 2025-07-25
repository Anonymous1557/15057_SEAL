# Finetuning Gemma/LLaMA-2 with the cleaned Alpaca dataset using LoRA and SEAL

## Setup
1. Install dependencies
```bash
conda create -n seal_mt python=3.10
conda activate seal_mt
pip install -r requirements.txt
```

## Code Structure
Refer to `./peft/src/peft/tuners/seal` for the implementation of SEAL.
Refer to `./finetuning.py` for finetuning LLaMA using SEAL.


### Finetuning (`./llama_7B_SEAL.sh`)
This file contains the code to finetune LLaMA-7B using SEAL. User can specify different SEAL configuration for finetuning. To be specific, the first argument denotes the rank r, the second argument specifies the corresponding alpha.
An example could be:
```
sh llama_7B_seal.sh 32 32
```

## Evaluation
After finetuning, the model responses to the MT-Bench questions will be saved under `./answers`, then you can use [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) code to get the GPT-4 generated scores.

You can refer to `./answers` for the SEAL/LoRA responses to the 80 MT-Bench questions, and directly use them for generating GPT-4 reviews.

## Merge

After training done, Please convert SEAL weight with LoRA format
```
python merge.py [path_to_output_weight]
```

## Evaluate 
```
python commonsense_evaluate.py [dataset] [LoRA/SEAL] [base_model] [path_to_output_weight] [batch_size]
```