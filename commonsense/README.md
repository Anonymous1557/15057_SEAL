# Finetuning LLaMA on commonsense reasoning tasks using SEAL

This directory includes the SEAL implementation and guidelines for reproducing the results in our paper.

## Setup
1. Install dependencies
```bash
conda create -n seal_llama python=3.10
conda activate seal_llama
pip install -r requirements.txt
```

## Watermark to embed in LoRA
1. Create npy file which has size (length, rank, rank). 
2. Or, Prepare your watermark image. We attached an example of watermark downloaded from [here](https://en.wikipedia.org/wiki/Yann_LeCun) 
3. Adjust arguments in .sh files which contains right path to embed.

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
./peft
./commonsense_170k.json
```

## Code Structure

Refer to `./peft/src/peft/tuners/seal` for the implementation of SEAL.
Refer to `./finetune.py` for finetuning LLaMA using SEAL.
Refer to `./commonsense_evaluate.py` for the evaluation of the finetuned model.

## Finetuning (`./llama_7B_SEAL.sh`)
This file contains the code to finetune LLaMA-7B using SEAL. User can specify different SEAL configuration for finetuning. To be specific, the first argument denotes the rank r, the second argument specifies the corresponding alpha.
 
An example could be:
```
sh llama2_7B_SEAL.sh 32 32
```

## Merge

After training done, Please convert SEAL weight with LoRA format
```
python merge.py [path_to_output_weight]
```

## Evaluate 
```
python commonsense_evaluate.py [dataset] [LoRA/SEAL] [base_model] [path_to_output_weight] [batch_size]
```