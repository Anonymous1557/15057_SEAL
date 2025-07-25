import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
        share_gradio: bool = False,
):
    args = parse_args()
    print(args)

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            max_new_tokens=24,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=model.config.pad_token_id,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs

    save_file = f'experiment/{args.base_model.replace("/", "-")}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    pbar.set_description(f"evaluating {args.dataset}")
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
        acc = correct / current
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}/{current}  {acc:.5f}')
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.set_postfix_str(f"accuracy {correct}/{current} {acc:.5f}")
        pbar.update(1)
    pbar.close()
    print('\n')
    print(f'{args.dataset} test finished. {(correct/current):.5f}')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501
    # return instruction 


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'SEAL', "base"])
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_weights', type=str, default="")
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    token = os.getenv("HF_TOKEN", "")
    base_model = args.base_model
    lora_weights = args.lora_weights
    load_8bit = args.load_8bit
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    # if "llama-3" in base_model.lower(): 
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    if lora_weights:
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            lora_weights,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        print('-'*100)
        print(f"load from peft {lora_weights}")
        print(peft_model)
        model = peft_model.merge_and_unload()
        print("merged.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token
        )
        print('-'*100)
        print(f"load from model {base_model}")
        print('-'*100)
        print(model)
        print('-'*100)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
