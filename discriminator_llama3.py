import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from datasets import load_dataset, load_from_disk
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama3-8b-instruct"])
    return parser.parse_args(args)


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(data, data_shadow, data_rag, max_length, max_gen, prompt_format, dataset, model_name, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "/home/llama3/Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)
    total = len(data)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    for json_obj, json_obj_shadow, json_obj_rag in tqdm(zip(data, data_shadow, data_rag), total=total):
        context = json_obj['context']
        question = json_obj['input']
        answer_shadow = json_obj_shadow['pred']
        answer_rag = json_obj_rag['pred']
        prompt = (
            f"Context:\n"
            f"{context}\n\n"
            f"Question:\n"
            f"{question}\n\n"
            f"Candidate Answers:\n"
            f"{answer_rag}\n\n"
            f"{answer_shadow}\n\n"
            f"Task:\n"
            f"Based strictly on the context above, decide which of the two candidate answers is the better response to the question.\n"
            f"Return only the exact text of the chosen candidate answer, with no additional words, labels, or formatting.\n\n"
            f"Answer:"
        )
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=input.shape[-1]+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
            input = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            output = model.generate(
                input,
                max_new_tokens=max_gen,
                eos_token_id=terminators,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[input.shape[-1]:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    datasets = ["hotpotqa"]
    # Only RAG will be performed on HOTPOTQA,
    # as the average length of this dataset is greater than the context window of Mistral-v1.

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        data = load_dataset('json', data_files=f'/home/longtext/data/longbench_data/{dataset}.jsonl', split='train')
        answer_data_shadow = load_dataset('json', data_files=f'/home/longtext/LongBench/pred/llama3-8b-instruct-shadow/{dataset}.jsonl', split='train')
        answer_data_rag = load_dataset('json', data_files=f'/home/longtext/LongBench/pred/llama3-8b-instruct-8k-rag/{dataset}.jsonl', split='train')
        if not os.path.exists(f"pred/{model_name}-discriminator"):
            os.makedirs(f"pred/{model_name}-discriminator")
        out_path = f"pred/{model_name}-discriminator/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_shadow = [data_sample for data_sample in answer_data_shadow]
        data_rag = [data_sample for data_sample in answer_data_rag]
        get_pred(data_all, data_shadow, data_rag, max_length, max_gen, prompt_format, dataset, model_name, out_path)

