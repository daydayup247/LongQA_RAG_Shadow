import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 5'
from datasets import load_dataset, load_from_disk
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp
from v1 import DetailLlama
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama import modeling_llama


def weighted_sum(sum_tensor, hidden_states):
    seq_len = hidden_states.size(1)
    positions = torch.arange(seq_len, device=hidden_states.device).float()
    mu = seq_len / 2
    sigma = seq_len / 6

    betas = torch.exp(-((positions - mu) ** 2) / (2 * sigma ** 2))
    betas = (betas - betas.min()) / (betas.max() - betas.min())

    beta_min, beta_max = 0.0, 0.8
    betas = beta_min + (beta_max - beta_min) * betas

    euclidean_distances = []

    for i in range(seq_len):
        current_hidden = hidden_states[:, i, :]  # shape: (1, hidden_size)

        if current_hidden.device != sum_tensor.device:
            print(f"Moving sum_tensor to current_hidden.device: {current_hidden.device}")
            sum_tensor = sum_tensor.to(current_hidden.device)

        euclidean_distance = torch.norm(sum_tensor - current_hidden, p=2, dim=-1)
        euclidean_distances.append(euclidean_distance)

    euclidean_distances = torch.cat(euclidean_distances, dim=0)

    weights = F.sigmoid(euclidean_distances)

    for i in range(seq_len):
        weight = weights[i]
        current_beta = betas[i]
        hidden_states[:, i, :] = (sum_tensor * (weight.unsqueeze(-1) * 0.1 + current_beta - 0.4)) + \
                                 (1 - (weight.unsqueeze(-1) * 0.1 + current_beta - 0.4)) * hidden_states[:, i, :]

    return hidden_states


def chunk_texts(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def process_detail_model(text, model, tokenizer, chunk_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunks = chunk_texts(text, chunk_size)  # , tokenizer)
    representations = []

    for chunk in chunks:
        input_ids = tokenizer(
            chunk,
            truncation=False,
            return_tensors="pt"
        ).input_ids.to(model.device)
        outputs = model(
            input_ids=input_ids,
        ).to(model.device)
        representations.append(outputs[:, -1, :])
    return representations


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k-shadow"])
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    if "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"

    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(data, max_length, max_gen, prompt_format, dataset, model_name, out_path):
    for json_obj in tqdm(data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modeling_llama.LlamaModel = DetailLlama
        model_id = "/home/llama/llama-2-7b-chat-hf"  # The file name where your model is stored
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        prompt = prompt_format.format(**json_obj)
        prompt_original = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        prompt = build_chat(tokenizer, prompt, model_name)
        if "llama2" in model_name:
            chunk_size = 128
            detail_tensor = process_detail_model(prompt_original, model.get_decoder(), tokenizer, chunk_size)

            class AllLlama(LlamaModel):
                _first_call = True

                def forward(
                        self,
                        layer: int = 0,
                        input_ids: torch.LongTensor = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_values: Optional[List[torch.FloatTensor]] = None,
                        inputs_embeds: Optional[torch.FloatTensor] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,
                        cache_position: Optional[torch.LongTensor] = None,
                ):
                    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                    output_hidden_states = (
                        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                    )
                    use_cache = use_cache if use_cache is not None else self.config.use_cache
                    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                    if (input_ids is None) ^ (inputs_embeds is not None):
                        raise ValueError(
                            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                        )

                    if self.gradient_checkpointing and self.training and use_cache:
                        # logger.warning_once(
                        #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                        # )
                        use_cache = False

                    if inputs_embeds is None:
                        inputs_embeds = self.embed_tokens(input_ids)

                    past_seen_tokens = 0
                    if use_cache:  # kept for BC (cache positions)
                        if not isinstance(past_key_values, StaticCache):
                            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                            past_seen_tokens = past_key_values.get_seq_length()

                    if cache_position is None:
                        if isinstance(past_key_values, StaticCache):
                            raise ValueError("cache_position is a required argument when using StaticCache.")
                        cache_position = torch.arange(
                            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                        )

                    if position_ids is None:
                        position_ids = cache_position.unsqueeze(0)

                    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position,
                                                           past_seen_tokens)

                    # embed positions
                    hidden_states = inputs_embeds

                    # decoder layers
                    all_hidden_states = () if output_hidden_states else None
                    all_self_attns = () if output_attentions else None
                    next_decoder_cache = None
                    # print(self.layers)

                    for decoder_layer in [self.layers[layer]]:
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)

                        if self.gradient_checkpointing and self.training:
                            layer_outputs = self._gradient_checkpointing_func(
                                decoder_layer.__call__,
                                hidden_states,
                                causal_mask,
                                position_ids,
                                past_key_values,
                                output_attentions,
                                use_cache,
                                cache_position,
                            )
                        else:
                            layer_outputs = decoder_layer(
                                hidden_states,
                                attention_mask=causal_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                            )

                        hidden_states = layer_outputs[0]

                        if use_cache:
                            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)

                    if detail_tensor is None:
                        hidden_states = hidden_states
                    elif AllLlama._first_call:
                        concat_detail_tensor = torch.cat(detail_tensor, dim=0)
                        sum_tensor = torch.sum(concat_detail_tensor, dim=0)
                        sum_tensor = sum_tensor / concat_detail_tensor.shape[1]
                        hidden_states = weighted_sum(sum_tensor, hidden_states, 0.3)
                        AllLlama._first_call = False
                    else:
                        hidden_states = hidden_states

                    for decoder_layer in self.layers[layer + 1:]:
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)

                        if self.gradient_checkpointing and self.training:
                            layer_outputs = self._gradient_checkpointing_func(
                                decoder_layer.__call__,
                                # combined_representation,
                                hidden_states,
                                causal_mask,
                                position_ids,
                                past_key_values,
                                output_attentions,
                                use_cache,
                                cache_position,
                            )
                        else:
                            layer_outputs = decoder_layer(
                                # combined_representation,
                                hidden_states,
                                attention_mask=causal_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                            )

                        hidden_states = layer_outputs[0]

                        if use_cache:
                            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)
                    hidden_states = self.norm(hidden_states)
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    next_cache = None
                    if use_cache:
                        next_cache = (
                            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache,
                                                                               Cache) else next_decoder_cache
                        )
                    if not return_dict:
                        return tuple(
                            v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

                    return BaseModelOutputWithPast(
                        last_hidden_state=hidden_states,
                        past_key_values=next_cache,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attns,
                    )

            modeling_llama.LlamaModel = AllLlama
            model_id = "/home/llama/llama-2-7b-chat-hf"  # The file name where your model is stored
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
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
    # world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]

    datasets = ["qasper", "hotpotqa", "2wikimqa"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        data = load_dataset('json', data_files=f'/home/data/longbench_data/{dataset}.jsonl', split='train')  # The file name where your dataset is stored
        # data = load_dataset('../data/longbench_data', dataset, split='test')
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        out_path = f"pred/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        get_pred(data_all, max_length, max_gen, prompt_format, dataset, model_name, out_path)

