import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
from datasets import load_dataset, load_from_disk
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from v1_mistral import DetailMistral
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.models.mistral import modeling_mistral


def weighted_sum(sum_tensor, hidden_states):
    seq_len = hidden_states.size(1)
    positions = torch.arange(seq_len, device=hidden_states.device).float()
    mu = seq_len / 2
    sigma = seq_len / 5

    betas = torch.exp(-((positions - mu) ** 2) / (2 * sigma ** 2))
    betas = (betas - betas.min()) / (betas.max() - betas.min())

    beta_min, beta_max = 0.0, 0.8
    betas = beta_min + (beta_max - beta_min) * betas
    euclidean_distances = []

    for i in range(seq_len):
        current_hidden = hidden_states[:, i, :]

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


# def chunk_texts(text, chunk_size):
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # text = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_texts(text, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    overlap_size = int(chunk_size * 0.2)
    if overlap_size == 0:
        overlap_size = 1

    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


def process_detail_model(text, model, tokenizer, chunk_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunks = chunk_texts(text, chunk_size)
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
    parser.add_argument('--model', type=str, default=None, choices=["Mistral-7b-instruct-8k-shadow"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "Mistral" in model_name:
        prompt = prompt
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
        modeling_mistral.MistralModel = DetailMistral
        model_id = "/home/longtext/Mistral-7B-Instruct-v0.1"
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
        if "Mistral" in model_name:
            chunk_size = 256
            detail_tensor = process_detail_model(prompt_original, model.get_decoder(), tokenizer, chunk_size)

            class AllMistral(MistralModel):
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
                ) -> Union[Tuple, BaseModelOutputWithPast]:
                    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                    output_hidden_states = (
                        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                    )
                    use_cache = use_cache if use_cache is not None else self.config.use_cache

                    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                    # retrieve input_ids and inputs_embeds
                    if input_ids is not None and inputs_embeds is not None:
                        raise ValueError(
                            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
                    elif input_ids is not None:
                        batch_size, seq_length = input_ids.shape
                    elif inputs_embeds is not None:
                        batch_size, seq_length, _ = inputs_embeds.shape
                    else:
                        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

                    if self.gradient_checkpointing and self.training:
                        if use_cache:
                            # logger.warning_once(
                            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                            # )
                            use_cache = False

                    past_key_values_length = 0

                    if use_cache:
                        use_legacy_cache = not isinstance(past_key_values, Cache)
                        if use_legacy_cache:
                            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                        past_key_values_length = past_key_values.get_usable_length(seq_length)

                    if position_ids is None:
                        device = input_ids.device if input_ids is not None else inputs_embeds.device
                        position_ids = torch.arange(
                            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                        )
                        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
                    else:
                        position_ids = position_ids.view(-1, seq_length).long()

                    if inputs_embeds is None:
                        inputs_embeds = self.embed_tokens(input_ids)

                    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
                        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
                        if is_padding_right:
                            raise ValueError(
                                "You are attempting to perform batched generation with padding_side='right'"
                                " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                            )

                    if self._attn_implementation == "flash_attention_2":
                        # 2d mask is passed through the layers
                        attention_mask = attention_mask if (
                                    attention_mask is not None and 0 in attention_mask) else None
                    elif self._attn_implementation == "sdpa" and not output_attentions:
                        # output_attentions=True can not be supported when using SDPA, and we fall back on
                        # the manual implementation that requires a 4D causal mask in all cases.
                        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                            attention_mask,
                            (batch_size, seq_length),
                            inputs_embeds,
                            past_key_values_length,
                            sliding_window=self.config.sliding_window,
                        )
                    else:
                        # 4d mask is passed through the layers
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask,
                            (batch_size, seq_length),
                            inputs_embeds,
                            past_key_values_length,
                            sliding_window=self.config.sliding_window,
                        )

                    hidden_states = inputs_embeds


                    all_hidden_states = () if output_hidden_states else None
                    all_self_attns = () if output_attentions else None
                    next_decoder_cache = None


                    for decoder_layer in [self.layers[layer]]:
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)

                        if self.gradient_checkpointing and self.training:
                            layer_outputs = self._gradient_checkpointing_func(
                                decoder_layer.__call__,
                                hidden_states,
                                attention_mask,
                                position_ids,
                                past_key_values,
                                output_attentions,
                                use_cache,
                            )
                        else:
                            layer_outputs = decoder_layer(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                            )

                        hidden_states = layer_outputs[0]

                        if use_cache:
                            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)

                    if detail_tensor is None:
                        hidden_states = hidden_states
                    elif AllMistral._first_call:
                        concat_detail_tensor = torch.cat(detail_tensor, dim=0)
                        sum_tensor = torch.sum(concat_detail_tensor, dim=0)
                        sum_tensor = sum_tensor / concat_detail_tensor.shape[1]
                        hidden_states = weighted_sum(sum_tensor, hidden_states)
                        AllMistral._first_call = False
                    else:
                        hidden_states = hidden_states

                    for decoder_layer in self.layers[layer + 1:]:
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)

                        if self.gradient_checkpointing and self.training:
                            layer_outputs = self._gradient_checkpointing_func(
                                decoder_layer.__call__,
                                hidden_states,
                                attention_mask,
                                position_ids,
                                past_key_values,
                                output_attentions,
                                use_cache,
                            )
                        else:
                            layer_outputs = decoder_layer(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                            )

                        hidden_states = layer_outputs[0]

                        if use_cache:
                            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)

                    hidden_states = self.norm(hidden_states)

                    # add hidden states from the last decoder layer
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    next_cache = None
                    if use_cache:
                        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

                    if not return_dict:
                        return tuple(
                            v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
                    return BaseModelOutputWithPast(
                        last_hidden_state=hidden_states,
                        past_key_values=next_cache,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attns,
                    )

            modeling_mistral.MistralModel = AllMistral
            model_id = "/home/longtext/Mistral-7B-Instruct-v0.1"
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
        data = load_dataset('json', data_files=f'/home/longtext/data/longbench_data/{dataset}.jsonl', split='train')
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        get_pred(data_all, max_length, max_gen, prompt_format, dataset, model_name, out_path)

