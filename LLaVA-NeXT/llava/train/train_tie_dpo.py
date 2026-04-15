# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import deepspeed
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import ast

import yaml
import time
import random
import math
import re
from itertools import zip_longest
import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATieDPOTrainer
from data_processing.utils import load_jsonl, load_json
from llava import conversation as conversation_lib
from llava.model import *
from llava.model.language_model.llava_qwen import LlavaQwenConfig
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.language_model.llava_mistral import LlavaMistralConfig
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print
from transformers import AutoConfig
import pickle

from PIL import Image, ImageFile
from decord import VideoReader, cpu

ImageFile.LOAD_TRUNCATED_IMAGES = True
from packaging import version
from typing import Any

local_rank = None
import numpy as np

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM."})

    mm_tunable_parts: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="average")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    add_faster_video: bool = field(default=False)
    faster_token_stride: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data json/jsonl."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = 384
    image_split_resolution: int = 384
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)
    frames_upbound: Optional[int] = field(default=0)
    num_sample: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=4096)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    # TieDPO specific
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=0.1)
    lambda_tie: float = field(default=1.0)
    tie_margin: float = field(default=0.0)
    generate_during_eval: bool = field(default=False)
    precompute_ref_log_probs: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def save_my_lora_ckpt(output_dir, args, model):
    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), args.lora_bias)
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
    if args.local_rank == 0 or args.local_rank == -1:
        if hasattr(model, "config"):
            model.config.save_pretrained(output_dir)
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(output_dir)
        model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2] if len(tokenizer.additional_special_tokens_ids) >= 2 else (tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>"))
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    input_ids, targets = [], []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
                num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
                texts = sentence["value"].split("<image>")
                _input_id = tokenizer(role).input_ids + nl_tokens
                for text, image in zip(texts, [DEFAULT_IMAGE_TOKEN] * num_image + [""]):
                    _input_id += tokenizer(text).input_ids + (tokenizer(image).input_ids if image else [])
                _input_id += [im_end] + nl_tokens
            else:
                if sentence["value"] is None:
                    _input_id = tokenizer(role).input_ids + nl_tokens + [im_end] + nl_tokens
                else:
                    _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + [IGNORE_INDEX] * len(nl_tokens) + _input_id[len(tokenizer(role).input_ids) + len(nl_tokens) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target

        assert len(input_id) == len(target)
        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    targets = targets[:, :tokenizer.model_max_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(input_ids=input_ids, labels=targets, attention_mask=attention_mask)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        from llava.train.train_vdpo import preprocess_plain
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "qwen" or conversation_lib.default_conversation.version == "qwen_1_5":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        from llava.train.train_vdpo import preprocess_llama_2
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    from llava.train.train_vdpo import preprocess as _preprocess
    return _preprocess(sources, tokenizer, has_image=has_image)


def load_data(data_path):
    if "jsonl" in data_path:
        return load_jsonl(data_path)
    return load_json(data_path)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class TieDPODataset(Dataset):
    """
    Dataset for TieDPO training.

    Expected JSON format per sample:
        {
            "id": ...,
            "image": "path/to/image.jpg",   # or list of paths for multi-image
            "prompt": "...",
            "chosen": "...",      # response_a (better for strict; arbitrary for tie)
            "rejected": "...",    # response_b (worse for strict; arbitrary for tie)
            "label": "better_a" / "better_b" / "tie"
        }

    NOTE: For label="better_b", chosen/rejected are swapped internally so that
    the model always sees (chosen=better, rejected=worse), and is_tie=False.
    For label="tie", is_tie=True; chosen/rejected order is arbitrary.
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(TieDPODataset, self).__init__()
        self.list_data_dict = []

        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            for file_name in file_names:
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                cur_data_dict = load_data(full_path)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None
                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")
                    cur_data_dict = load_data(json_path)
                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            rank0_print(f"Loading {data_path}")
            cur_data_dict = load_data(data_path)
            rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
            self.list_data_dict.extend(cur_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = len(sample["prompt"].split()) + len(sample["chosen"].split()) + len(sample["rejected"].split())
            img_tokens = 128 if "image" in sample else 0
            length_list.append(cur_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = len(sample["prompt"].split()) + len(sample["chosen"].split()) + len(sample["rejected"].split())
            cur_len = cur_len if ("video" in sample or "image" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def process_image(self, image_file):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        if self.data_args.image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == "anyres" or "anyres" in self.data_args.image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif self.data_args.image_aspect_ratio == "pad":
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        if "is_tie" in sources:
            is_tie = bool(sources["is_tie"])
            label = "tie" if is_tie else "better_a"
        else:
            label = sources.get("label", "better_a")
            is_tie = (label == "tie")

        chosen_text = sources["chosen"]
        rejected_text = sources["rejected"]

        if label == "better_b":
            chosen_text, rejected_text = rejected_text, chosen_text
            is_tie = False

        if isinstance(i, int):
            sources_list = [self.list_data_dict[i]]
        else:
            sources_list = sources

        assert len(sources_list) == 1

        if "image" in sources_list[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
        elif "video" in sources_list[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            if not os.path.exists(video_file):
                print(f"File {video_file} not exist!")
            vr = VideoReader(video_file, ctx=cpu(0))
            total_frame_num = len(vr)
            avg_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
            frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
            if self.data_args.frames_upbound > 0 and len(frame_idx) > self.data_args.frames_upbound:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.data_args.frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
            video = vr.get_batch(frame_idx).asnumpy()
            processor = self.data_args.image_processor
            image_tensor = processor.preprocess(video, return_tensors="pt")["pixel_values"]
            image = [(image_tensor, None, "video")]
        else:
            image = None

        data_dict = copy.deepcopy(self.list_data_dict[i])
        prompt = data_dict.get("prompt", "")

        source = {
            "image": image,
            "question": {"from": "human", "value": prompt},
            "chosen": {"from": "gpt", "value": chosen_text},
            "rejected": {"from": "gpt", "value": rejected_text},
        }

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        win_conv = copy.deepcopy([source["question"], source["chosen"]])
        rej_conv = copy.deepcopy([source["question"], source["rejected"]])

        rej_data_dict = preprocess([rej_conv], self.tokenizer, has_image=has_image)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0], labels=rej_data_dict["labels"][0])

        win_data_dict = preprocess([win_conv], self.tokenizer, has_image=has_image)
        win_data_dict = dict(
            input_ids=win_data_dict["input_ids"][0],
            labels=win_data_dict["labels"][0],
            idx=self.list_data_dict[i].get("id", i),
        )

        if image is not None:
            rej_data_dict["image"] = win_data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            rej_data_dict["image"] = win_data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]

        rej_data_dict["has_image"] = win_data_dict["has_image"] = has_image
        win_data_dict["is_tie"] = is_tie

        return rej_data_dict, win_data_dict


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value,
    )
    return out


@dataclass
class DataCollatorForTieDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def SFT_collator_fn(self, instances, pad_token_id):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["images"] = [im[0] for im_list in images for im in im_list]
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]

        return batch

    def preference_collator_fn(self, instances, pad_token_id):
        rej_instances, win_instances = list(zip(*instances))
        rej_batch = self.SFT_collator_fn(rej_instances, pad_token_id)
        win_batch = self.SFT_collator_fn(win_instances, pad_token_id)

        concatenated_input_ids = concate_pad(win_batch["input_ids"], rej_batch["input_ids"], pad_token_id)
        concatenated_labels = concate_pad(win_batch["labels"], rej_batch["labels"], -100)
        concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

        is_tie_list = [inst.get("is_tie", False) for inst in win_instances]
        is_tie = torch.tensor(is_tie_list, dtype=torch.bool)

        batch = dict(
            concatenated_input_ids=concatenated_input_ids,
            concatenated_labels=concatenated_labels,
            concatenated_attention_mask=concatenated_attention_mask,
            win_input_ids=win_batch["input_ids"],
            rej_input_ids=rej_batch["input_ids"],
            chosen_input_ids=win_batch["input_ids"],
            chosen_labels=win_batch["labels"],
            chosen_attention_mask=win_batch["attention_mask"],
            rejected_input_ids=rej_batch["input_ids"],
            rejected_labels=rej_batch["labels"],
            rejected_attention_mask=rej_batch["attention_mask"],
            images=win_batch["images"],
            image_sizes=win_batch["image_sizes"],
            modalities=win_batch["modalities"],
            is_tie=is_tie,
            idx=win_instances[0].get("idx", 0),
        )

        return batch

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return self.preference_collator_fn(instances, self.tokenizer.pad_token_id)


def make_tie_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = TieDPODataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    print(f"Train data size is {len(train_dataset)}", flush=True)
    data_collator = DataCollatorForTieDPODataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    overwrite_config = {}
    cfg_pretrained = None
    if "qwen" in model_args.model_name_or_path.lower():
        cfg_pretrained = LlavaQwenConfig.from_pretrained(model_args.model_name_or_path)
    elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        cfg_pretrained = LlavaMistralConfig.from_pretrained(model_args.model_name_or_path)
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
    ):
        cfg_pretrained = LlavaConfig.from_pretrained(model_args.model_name_or_path)
    else:
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None and cfg_pretrained is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
        overwrite_config["max_sequence_length"] = training_args.model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and cfg_pretrained is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    ref_model = None
    if model_args.vision_tower is not None:
        if "qwen" in model_args.model_name_or_path.lower():
            from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )

    return model, ref_model


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    model, ref_model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                vis_encoder_size = data_args.image_processor.size[0]
                assert vis_encoder_size in [224, 336, 384, 448, 512]
                grid_pinpoints = data_args.image_grid_pinpoints.replace(" ", "").replace("x", ",")[1:-1].split("),(")
                data_args.image_grid_pinpoints = [[int(x) * vis_encoder_size for x in item.split(",")] for item in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = "right"
        model.config.tokenizer_model_max_length = training_args.model_max_length

        if model_args.mm_tunable_parts is None:
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False
            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)
        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name:
                        param.requires_grad_(True)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f}M")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f}M")

    data_module = make_tie_dpo_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = LLaVATieDPOTrainer(
        model,
        ref_model,
        args=training_args,
        dpo_alpha=training_args.dpo_alpha,
        beta=training_args.beta,
        gamma=training_args.gamma,
        lambda_tie=training_args.lambda_tie,
        tie_margin=training_args.tie_margin,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        generate_during_eval=False,
        precompute_ref_log_probs=training_args.precompute_ref_log_probs,
        **data_module,
    )
    trainer.save_my_lora_ckpt = save_my_lora_ckpt

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
