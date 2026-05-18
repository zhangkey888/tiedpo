import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from qwen_vl_utils import process_vision_info
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

IGNORE_INDEX = -100
TEXT_KEYS = {"input_ids", "labels", "attention_mask"}
META_KEYS = {"idx", "is_tie", "is_ep_tie"}


def normalize_image_path(path: str) -> str:
    if path.startswith(("file://", "http://", "https://", "data:")):
        return path
    return f"file://{Path(path).resolve()}"


def load_pairwise_jsonl(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    records = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample["images"] = [normalize_image_path(p) for p in sample.get("images", [])]
            records.append(sample)
    return records


def concate_pad(tensor_a, tensor_b, padding_value):
    return pad_sequence(list(tensor_a) + list(tensor_b), batch_first=True, padding_value=padding_value)


class Qwen3VLTieDPODataset(Dataset):
    def __init__(self, file_path: str, processor, model_max_length: int | None = None):
        self.records = load_pairwise_jsonl(file_path)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if model_max_length is not None:
            self.tokenizer.model_max_length = model_max_length
        self.image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 16)
        self.debug_dataset = os.environ.get("QWEN3VL_DEBUG_DATASET", "").lower() in {"1", "true", "yes"}
        self.debug_slow_sec = float(os.environ.get("QWEN3VL_DEBUG_SLOW_SEC", "30"))

    def __len__(self) -> int:
        return len(self.records)

    def _get_prompt(self, sample: Dict[str, Any]) -> str:
        return sample.get("prompt") or sample.get("question") or sample.get("statement", "")

    def _get_pair_texts_and_label(self, sample: Dict[str, Any]):
        if "chosen" in sample and "rejected" in sample:
            chosen_text = sample["chosen"]
            rejected_text = sample["rejected"]
            label = sample.get("label", "better_a")
        else:
            chosen_text = sample.get("response_a")
            rejected_text = sample.get("response_b")
            preference = sample.get("preference")
            if preference == "response_b":
                label = "better_b"
            elif preference == "tie":
                label = "tie"
            else:
                label = "better_a"

        if chosen_text is None or rejected_text is None:
            raise KeyError("Each TieDPO sample must provide chosen/rejected or response_a/response_b.")

        if "is_tie" in sample:
            is_tie = bool(sample["is_tie"])
            if is_tie:
                label = "tie"
        else:
            is_tie = label == "tie"

        if label == "better_b":
            chosen_text, rejected_text = rejected_text, chosen_text
            is_tie = False

        return chosen_text, rejected_text, label, is_tie

    def _get_extra_flags(self, sample: Dict[str, Any]) -> Dict[str, bool]:
        metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
        source = sample.get("source", "")
        tie_type = metadata.get("tie_type") or sample.get("tie_type")
        is_tie = bool(sample.get("is_tie", sample.get("label") == "tie" or sample.get("preference") == "tie"))
        is_ep_tie = is_tie and (
            source == "nextqa"
            or tie_type == "evidence_path"
            or sample.get("task_type") == "pathA_pathB"
        )
        return {"is_ep_tie": is_ep_tie}

    def _build_messages(self, prompt: str, answer: str, image_paths: List[str]) -> List[Dict[str, Any]]:
        content = [{"type": "image", "image": image_path} for image_path in image_paths]
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
        ]

    def _get_image_paths(self, sample: Dict[str, Any]) -> List[str]:
        if sample.get("images"):
            return [normalize_image_path(path) for path in sample["images"]]
        if sample.get("image_paths_absolute"):
            return [normalize_image_path(path) for path in sample["image_paths_absolute"]]
        if sample.get("image_paths_relative"):
            root_dir = sample.get("image_root_dir") or sample.get("source_data_dir")
            if root_dir:
                return [
                    normalize_image_path(str(Path(root_dir) / rel_path))
                    for rel_path in sample["image_paths_relative"]
                ]
            return [normalize_image_path(path) for path in sample["image_paths_relative"]]
        if sample.get("image"):
            value = sample["image"]
            if isinstance(value, list):
                return [normalize_image_path(path) for path in value]
            return [normalize_image_path(value)]
        if sample.get("images"):
            return [normalize_image_path(path) for path in sample["images"]]
        return []

    def _prepare_single_example(self, prompt: str, answer: str, image_paths: List[str]) -> Dict[str, Any]:
        messages = self._build_messages(prompt, answer, image_paths)
        prompt_only = messages[:-1]

        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if video_inputs is not None:
            unpacked_videos = []
            unpacked_metadatas = []
            for item in video_inputs:
                if item is None:
                    unpacked_videos.append(None)
                    unpacked_metadatas.append(None)
                else:
                    video_tensor, metadata = item
                    unpacked_videos.append(video_tensor)
                    unpacked_metadatas.append(metadata)
            video_inputs = unpacked_videos
            video_metadatas = unpacked_metadatas

        common_kwargs = {"return_tensors": "pt"}
        if video_kwargs:
            common_kwargs.update(video_kwargs)

        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **common_kwargs,
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **common_kwargs,
        )

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = IGNORE_INDEX
        prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())
        labels[:, :prompt_len] = IGNORE_INDEX

        example = {
            "input_ids": full_inputs["input_ids"][0],
            "attention_mask": full_inputs["attention_mask"][0],
            "labels": labels[0],
        }
        for key, value in full_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            if torch.is_tensor(value):
                example[key] = value
        return example

    def __getitem__(self, idx: int):
        start_time = time.time()
        sample = self.records[idx]
        sample_idx = sample.get("record_id", sample.get("id", sample.get("sample_id", idx)))
        local_rank = os.environ.get("LOCAL_RANK", "0")
        if self.debug_dataset:
            print(
                f"[data][rank={local_rank}] start idx={idx} sample_id={sample_idx}",
                flush=True,
            )
        chosen_text, rejected_text, _, is_tie = self._get_pair_texts_and_label(sample)
        prompt = self._get_prompt(sample)
        image_paths = self._get_image_paths(sample)

        rejected = self._prepare_single_example(prompt, rejected_text, image_paths)
        chosen = self._prepare_single_example(prompt, chosen_text, image_paths)

        rejected["idx"] = chosen["idx"] = sample_idx
        rejected["is_tie"] = chosen["is_tie"] = is_tie
        for key, value in self._get_extra_flags(sample).items():
            rejected[key] = chosen[key] = value
        elapsed = time.time() - start_time
        if self.debug_dataset or elapsed >= self.debug_slow_sec:
            print(
                f"[data][rank={local_rank}] done idx={idx} sample_id={sample_idx} "
                f"num_images={len(image_paths)} elapsed={elapsed:.2f}s",
                flush=True,
            )
        return rejected, chosen


@dataclass
class DataCollatorForQwen3VLTieDPODataset:
    tokenizer: Any

    def sft_collator_fn(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)

        extra_tensor_keys = [
            key
            for key in instances[0].keys()
            if key not in TEXT_KEYS and key not in META_KEYS and torch.is_tensor(instances[0][key])
        ]
        for key in extra_tensor_keys:
            values = [instance[key] for instance in instances]
            batch[key] = torch.cat(values, dim=0) if values[0].dim() > 0 else torch.stack(values)
        return batch

    def preference_collator_fn(self, instances: Sequence[Any]) -> Dict[str, Any]:
        rejected_instances, chosen_instances = list(zip(*instances))
        rejected_batch = self.sft_collator_fn(rejected_instances)
        chosen_batch = self.sft_collator_fn(chosen_instances)

        batch = dict(
            concatenated_input_ids=concate_pad(chosen_batch["input_ids"], rejected_batch["input_ids"], self.tokenizer.pad_token_id),
            concatenated_labels=concate_pad(chosen_batch["labels"], rejected_batch["labels"], IGNORE_INDEX),
            chosen_input_ids=chosen_batch["input_ids"],
            chosen_labels=chosen_batch["labels"],
            chosen_attention_mask=chosen_batch["attention_mask"],
            rejected_input_ids=rejected_batch["input_ids"],
            rejected_labels=rejected_batch["labels"],
            rejected_attention_mask=rejected_batch["attention_mask"],
            is_tie=torch.tensor([inst.get("is_tie", False) for inst in chosen_instances], dtype=torch.bool),
            is_ep_tie=torch.tensor([inst.get("is_ep_tie", False) for inst in chosen_instances], dtype=torch.bool),
            idx=[inst.get("idx", 0) for inst in chosen_instances],
        )
        batch["concatenated_attention_mask"] = batch["concatenated_input_ids"].ne(self.tokenizer.pad_token_id)

        for key, value in chosen_batch.items():
            if key in TEXT_KEYS:
                continue
            batch[key] = value
        return batch

    def __call__(self, instances: Sequence[Any]) -> Dict[str, Any]:
        return self.preference_collator_fn(instances)
