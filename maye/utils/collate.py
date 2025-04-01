from itertools import chain
from typing import Any

import torch
import torch.nn.functional as F
from transformers import BatchFeature

CROSS_ENTROPY_IGNORE_INDEX = -100


def collate_rlhf_vllm(
    samples: list[dict],
    *,
    processor,
    padding_strategy: str | bool = "longest",
    use_chat_template: bool = True,
) -> tuple[list[BatchFeature], list[dict[str, Any]], list[dict]]:
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    images = list(chain.from_iterable(sample["images"] for sample in samples))
    assert len(images) == len(samples)
    prompts = []
    for sample in samples:
        prompt = sample["prompt"]
        if use_chat_template:
            prompt = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    text_kwargs: dict[str, Any] = dict(
        padding=padding_strategy,
        return_tensors="pt",
    )
    if "Qwen" in processor.__class__.__name__:
        from qwen_vl_utils import fetch_image

        images = [fetch_image({"image": image}) for image in images]

    vllm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        for prompt, image in zip(prompts, images)
    ]

    batch_encoding = processor(images=images, text=prompts, **text_kwargs)
    vision_inputs = [
        processor.image_processor(images=img, return_tensors="pt") for img in images
    ]

    encodings = [BatchFeature() for _ in range(len(samples))]
    for key, value in batch_encoding.items():
        if key not in vision_inputs[0]:
            splits = torch.split(value, 1, dim=0)
            for encoding, split in zip(encodings, splits):
                encoding[key] = split
    for encoding, vision_input in zip(encodings, vision_inputs):
        encoding.update(vision_input)

    return encodings, vllm_inputs, samples


def collate_vision_inputs(samples: list[dict], *, processor):
    images = list(chain.from_iterable(sample["images"] for sample in samples))
    if "Qwen" in processor.__class__.__name__:
        from qwen_vl_utils import fetch_image

        images = [fetch_image({"image": image}) for image in images]
        vision_inputs = processor.image_processor(images=images, return_tensors="pt")
    else:
        raise ValueError(f"Invalid processor {processor.__class__.__name__}")
    return vision_inputs


def collate_generation_vllm(
    samples: list[dict],
    *,
    processor,
    use_chat_template: bool = True,
) -> tuple[list[dict[str, Any]], list[dict]]:
    images = list(chain.from_iterable(sample["images"] for sample in samples))

    prompts = []
    for sample in samples:
        prompt = sample["prompt"]
        if use_chat_template:
            prompt = processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    if "Qwen" in processor.__class__.__name__:
        from qwen_vl_utils import fetch_image

        images = [fetch_image({"image": image}) for image in images]

    assert len(prompts) == len(images), "Only single image inference is supported."

    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        for prompt, image in zip(prompts, images)
    ]

    return inputs, samples


def pad_sequence(
    list_of_tensors: list[torch.Tensor],
    dim: int,
    padding_value: int,
):
    max_len = max(t.size(dim) for t in list_of_tensors)

    padded_list = []
    for t in list_of_tensors:
        pad_len = max_len - t.size(dim)
        t_padded = F.pad(t, (0, pad_len), value=padding_value)
        padded_list.append(t_padded)

    padded_sequence = torch.cat(padded_list, dim=0)
    return padded_sequence
