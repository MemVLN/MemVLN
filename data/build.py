import os
from typing import List, Dict, Any

import torch
from fvcore.common.registry import Registry
from torch.utils.data import DataLoader

from common.misc import default_collate
from model.qwen.factory import (
    apply_chat_template,
    count_multimodal_tokens,
    mask_labels_for_last_assistant,
    resolve_model_family,
)
from model.streamvggt.processing import (
    build_online_vggt_input,
    resolve_dense_vggt_input,
)

from data.video_sampler import VideoDistributedSampler


DATASET_REGISTRY = Registry('Dataset')


def get_dataset_leo(cfg, split, dataset_name):
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)
    return dataset


def make_collate_fn_qwen(
    processor,
    model_family="qwen3_vl",
    use_vggt=False,
    dense_vggt_input=False,
):

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages = [data_dict.pop('messages') for data_dict in batch]
        inputs = apply_chat_template(
            processor,
            messages,
            model_family,
            tokenize=True,
            add_generation_prompt=False if len(messages[0]) > 1 else True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )
        if use_vggt:
            inputs['vggt_input'] = build_online_vggt_input(
                processor,
                messages,
                inputs["image_grid_thw"],
                dense_history=dense_vggt_input,
            )

        labels = mask_labels_for_last_assistant(
            inputs['input_ids'],
            processor.tokenizer,
            attention_mask=inputs.get('attention_mask'),
        )
        inputs.update({'labels': labels})
        if model_family == "qwen2_5_vl":
            inputs.update(count_multimodal_tokens(inputs['input_ids'], processor))

        inputs.update(default_collate(batch))

        return inputs

    return collate_fn


def build_dataloader_leo(cfg, split, dataset_name, dataloader_args):
    dataset = get_dataset_leo(cfg, split, dataset_name)
    model_family = resolve_model_family(cfg)
    use_vggt = 'vggt' in cfg.vision3d.name.lower()
    dense_vggt_input = resolve_dense_vggt_input(cfg.data.vln) if use_vggt else False
    if split == "train" and dataset.use_custom_sampler:
        sampler = VideoDistributedSampler(dataset=dataset, drop_last=True, num_streams=dataloader_args.batchsize,
                                          seed=cfg.rng_seed, log_dir=os.path.join(cfg.exp_dir, "sampler_logs"))
    else:
        sampler = None
    return DataLoader(
        dataset, batch_size = dataloader_args.batchsize, num_workers = dataloader_args.num_workers,
        collate_fn = make_collate_fn_qwen(
            dataset.processor,
            model_family=model_family,
            use_vggt=use_vggt,
            dense_vggt_input=dense_vggt_input,
        ), pin_memory = True,
        shuffle = True if split == 'train' and sampler is None else False, drop_last = True if split == 'train' else False,
        sampler=sampler
    )
