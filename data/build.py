import os
from typing import List, Dict, Any

import torch
from fvcore.common.registry import Registry
from torch.utils.data import DataLoader

from common.misc import default_collate
from model.streamvggt.processing import (
    build_online_vggt_input,
    resolve_dense_vggt_input,
)

from data.video_sampler import VideoDistributedSampler


DATASET_REGISTRY = Registry('Dataset')


def get_dataset_leo(cfg, split, dataset_name):
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)
    return dataset


def make_collate_fn_qwen(processor, use_vggt=False, dense_vggt_input=False):

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages = [data_dict.pop('messages') for data_dict in batch]
        inputs = processor.apply_chat_template(
            messages,
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

        labels = inputs['input_ids'].clone()
        for i in range(len(labels)):
            # hardcode <|im_start|>, followed by `assistant` and `\n`
            output_start_index = (labels[i] == 151644).nonzero()[-1, 0] + 3
            labels[i, :output_start_index] = -100
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs.update({'labels': labels})

        inputs.update(default_collate(batch))

        return inputs

    return collate_fn


def build_dataloader_leo(cfg, split, dataset_name, dataloader_args):
    dataset = get_dataset_leo(cfg, split, dataset_name)
    use_vggt = 'vggt' in cfg.vision3d.name.lower()
    dense_vggt_input = resolve_dense_vggt_input(cfg.data.vln) if use_vggt else False
    if split == "train" and dataset.use_custom_sampler:
        sampler = VideoDistributedSampler(dataset=dataset, drop_last=True, num_streams=dataloader_args.batchsize,
                                          seed=cfg.rng_seed, log_dir=os.path.join(cfg.exp_dir, "sampler_logs"))
    else:
        sampler = None
    return DataLoader(
        dataset, batch_size = dataloader_args.batchsize, num_workers = dataloader_args.num_workers,
        collate_fn = make_collate_fn_qwen(dataset.processor, use_vggt=use_vggt, dense_vggt_input=dense_vggt_input), pin_memory = True,
        shuffle = True if split == 'train' and sampler is None else False, drop_last = True if split == 'train' else False,
        sampler=sampler
    )
