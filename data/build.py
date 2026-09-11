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
from data.probabilistic_episode_sampler import ProbabilisticEpisodeSampler


DATASET_REGISTRY = Registry('Dataset')


def get_dataset(cfg, split, dataset_name):
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


def build_training_sampler(cfg, dataset, batchsize):
    sampling = getattr(cfg.task.dataset_mix.VLNDatasetMix, 'sampling', None)
    sampling_mode = getattr(sampling, 'mode', 'epoch')
    max_updates = getattr(cfg.training, 'max_updates', None)
    if sampling_mode not in ('epoch', 'probabilistic_episode'):
        raise ValueError(f"Unsupported episode sampling mode: {sampling_mode}")
    if sampling_mode == 'probabilistic_episode':
        if not dataset.use_custom_sampler:
            raise ValueError("probabilistic_episode requires use_custom_sampler=true")
        if batchsize != 1:
            raise ValueError("probabilistic_episode requires per-GPU batchsize=1 (one episode stream per rank)")
        if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0:
            raise ValueError("probabilistic_episode requires training.max_updates to be a positive integer")
        accumulation = cfg.training.get('gradient_accumulation_steps', 1)
        if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if getattr(cfg.task.training, 'dummy_lr_decay', False):
            raise ValueError("probabilistic_episode requires task.training.dummy_lr_decay=false")
        probabilities = getattr(sampling, 'dataset_probabilities', None)
        if probabilities is None:
            raise ValueError("probabilistic_episode requires sampling.dataset_probabilities")
        return ProbabilisticEpisodeSampler(
            dataset=dataset, num_samples=max_updates * accumulation,
            dataset_probabilities=dict(probabilities), seed=cfg.rng_seed,
        )
    if max_updates is not None:
        raise ValueError("training.max_updates requires sampling.mode=probabilistic_episode")
    if dataset.use_custom_sampler:
        return VideoDistributedSampler(
            dataset=dataset, drop_last=True, num_streams=batchsize,
            seed=cfg.rng_seed, log_dir=os.path.join(cfg.exp_dir, "sampler_logs"),
        )
    return None


def build_dataloader(cfg, split, dataset_name, dataloader_args):
    dataset = get_dataset(cfg, split, dataset_name)
    use_vggt = 'vggt' in cfg.vision3d.name.lower()
    dense_vggt_input = resolve_dense_vggt_input(cfg.data.vln) if use_vggt else False
    sampler = build_training_sampler(cfg, dataset, dataloader_args.batchsize) if split == 'train' else None
    return DataLoader(
        dataset, batch_size = dataloader_args.batchsize, num_workers = dataloader_args.num_workers,
        collate_fn = make_collate_fn_qwen(dataset.processor, use_vggt=use_vggt, dense_vggt_input=dense_vggt_input), pin_memory = True,
        shuffle = True if split == 'train' and sampler is None else False, drop_last = True if split == 'train' else False,
        sampler=sampler
    )
