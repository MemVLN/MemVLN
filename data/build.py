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


def _safe_repr(value: Any, limit: int = 512) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        text = f"<repr failed: {type(exc).__name__}>"
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _sample_context(
    sample: Dict[str, Any],
    sample_index: int,
    *,
    token_length: int | None = None,
) -> str:
    messages = sample.get('messages', [])
    roles = [
        message.get('role', '<missing>')
        for message in messages
        if isinstance(message, dict)
    ]
    fields = [
        f"sample_index={sample_index}",
        f"source={_safe_repr(sample.get('source'))}",
        f"scene_id={_safe_repr(sample.get('scene_id'))}",
        f"episode_id={_safe_repr(sample.get('episode_id'))}",
        f"clip_id={_safe_repr(sample.get('clip_id'))}",
        f"roles={_safe_repr(roles)}",
        f"output_gt={_safe_repr(sample.get('output_gt'))}",
    ]
    if token_length is not None:
        fields.append(f"active_token_length={token_length}")
    return ", ".join(fields)


def _resolve_add_generation_prompt(
    batch: List[Dict[str, Any]],
    messages: List[List[Dict[str, Any]]],
) -> bool:
    if not messages or any(not conversation for conversation in messages):
        raise ValueError("Every sample must contain a non-empty conversation.")

    last_roles = [conversation[-1].get('role') for conversation in messages]
    supervised = [role == 'assistant' for role in last_roles]
    if any(supervised) and not all(supervised):
        contexts = '\n'.join(
            _sample_context(sample, sample_index)
            for sample_index, sample in enumerate(batch)
        )
        raise ValueError(
            "Cannot mix supervised and generation-only conversations in one "
            f"batch:\n{contexts}"
        )
    return not all(supervised)


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
        messages = [data_dict['messages'] for data_dict in batch]
        metadata_batch = [
            {
                key: value
                for key, value in data_dict.items()
                if key != 'messages'
            }
            for data_dict in batch
        ]
        add_generation_prompt = _resolve_add_generation_prompt(batch, messages)
        inputs = apply_chat_template(
            processor,
            messages,
            model_family,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
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

        try:
            labels = mask_labels_for_last_assistant(
                inputs['input_ids'],
                processor.tokenizer,
                attention_mask=inputs.get('attention_mask'),
            )
        except (TypeError, ValueError) as exc:
            attention_mask = inputs.get('attention_mask')
            contexts = []
            for sample_index, sample in enumerate(batch):
                token_length = None
                input_ids = inputs.get('input_ids')
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.ndim == 1 and sample_index == 0:
                        token_length = int(input_ids.shape[0])
                    elif input_ids.ndim >= 2 and sample_index < input_ids.shape[0]:
                        token_length = int(input_ids.shape[-1])
                if (
                    isinstance(attention_mask, torch.Tensor)
                    and attention_mask.ndim == 2
                    and sample_index < attention_mask.shape[0]
                ):
                    token_length = int(attention_mask[sample_index].sum().item())
                contexts.append(
                    _sample_context(
                        sample,
                        sample_index,
                        token_length=token_length,
                    )
                )
            raise ValueError(
                f"Qwen assistant label masking failed: {exc}\n"
                + "\n".join(contexts)
            ) from exc
        inputs.update({'labels': labels})
        if model_family == "qwen2_5_vl":
            inputs.update(count_multimodal_tokens(inputs['input_ids'], processor))

        inputs.update(default_collate(metadata_batch))

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
