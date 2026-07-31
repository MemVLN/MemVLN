"""Shared preprocessing for online StreamVGGT inputs."""

from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor

from model.qwen.processing_qwen3_vl import (
    extract_image_info,
    extract_image_info_from_video,
)


def _cfg_value(cfg: Any, name: str) -> Any:
    if isinstance(cfg, Mapping):
        if name not in cfg:
            raise ValueError(f"Missing required VLN configuration field: {name}")
        return cfg[name]
    if not hasattr(cfg, name):
        raise ValueError(f"Missing required VLN configuration field: {name}")
    return getattr(cfg, name)


def resolve_dense_vggt_input(vln_cfg: Any) -> bool:
    """Resolve whether StreamVGGT must fill observations skipped by merged actions."""
    dense = (
        bool(_cfg_value(vln_cfg, "merge_action"))
        and not bool(_cfg_value(vln_cfg, "skip_merge_step_obs"))
    )
    if dense and not bool(_cfg_value(vln_cfg, "vggt_cache_feat")):
        raise ValueError(
            "Merged actions with skip_merge_step_obs=false require "
            "data.vln.vggt_cache_feat=true so pending observations can be "
            "deduplicated and retained by StreamVGGT."
        )
    if dense and int(_cfg_value(vln_cfg, "max_num_frames")) < 3:
        raise ValueError(
            "Dense StreamVGGT input requires data.vln.max_num_frames>=3 "
            "to preserve two pending observations plus the current frame."
        )
    if dense and _cfg_value(vln_cfg, "obs_hist_sample") != "nearest":
        raise ValueError(
            "Dense StreamVGGT input requires data.vln.obs_hist_sample=nearest "
            "so clip IDs remain consecutive."
        )
    return dense


def _conversation_batch(messages: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("StreamVGGT messages must be a non-empty conversation or batch.")
    if isinstance(messages[0], dict):
        return [messages]
    if not all(isinstance(conversation, list) and conversation for conversation in messages):
        raise ValueError("Each StreamVGGT batch item must be a non-empty conversation.")
    return messages


def build_online_vggt_input(
    processor: Any,
    messages: Any,
    image_grid_thw: Any,
    dense_history: bool = True,
) -> list[torch.Tensor]:
    """Build per-sample ``[frames, 3, height, width]`` online VGGT inputs.

    Dense history adds at most the last two video frames before the current
    image. This fills the largest temporal gap produced by the project's
    three-action merge limit while StreamVGGT keeps the full episode cache.
    """
    conversations = _conversation_batch(messages)
    batch_size = len(conversations)

    current_images = []
    for sample_index, conversation in enumerate(conversations):
        sample_images = extract_image_info(conversation)
        if sample_images is None or len(sample_images) != 1:
            found = 0 if sample_images is None else len(sample_images)
            raise ValueError(
                "StreamVGGT requires exactly one current image per sample; "
                f"sample {sample_index} has {found}."
            )
        current_images.append(sample_images[0])
    if not all(isinstance(image, Image.Image) for image in current_images):
        raise TypeError("StreamVGGT current observations must resolve to PIL images.")

    if image_grid_thw is None:
        raise ValueError("StreamVGGT requires image_grid_thw from the Qwen processor.")
    try:
        grid_batch_size = len(image_grid_thw)
    except TypeError as exc:
        raise ValueError("image_grid_thw must contain one grid per sample.") from exc
    if grid_batch_size != batch_size:
        raise ValueError(
            "StreamVGGT image grid and message batch sizes differ: "
            f"{grid_batch_size} grids for {batch_size} samples."
        )

    if dense_history:
        history_images = extract_image_info_from_video(messages, max_frames=2)
        if len(history_images) != batch_size:
            raise ValueError(
                "StreamVGGT history and message batch sizes differ: "
                f"{len(history_images)} histories for {batch_size} samples."
            )
    else:
        history_images = [[] for _ in range(batch_size)]

    frame_groups = []
    for sample_index, (history, current) in enumerate(
        zip(history_images, current_images)
    ):
        frames = [*history[-2:], current]
        if not all(isinstance(image, Image.Image) for image in frames):
            raise TypeError(
                f"StreamVGGT sample {sample_index} contains a non-PIL observation."
            )
        frame_groups.append(torch.stack([pil_to_tensor(image) for image in frames]))

    video_processor = getattr(processor, "video_processor", None)
    preprocess = getattr(video_processor, "vggt_preprocess", None)
    if not callable(preprocess):
        raise ValueError(
            "Processor is not initialized for StreamVGGT; call "
            "processor.fix_video_preprocess(use_vggt=True) first."
        )

    processed = preprocess(
        videos=frame_groups,
        video_grid_thw=image_grid_thw,
        patch_size=14,
        temporal_patch_size=2,
        temporal_pad=False,
    )
    if not isinstance(processed, list) or len(processed) != batch_size:
        raise ValueError(
            "StreamVGGT preprocessing must return one tensor per sample; "
            f"got {type(processed).__name__}."
        )
    for sample_index, tensor in enumerate(processed):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(
                "StreamVGGT preprocessing must return tensors shaped "
                f"[frames, 3, height, width]; sample {sample_index} is invalid."
            )
        if tensor.shape[0] != len(frame_groups[sample_index]) or tensor.shape[1] != 3:
            raise ValueError(
                f"StreamVGGT sample {sample_index} changed frame/channel dimensions "
                f"unexpectedly: {tuple(tensor.shape)}."
            )
    return processed
