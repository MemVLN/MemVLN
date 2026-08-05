"""Qwen2.5-VL with the project-specific Memory and StreamVGGT hooks.

The backbone, multimodal RoPE, cache, and generation behavior stay in the
Transformers 4.57 Qwen2.5-VL implementation.  This module only prepares fused
visual/memory embeddings before delegating back to the official model forward.
Memory and StreamVGGT modules are attached by :class:`LeoAgent` after loading
the base checkpoint, so this model does not initialize either module itself.
"""

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .modeling_qwen3_vl import Qwen3VLModel_VGGT
from .modeling_qwen3_vl_mem_vggt import Qwen3VLModel_Mem_VGGT


class Qwen2_5_VLModel_Mem_VGGT(Qwen2_5_VLModel):
    """Official Qwen2.5-VL model with thin, externally attached MemVLN hooks."""

    # Reuse the existing project-only helpers verbatim.  None of these methods
    # calls the Qwen3 backbone or its DeepStack implementation.
    split_prev_current_features = Qwen3VLModel_Mem_VGGT.split_prev_current_features
    get_placeholder_mask = Qwen3VLModel_Mem_VGGT.get_placeholder_mask
    _build_stream_clip_ids = staticmethod(Qwen3VLModel_VGGT._build_stream_clip_ids)
    get_vggt_embeds = Qwen3VLModel_VGGT.get_vggt_embeds
    _select_stream_vggt_embeds = staticmethod(
        Qwen3VLModel_VGGT._select_stream_vggt_embeds
    )
    fuse_vggt_embeds = Qwen3VLModel_VGGT.fuse_vggt_embeds

    def _validate_attached_modules(self, use_vggt: bool) -> None:
        required = ["memory", "memory_token_id"]
        if use_vggt:
            required.extend(
                ["vggt", "vggt_embedding_merger", "merger_type", "fusion_op"]
            )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "Qwen2.5 MemVLN modules must be attached by LeoAgent before "
                f"forward; missing: {', '.join(missing)}"
            )

    @staticmethod
    def _stack_visual_features(
        features: Tuple[torch.Tensor, ...], media_type: str
    ) -> torch.Tensor:
        if not features:
            raise ValueError(f"Qwen2.5 returned no {media_type} features")
        expected_shape = features[0].shape
        if any(feature.shape != expected_shape for feature in features[1:]):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(
                f"StreamVGGT fusion requires equal-length {media_type} features "
                f"within a batch, got {shapes}"
            )
        return torch.stack(features, dim=0)

    @staticmethod
    def _flatten_visual_features(
        features: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            if features.ndim != 3:
                raise ValueError(
                    "Batched visual features must have shape (batch, tokens, hidden), "
                    f"got {tuple(features.shape)}"
                )
            return features.reshape(-1, features.shape[-1])
        return torch.cat(features, dim=0)

    def _mask_visual_observations(
        self,
        attention_mask: Optional[torch.Tensor],
        visual_pos_masks: Optional[torch.Tensor],
        is_aux: Optional[List[bool]],
    ) -> None:
        if not self.training or attention_mask is None or visual_pos_masks is None:
            return
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            raise ValueError(
                "Qwen2.5 visual observation masking requires a 2D tensor attention_mask"
            )

        batch_size = visual_pos_masks.shape[0]
        if is_aux is None:
            is_aux = [False] * batch_size
        if len(is_aux) != batch_size:
            raise ValueError(
                f"is_aux has {len(is_aux)} entries for a batch of {batch_size}"
            )

        nav_ratio = getattr(self, "visual_obs_mask_ratio_nav", 0.0)
        aux_ratio = getattr(self, "visual_obs_mask_ratio_aux", 1.0)
        for batch_idx in range(batch_size):
            ratio = aux_ratio if bool(is_aux[batch_idx]) else nav_ratio
            if torch.rand(1).item() < ratio:
                attention_mask[batch_idx].masked_fill_(
                    visual_pos_masks[batch_idx], 0
                )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        episode_id: Optional[List[Any]] = None,
        clip_id: Optional[torch.LongTensor] = None,
        current_odom: Optional[torch.Tensor] = None,
        vggt_input: Optional[List[torch.Tensor]] = None,
        is_aux: Optional[List[bool]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        has_visual_input = pixel_values is not None or pixel_values_videos is not None
        if has_visual_input:
            self._validate_attached_modules(use_vggt=vggt_input is not None)
            missing_context = [
                name
                for name, value in (
                    ("episode_id", episode_id),
                    ("clip_id", clip_id),
                    ("current_odom", current_odom),
                )
                if value is None
            ]
            if missing_context:
                raise ValueError(
                    "Memory forward requires " + ", ".join(missing_context)
                )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        current_features = None
        memory_features = None
        vggt_embeds = None
        stream_clip_ids = None

        if vggt_input is not None:
            if episode_id is None or clip_id is None:
                raise ValueError("StreamVGGT requires episode_id and clip_id")
            stream_clip_ids = self._build_stream_clip_ids(vggt_input, clip_id)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            if vggt_input is not None:
                vggt_embeds = self.get_vggt_embeds(
                    media_input=vggt_input,
                    media_type="image",
                    episode_id=episode_id,
                    clip_ids=stream_clip_ids,
                )

            image_embeds = self._stack_visual_features(image_embeds, "image")
            current_features = image_embeds.clone()
            if vggt_embeds is not None:
                image_embeds = self.fuse_vggt_embeds(
                    image_embeds,
                    self._select_stream_vggt_embeds(vggt_embeds, "image"),
                )
                if getattr(self, "memory_merge_vggt_feat", False):
                    current_features = image_embeds.clone()

            memory_features = self.memory(
                current_features, episode_id, clip_id, current_odom
            )
            memory_features = memory_features.reshape(
                -1, memory_features.shape[-1]
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            flat_image_embeds = self._flatten_visual_features(image_embeds).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _, memory_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=flat_image_embeds,
                memory_features=memory_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, flat_image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                memory_mask, memory_features
            )

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )

            if vggt_embeds is None and vggt_input is not None:
                vggt_embeds = self.get_vggt_embeds(
                    media_input=vggt_input,
                    media_type="video",
                    episode_id=episode_id,
                    clip_ids=stream_clip_ids,
                )
                if vggt_embeds is not None:
                    video_embeds = self._stack_visual_features(
                        video_embeds, "video"
                    )
                    video_embeds = self.fuse_vggt_embeds(
                        video_embeds,
                        self._select_stream_vggt_embeds(vggt_embeds, "video"),
                    )
            elif isinstance(vggt_embeds, tuple):
                video_embeds = self._stack_visual_features(video_embeds, "video")
                video_embeds = self.fuse_vggt_embeds(
                    video_embeds,
                    self._select_stream_vggt_embeds(vggt_embeds, "video"),
                )

            if current_features is None:
                _, current_features = self.split_prev_current_features(
                    video_embeds, video_grid_thw
                )
                memory_features = self.memory(
                    current_features, episode_id, clip_id, current_odom
                )
                memory_features = memory_features.reshape(
                    -1, memory_features.shape[-1]
                ).to(inputs_embeds.device, inputs_embeds.dtype)

            flat_video_embeds = self._flatten_visual_features(video_embeds).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask, memory_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=flat_video_embeds,
                memory_features=memory_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, flat_video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                memory_mask, memory_features
            )

        visual_pos_masks = None
        if image_mask is not None and video_mask is not None:
            visual_pos_masks = image_mask[..., 0] | video_mask[..., 0]
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
        self._mask_visual_observations(
            attention_mask, visual_pos_masks, is_aux
        )

        # The official Qwen2.5-VL forward remains responsible for multimodal
        # RoPE (including second_per_grid_ts), cache handling, and the decoder.
        # Pixel tensors are omitted here because their embeddings were already
        # computed and inserted above.
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )


class Qwen2_5_VLForConditionalGeneration_Mem_VGGT(
    Qwen2_5_VLForConditionalGeneration
):
    """Conditional-generation wrapper using the thin Mem+VGGT model above."""

    def __init__(self, config):
        # Skip the official wrapper's model construction so the 7B backbone is
        # never allocated twice.  The modules and names otherwise match it.
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen2_5_VLModel_Mem_VGGT(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        vggt_input=None,
        episode_id=None,
        clip_id=None,
        current_odom=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        is_prefill = (
            cache_position is None
            or cache_position.numel() == 0
            or cache_position[0].item() == 0
        )
        model_inputs["vggt_input"] = vggt_input if is_prefill else None
        model_inputs["episode_id"] = episode_id if is_prefill else None
        model_inputs["clip_id"] = clip_id if is_prefill else None
        model_inputs["current_odom"] = current_odom if is_prefill else None
        return model_inputs


__all__ = [
    "Qwen2_5_VLModel_Mem_VGGT",
    "Qwen2_5_VLForConditionalGeneration_Mem_VGGT",
]
