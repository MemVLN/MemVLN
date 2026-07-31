import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Any
from einops import rearrange
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack

from .modeling_qwen3_vl import (
    Qwen3VLModel_VGGT,
    Qwen3VLForConditionalGeneration_VGGT,
    Qwen3VLModelOutputWithPast
)


class Qwen3VLModel_Mem_VGGT(Qwen3VLModel_VGGT):
    """
    Inherits from VGGT Model to kepp VGGT helper methods (fuse/get_embeds).
    We manually add the Memory-specific methods and override forward.
    """
    # ------------------------------------------------------------------
    # 1. Add Memory Specific Methods (Copied from Qwen3VLModel_Mem)
    # ------------------------------------------------------------------
    @torch.no_grad
    def split_prev_current_features(
        self,
        video_embeds: Tuple[torch.Tensor, ...],
        video_grid_thw: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B = len(video_embeds)
        if video_grid_thw.shape[0] != B:
            raise ValueError(f"Number of video_embeds should match video_grid_thw. Got {B} vs {video_grid_thw.shape[0]}")
        
        prev_list: List[torch.Tensor] = []
        curr_list: List[torch.Tensor] = []

        for i, emb in enumerate(video_embeds):
            t, h, w = map(int, video_grid_thw[i].tolist())
            
            if not torch.is_tensor(emb):
                raise ValueError(f"Each video_embed should be a tensor. Got {type(emb)}")
            
            if emb.ndim == 2:
                total_tokens, C = emb.shape
                emb2d = emb
            elif emb.ndim == 3:
                if emb.shape[0] == t and emb.shape[1] == h * w:
                    emb2d = emb.reshape(t * h * w, emb.shape[2])
                    total_tokens, C = emb2d.shape
                else:
                    raise ValueError(f"Unexpected video_embed shape: {emb.shape} for grid_thw: {t,h,w}")
            elif emb.ndim == 4:
                if emb.shape[0] != t or emb.shape[1] != h or emb.shape[2] != w:
                    raise ValueError(f"Unexpected video_embed shape: {emb.shape} for grid_thw: {t,h,w}")
                emb2d = emb.reshape(t * h * w, emb.shape[3])
                total_tokens, C = emb2d.shape
            else:
                raise ValueError(f"Unexpected video_embed ndim: {emb.ndim}")
            
            if total_tokens % t != 0:
                raise ValueError(f"Total tokens {total_tokens} not divisible by time {t}")
            per_frame = total_tokens // t

            frames = torch.split(emb2d, per_frame, dim=0)
            curr = frames[-1]
            if t == 1:
                prev = emb2d.new_empty((0, C))
            else:
                prev = torch.cat(frames[:-1], dim=0)

            prev_list.append(prev)
            curr_list.append(curr)

        curr_batch = torch.stack(curr_list, dim=0)

        return prev_list, curr_batch
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
        memory_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_memory_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.memory_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_memory_mask = special_memory_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_memory_mask = input_ids == self.memory_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )
        
        n_memory_tokens = special_memory_mask.sum()
        special_memory_mask = special_memory_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if memory_features is not None and inputs_embeds[special_memory_mask].numel() != memory_features.numel():
            raise ValueError(
                f"Memory features and memory tokens do not match: tokens: {n_memory_tokens}, features {memory_features.shape[0]}"
            )

        return special_image_mask, special_video_mask, special_memory_mask
    
    # ------------------------------------------------------------------
    # 2. Override Forward Method to sequence VGGT fusion -> Memory Split
    # ------------------------------------------------------------------
    @auto_docstring
    @check_model_inputs
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            # Mem args
            episode_id: Optional[torch.LongTensor] = None,
            clip_id: Optional[torch.LongTensor] = None,
            current_odom: Optional[torch.Tensor] = None,
            # VGGT args
            vggt_input: Optional[List[torch.Tensor]] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        episode_id (`list[str]`, *optional*):
            Episode identifiers used to isolate memory and StreamVGGT caches.
        clip_id (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Current physical observation index for each sample.
        current_odom (`torch.Tensor` of shape `(batch_size, 3)`, *optional*):
            Current relative planar position and yaw for each sample.
        vggt_input (`list[torch.Tensor]`, *optional*):
            Online StreamVGGT frame groups shaped `(frames, 3, height, width)`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        vggt_embeds = None
        curr_feat = None
        clip_ids_vggt = None
        if vggt_input is not None:
            if episode_id is None or clip_id is None:
                raise ValueError("StreamVGGT requires episode_id and clip_id")
            clip_ids_vggt = self._build_stream_clip_ids(vggt_input, clip_id)

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)

            # -- get vggt features --
            if vggt_input is not None:
                vggt_embeds = self.get_vggt_embeds(
                    media_input=vggt_input,
                    media_type='image',
                    episode_id=episode_id,
                    clip_ids=clip_ids_vggt,
                )
            else:
                vggt_embeds = None

            # -- fuse vggt features with image features --
            image_embeds = torch.stack(image_embeds, dim=0)
            curr_feat = image_embeds.clone()
            if vggt_embeds is not None:
                if isinstance(vggt_embeds, tuple):
                    image_embeds = self.fuse_vggt_embeds(image_embeds, vggt_embeds[1])
                else:
                    image_embeds = self.fuse_vggt_embeds(image_embeds, vggt_embeds)
                if self.memory_merge_vggt_feat:
                    curr_feat = image_embeds.clone()

            # -- memory module --
            # currently, only one image per sample, so len(image_embeds) == batch_size
            global_env_memory = self.memory(curr_feat, episode_id, clip_id, current_odom)
            global_env_memory_embeds = global_env_memory.reshape(-1, global_env_memory.shape[-1]).to(inputs_embeds.device, inputs_embeds.dtype)

            image_embeds = rearrange(image_embeds, 'b n d -> (b n) d').to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, memory_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds, memory_features=global_env_memory_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(memory_mask, global_env_memory_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)

            # -- vggt module --
            if vggt_embeds is None:
                if vggt_input is not None:
                    vggt_embeds = self.get_vggt_embeds(
                        media_input=vggt_input,
                        media_type='video',
                        episode_id=episode_id,
                        clip_ids=clip_ids_vggt,
                    )
                else:
                    vggt_embeds = None

                if vggt_embeds is not None:
                    # -- fuse vggt features with video features --
                    video_embeds = torch.stack(video_embeds, dim=0)
                    video_embeds = self.fuse_vggt_embeds(
                        video_embeds,
                        self._select_stream_vggt_embeds(
                            vggt_embeds, "video"
                        ),
                    )
            elif isinstance(vggt_embeds, tuple):
                # -- fuse vggt features with video features --
                video_embeds = torch.stack(video_embeds, dim=0)
                video_embeds = self.fuse_vggt_embeds(
                    video_embeds,
                    self._select_stream_vggt_embeds(
                        vggt_embeds, "video"
                    ),
                )

            # -- memory module --
            if curr_feat is None:
                _, curr_feat = self.split_prev_current_features(video_embeds, video_grid_thw)
                global_env_memory = self.memory(curr_feat, episode_id, clip_id, current_odom)
                global_env_memory_embeds = global_env_memory.reshape(-1, global_env_memory.shape[-1]).to(inputs_embeds.device, inputs_embeds.dtype)

            if isinstance(video_embeds, torch.Tensor):
                video_embeds = rearrange(video_embeds, 'b n d -> (b n) d').to(inputs_embeds.device, inputs_embeds.dtype)
            else:
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, memory_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds, memory_features=global_env_memory_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(memory_mask, global_env_memory_embeds)
        
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if self.training:
            # memory augmentation: random masking visual obs
            is_aux = kwargs.get('is_aux', [False] * visual_pos_masks.shape[0])
            for i in range(visual_pos_masks.shape[0]):
                if is_aux[i]:
                    if torch.rand(1).item() < self.visual_obs_mask_ratio_aux:
                        attention_mask[i].masked_fill_(visual_pos_masks[i], 0)
                else:
                    if torch.rand(1).item() < self.visual_obs_mask_ratio_nav:
                        attention_mask[i].masked_fill_(visual_pos_masks[i], 0)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration_Mem_VGGT(Qwen3VLForConditionalGeneration_VGGT):
    """
    Inherits from VGGT Generation Class.
    """
    def __init__(self, config):
        super(Qwen3VLForConditionalGeneration_VGGT, self).__init__(config)
        self.model = Qwen3VLModel_Mem_VGGT(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def _expand_inputs_for_generation(
            self,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            input_ids: Optional[torch.LongTensor] = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, dict]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs
        
        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "episode_id",
            "clip_id",
            "current_odom",
            "vggt_input",
        ]
        
        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    # samples = torch.split(video_grid_thw, list(video_nums))
                    samples = torch.split(video_grid_thw, 1)   # hardcode for temporary fix
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    # lengths = list(video_nums)
                    lengths = 1   # hardcode for temporary fix
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
                elif key == "episode_id":
                    assert len(dict_to_expand[key]) == 1
                    repeated_episode_ids = []
                    for i in range(expand_size):
                        repeated_episode_ids.append(dict_to_expand[key][0] + f"_{i}")
                    dict_to_expand[key] = repeated_episode_ids
                elif key == "clip_id":
                    assert len(dict_to_expand[key]) == 1
                    repeated_clip_ids = []
                    for i in range(expand_size):
                        repeated_clip_ids.append(dict_to_expand[key][0])
                    dict_to_expand[key] = repeated_clip_ids
                elif key == "current_odom":
                    assert len(dict_to_expand[key]) == 1
                    repeated_current_odoms = []
                    for i in range(expand_size):
                        repeated_current_odoms.append(dict_to_expand[key][0])
                    dict_to_expand[key] = repeated_current_odoms
                elif key == "vggt_input":
                    lengths = 1
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand
        
        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

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
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        if cache_position[0] != 0:
            model_inputs["vggt_input"] = None
        else:
            model_inputs["vggt_input"] = vggt_input
        model_inputs['episode_id'] = episode_id
        model_inputs['clip_id'] = clip_id
        model_inputs['current_odom'] = current_odom
        return model_inputs


__all__ = [
    "Qwen3VLModel_Mem_VGGT",
    "Qwen3VLForConditionalGeneration_Mem_VGGT",
]
