# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from ..layers import PatchEmbed
from ..layers.block import Block
from ..layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ..layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

        self.past_key_values = {}
        self.past_clip_ids = {}
        self.past_feats = {}

    def update_past_key_values(self, episode_id, past_key_values):
        # add
        for i, ep_id in enumerate(episode_id):
            if ep_id not in self.past_key_values:
                self.past_key_values[ep_id] = [None] * self.depth
            for j, layer_kv in enumerate(past_key_values):
                layer_k, layer_v = layer_kv
                # each k/v is at shape [batch_size, num_heads, num_frames, tokens_per_frame, head_dim]
                if layer_k.shape[2] <= self.max_num_frames:
                    layer_k_ep = layer_k[i, :, :, :, :]
                    layer_v_ep = layer_v[i, :, :, :, :]
                else:
                    layer_k_ep = layer_k[i, :, -(self.max_num_frames-1):, :, :]
                    layer_v_ep = layer_v[i, :, -(self.max_num_frames-1):, :, :]
                    layer_k_ep_0 = layer_k[i, :, :1, :, :]
                    layer_v_ep_0 = layer_v[i, :, :1, :, :]
                    layer_k_ep = torch.cat([layer_k_ep_0, layer_k_ep], dim=1)
                    layer_v_ep = torch.cat([layer_v_ep_0, layer_v_ep], dim=1)
                self.past_key_values[ep_id][j] = (layer_k_ep, layer_v_ep)

        # prune
        existing_ep_ids = list(self.past_key_values.keys())
        for existing_ep_id in existing_ep_ids:
            if existing_ep_id not in episode_id:
                del self.past_key_values[existing_ep_id]

    def update_past_feats(self, episode_id, output_feats, clip_ids=None):
        # add
        for i, ep_id in enumerate(episode_id):
            if ep_id not in self.past_feats:
                self.past_feats[ep_id] = []
            if ep_id not in self.past_clip_ids:
                self.past_clip_ids[ep_id] = []
            for _id, feat in zip(clip_ids, output_feats[i]):
                if _id not in self.past_clip_ids[ep_id]:
                    self.past_clip_ids[ep_id].append(_id)
                    self.past_feats[ep_id].append(feat)
            self.past_clip_ids[ep_id] = self.past_clip_ids[ep_id][-self.max_num_frames :]
            self.past_feats[ep_id] = self.past_feats[ep_id][-self.max_num_frames :]

        # prune
        existing_ep_ids = list(self.past_feats.keys())
        for existing_ep_id in existing_ep_ids:
            if existing_ep_id not in episode_id:
                del self.past_feats[existing_ep_id]
                del self.past_clip_ids[existing_ep_id]

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        episode_id=None,
        clip_ids=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        if isinstance(episode_id, torch.Tensor):
            episode_id = episode_id.tolist()

        if isinstance(images, list):
            assert len(images) == 1 and len(clip_ids) == 1 and len(images[0]) == len(clip_ids[0])
            images_forward = []
            clip_ids_forward = []
            for image, _id in zip(images[0], clip_ids[0]):
                if _id not in self.past_clip_ids.get(episode_id[0], []):
                    images_forward.append(image)
                    clip_ids_forward.append(_id)

            if len(images_forward) == 0:
                output_list = [torch.stack(self.past_feats[episode_id[0]], dim=0).unsqueeze(0)]
                # (Batch, Frame, Token_num, vggt_dim)
                if use_cache:
                    return output_list, self.patch_start_idx, past_key_values
                return output_list, self.patch_start_idx

            images = torch.stack(images_forward, dim=0).unsqueeze(0)
        else:
            clip_ids_forward = clip_ids[0] if clip_ids is not None else None

        B, S, C_in, H, W = images.shape

        if use_cache and past_key_values is None:
            if not all(ep_id in self.past_key_values for ep_id in episode_id):
                past_key_values = [None] * self.depth
            else:
                past_key_values = []
                for i in range(self.depth):
                    layer_k = [self.past_key_values[ep_id][i][0] for ep_id in episode_id]
                    layer_k = torch.stack(layer_k, dim=0)
                    layer_v = [self.past_key_values[ep_id][i][1] for ep_id in episode_id]
                    layer_v = torch.stack(layer_v, dim=0)
                    past_key_values.append((layer_k, layer_v))

        if use_cache and past_key_values is not None and past_key_values[0] is not None:
            _, _, past_frames, _, _ = past_key_values[0][0].shape
            S_true = past_frames + S
        else:
            S_true = S

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if self._resnet_mean.device != self.patch_embed.norm.weight.device:
            self._resnet_mean = self._resnet_mean.to(self.patch_embed.norm.weight.device)
        if self._resnet_std.device != self.patch_embed.norm.weight.device:
            self._resnet_std = self._resnet_std.to(self.patch_embed.norm.weight.device)
        if self._resnet_mean.dtype != self.patch_embed.norm.weight.dtype:
            self._resnet_mean = self._resnet_mean.to(self.patch_embed.norm.weight.dtype)
        if self._resnet_std.dtype != self.patch_embed.norm.weight.dtype:
            self._resnet_std = self._resnet_std.to(self.patch_embed.norm.weight.dtype)

        # Normalize images and reshape for patch embed
        images = images.to(self._resnet_mean.device)
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :].expand(patch_tokens.shape[0], -1, -1)
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :].expand(patch_tokens.shape[0], -1, -1)
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]
                        tokens, global_idx, global_intermediates, new_kv = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            tokens_per_frame=P,
                        )
                        past_key_values[global_idx - 1] = new_kv
                    else: 
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        # update past_key_values
        if use_cache:
            self.update_past_key_values(episode_id, past_key_values)

        # update past_feats
        if self.cache_feat:
            self.update_past_feats(episode_id, output_list[-1], clip_ids=clip_ids_forward)
            ep_feats = []
            for ep_id in episode_id:
                ep_feats.append(torch.stack(self.past_feats[ep_id], dim=0))
            output_list = [torch.stack(ep_feats, dim=0)]
        else:
            output_list = [output_list[-1][:, -1:, :, :]]

        if use_cache:
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0,
        tokens_per_frame=None,
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None

            if use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    tokens_per_frame=tokens_per_frame,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
