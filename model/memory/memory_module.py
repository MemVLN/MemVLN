import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
from collections import OrderedDict

from model.memory.tome import *
from model.memory.time_embedding import TimeEmbedding
from model.memory.position_embedding import PositionEmbedding


# --------------------------------------------------------------------------
# 1. CrossTransformerBlock (Deep Interaction)
# --------------------------------------------------------------------------
def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


class CrossTransformerBlock(nn.Module): # TODO: DETR variant
    def __init__(self, feature_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        
        hidden_dim = int(feature_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(), # GELU is generally better for Transformers
            nn.Linear(hidden_dim, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True):
        # 1. Attention + Residual + Post-Norm (Structure from memory_vla.py)
        attn_out, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm1(query + attn_out)
        
        # 2. FFN + Residual + Post-Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
    

# --------------------------------------------------------------------------
# 2. Memory Bank
# --------------------------------------------------------------------------
class ToMeMemoryBank(nn.Module):
    def __init__(self, config, inference_config=None):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim

        inference_enabled = bool(_cfg_get(inference_config, "enabled", False))
        self.inference_cache_enabled = inference_enabled and bool(
            _cfg_get(inference_config, "cache_grounded_memory", False)
        )
        self.skip_memory_attention_weights = inference_enabled and bool(
            _cfg_get(inference_config, "skip_memory_attention_weights", False)
        )
        
        # Hyperparameters
        self.frame_max_tokens_num = config.frame_max_tokens_num
        self.episode_max_frame_num = config.episode_max_frame_num
        self.memory_consolidate_type = config.memory_consolidate_type

        assert self.memory_consolidate_type in ('fifo', 'frme') # 'frme': 'frame merge'

        # Config for Embeddings
        self.use_pos_embed = config.use_pos_embed
        if self.use_pos_embed:
            self.pos_embed_fourier_feat_num = config.pos_embed_fourier_feat_num
            self.pos_embed_learnable = config.pos_embed_learnable
            self.pos_embed_mlp_hidden = config.pos_embed_mlp_hidden
            self.pos_embed_mapping_type = config.pos_embed_mapping_type
        self.use_time_embed = config.use_time_embed
        if self.use_time_embed:
            self.time_embed_max_length_num = config.time_embed_max_length_num

        # Config for Retrieval Layers
        self.retrieval_mha_heads_num = config.retrieval_mha_heads_num
        self.retrieval_layers_num = config.retrieval_layers_num
        
        # Config for Token Merging Layers
        self.tome_mha_heads_num = config.tome_mha_heads_num
        self.tome_layers_num = config.tome_layers_num
        
        # --- Option 2: Latent Query ---
        self.learnable_latent_query_num = config.memory_token_num # Qwen3: 300
        self.learnable_latent_query = nn.Parameter(torch.randn(1, self.learnable_latent_query_num, self.feature_dim) * 0.02)

        # --- Embeddings ---
        if self.use_time_embed:
            # Assumes updated TimeEmbedding from previous discussion
            self.time_embed = TimeEmbedding(self.use_time_embed, max_len=self.time_embed_max_length_num, dim=self.feature_dim)
        if self.use_pos_embed:
            # Assumes updated PositionEmbedding from previous discussion
            self.pos_embed = PositionEmbedding(self.use_pos_embed, dim=self.feature_dim, fourier_feats=self.pos_embed_fourier_feat_num, 
                                               learnable=self.pos_embed_learnable, mlp_hidden=self.pos_embed_mlp_hidden, mapping_type=self.pos_embed_mapping_type)

        # --- NEW: Deep Retrieval & Gating ---
        # Replaced single MultiheadAttention with ModuleList of CrossTransformerBlocks
        self.retrieval_blocks = nn.ModuleList([
            CrossTransformerBlock(self.feature_dim, self.retrieval_mha_heads_num)
            for _ in range(self.retrieval_layers_num)
        ])
        
        # Replaced Residual with GateFusion
        # self.gate_fusion = GateFusion(self.feature_dim)

        # ToMe Compression Blocks (for history)
        self.tome_blocks = nn.ModuleList([
            ToMeBlock(dim=self.feature_dim, num_heads=self.tome_mha_heads_num) 
            for _ in range(self.tome_layers_num)
        ])

        # Raw storage remains the source of truth for FIFO/FRME consolidation.
        self.history_memory: Dict[Any, Dict] = {}
        # Eval-only derived storage:
        # grounded_history_memory[episode_id][timestamp] =
        #     ToMe(raw_feat) + time_embed(timestamp) + pos_embed(odom)
        self.grounded_history_memory: Dict[Any, Dict] = {}
        self._grounded_cache_meta: Dict[Any, tuple] = {}

    def train(self, mode: bool = True):
        # A training pass may update both the raw FRME bank and the embedding
        # parameters without touching the eval-only derived cache.  Invalidate
        # it at the mode boundary so a later eval cannot reuse stale entries.
        if self.training != mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _apply(self, fn):
        result = super()._apply(fn)
        self.clear_inference_cache()
        if hasattr(self, "pos_embed"):
            self.pos_embed._cached_param_device = None
            self.pos_embed._cached_param_dtype = None
        return result

    def clear_inference_cache(self, episode_id=None):
        if episode_id is None:
            self.grounded_history_memory.clear()
            self._grounded_cache_meta.clear()
            return
        self.grounded_history_memory.pop(episode_id, None)
        self._grounded_cache_meta.pop(episode_id, None)

    def _compress_batch_clips_tensor(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Compresses a batch of frames using the CURRENT ToMe blocks.
        stacked: (Total_Frames, N, C)
        Returns: (Total_Frames, N_compressed, C)
        """
        target = self.frame_max_tokens_num
        x = stacked

        for block in self.tome_blocks:
            x_attn_in = block.norm1(x)
            x_attn_out, metric = block.attn(x_attn_in, size=None)
            x = x + x_attn_out

            need_remove = max(0, x.shape[1] - target)
            r = min(x.shape[1] // 2, need_remove)
            if r > 0:
                merge_fn, _ = bipartite_soft_matching(metric, r)
                x, _ = merge_wavg(merge_fn, x, None)
            
            x = block.forward_mlp(x)
            if x.shape[1] <= target:
                break
        
        return x
    
    @torch.no_grad()
    def _consolidate_with_frame_merge(self, episode_id):
        # Old version used list, new version uses dict to avoid duplicates
        # get list from dict values
        bank = list(self.history_memory.get(episode_id, {}).values())
        sorted_bank = sorted(bank, key=lambda x: x[0])  # sort by timestep
        T = len(bank)
        if T < 2:
            return None
        
        feats = [feat for (_, _, feat) in sorted_bank]
        all_feats = torch.stack(feats)
        feats_current = all_feats[:-1]
        feats_next = all_feats[1:]
        sims_per_feature = F.cosine_similarity(feats_current, feats_next, dim=-1)
        frame_sims = sims_per_feature.mean(dim=1)

        idx_max = torch.argmax(frame_sims).item()
        timestep_i, odometry_i, feat_i = sorted_bank[idx_max]
        timestep_j, odometry_j, feat_j = sorted_bank[idx_max + 1]

        fused_feat = 0.5 * (feat_i + feat_j)
        # Update the dict accordingly
        self.history_memory[episode_id][timestep_i] = (timestep_i, odometry_i, fused_feat.detach().clone())
        del self.history_memory[episode_id][timestep_j]
        return timestep_i, timestep_j

    @torch.no_grad()
    def _memory_consolidate(self, episode_id, feat: torch.Tensor, timestep=None, odometry=None):
        dirty_timesteps = set()
        removed_timesteps = set()

        # Store new observation
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
        if timestep in self.history_memory[episode_id]:
            pass  # Already stored
        else:
            self.history_memory[episode_id][timestep] = (timestep, odometry, feat.detach().clone())
            dirty_timesteps.add(timestep)

        # Re-sort the OrderedDict by timestep
        self.history_memory[episode_id] = OrderedDict(
            sorted(
                self.history_memory[episode_id].items(),
                key=lambda x: x[0].item() if isinstance(x[0], torch.Tensor) else x[0]
            )
        )

        # Compress memory
        while len(self.history_memory[episode_id]) > self.episode_max_frame_num:
            if self.memory_consolidate_type == 'fifo':
                # Delete smallest timestep
                removed_timestep, _ = self.history_memory[episode_id].popitem(last=False)
                removed_timesteps.add(removed_timestep)

            elif self.memory_consolidate_type == 'frme':
                merged = self._consolidate_with_frame_merge(episode_id)
                if merged is None:
                    raise RuntimeError("FRME consolidation requires at least two frames")
                updated_timestep, removed_timestep = merged
                dirty_timesteps.add(updated_timestep)
                removed_timesteps.add(removed_timestep)
            else:
                raise NotImplementedError

        return dirty_timesteps, removed_timesteps

    @torch.no_grad()
    def _sync_grounded_cache(
        self,
        episode_id,
        dirty_timesteps,
        removed_timesteps,
        device,
        dtype,
    ):
        cache_meta = (device.type, device.index, dtype)
        if self._grounded_cache_meta.get(episode_id) != cache_meta:
            self.grounded_history_memory[episode_id] = OrderedDict()
            self._grounded_cache_meta[episode_id] = cache_meta

        cache = self.grounded_history_memory.setdefault(episode_id, OrderedDict())
        raw_bank = self.history_memory[episode_id]

        for timestep in removed_timesteps:
            cache.pop(timestep, None)

        raw_keys = set(raw_bank.keys())
        for timestep in list(cache.keys()):
            if timestep not in raw_keys:
                del cache[timestep]

        missing_timesteps = {
            timestep for timestep in raw_bank.keys() if timestep not in cache
        }
        rebuild = [
            timestep
            for timestep in raw_bank.keys()
            if timestep in dirty_timesteps or timestep in missing_timesteps
        ]

        if rebuild:
            raw_feats = torch.stack(
                [
                    raw_bank[timestep][2].to(device=device, dtype=dtype)
                    for timestep in rebuild
                ],
                dim=0,
            )
            if (
                self.frame_max_tokens_num is not None
                and self.frame_max_tokens_num < raw_feats.shape[1]
            ):
                grounded = self._compress_batch_clips_tensor(raw_feats)
            else:
                grounded = raw_feats

            if self.use_time_embed:
                timesteps = torch.stack(
                    [torch.tensor(raw_bank[timestep][0], device=device) for timestep in rebuild]
                )
                grounded = grounded + self.time_embed(timesteps).unsqueeze(1)

            if self.use_pos_embed:
                odometry = torch.stack(
                    [raw_bank[timestep][1].to(device=device) for timestep in rebuild]
                )
                grounded = grounded + self.pos_embed(odometry).unsqueeze(1)

            for timestep, feature in zip(rebuild, grounded):
                cache[timestep] = feature.detach().clone()

        self.grounded_history_memory[episode_id] = OrderedDict(
            (timestep, cache[timestep]) for timestep in raw_bank.keys()
        )

    def _prepare_cached_history(self, episode_ids, device, dtype, channels):
        per_sample_mem = []
        mem_lengths = []
        for episode_id in episode_ids:
            cache = self.grounded_history_memory.get(episode_id, OrderedDict())
            if cache:
                memory = torch.cat(list(cache.values()), dim=0)
            else:
                memory = torch.empty((0, channels), device=device, dtype=dtype)
            per_sample_mem.append(memory)
            mem_lengths.append(memory.shape[0])
        return per_sample_mem, mem_lengths
    
    def forward(self, visual_features: torch.Tensor, episode_ids: List[Any], clip_ids: List[Any], current_odom=None):
        B, N, C = visual_features.shape
        device = visual_features.device
        dtype = visual_features.dtype
        use_grounded_cache = (not self.training) and self.inference_cache_enabled

        # ------------------------------------------------------------------
        # 1. Prepare Latent Queries (Recurrent + Detach)
        # ------------------------------------------------------------------
        queries = self.learnable_latent_query.expand(B, -1, -1).to(device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 2. Store Raw Observation
        # ------------------------------------------------------------------
        seen = set(episode_ids)
        for i, eps in enumerate(episode_ids):
            curr_clip_id = clip_ids[i]
            curr_odom = current_odom[i]
            curr_feat = visual_features[i]
            
            if eps not in self.history_memory:
                self.history_memory[eps] = OrderedDict()

            dirty, removed = self._memory_consolidate(
                eps, curr_feat, curr_clip_id, curr_odom
            )
            if use_grounded_cache:
                self._sync_grounded_cache(
                    eps,
                    dirty_timesteps=dirty,
                    removed_timesteps=removed,
                    device=device,
                    dtype=dtype,
                )
        
        # Cleanup finished episodes
        existing_eps = list(self.history_memory.keys())
        for eps in existing_eps:
            if eps not in seen:
                del self.history_memory[eps]
                self.clear_inference_cache(eps)
        
        # ------------------------------------------------------------------
        # 3. Retrieve, Batch, and Compress History (Unchanged)
        # ------------------------------------------------------------------
        flat_clips = []
        flat_odoms = []
        flat_timesteps = []
        flat_owner = []

        # Grounding Embeddings for CURRENT Step
        current_p_embs = []
        current_t_embs = []

        for i, eps in enumerate(episode_ids):
            if self.use_pos_embed:
                current_p_embs.append(self.pos_embed(current_odom[i].unsqueeze(0)))
            if self.use_time_embed:
                current_t_embs.append(self.time_embed(clip_ids[i].unsqueeze(0)))
            
            if not use_grounded_cache:
                # Reference path: collect and recompute every stored history entry.
                for _, (ts, od, feat) in self.history_memory[eps].items():
                    flat_clips.append(feat.to(device=device, dtype=dtype))
                    flat_odoms.append(od.to(device))
                    flat_timesteps.append(torch.tensor(ts).to(device))
                    flat_owner.append(i)
        
        # Compress ALL history clips in one batch
        compressed_flat = None
        if not use_grounded_cache and len(flat_clips) > 0:
            stacked = torch.stack(flat_clips, dim=0) # (Total_All_Clips, N, C)
            if self.frame_max_tokens_num is not None and self.frame_max_tokens_num < stacked.shape[1]:
                compressed_flat = self._compress_batch_clips_tensor(stacked)
            else:
                compressed_flat = stacked
            
        # ------------------------------------------------------------------
        # 4. Reassemble and Add Embeddings
        # ------------------------------------------------------------------
        per_sample_mem = []
        mem_lengths = []

        if use_grounded_cache:
            per_sample_mem, mem_lengths = self._prepare_cached_history(
                episode_ids, device, dtype, C
            )
        elif compressed_flat is None:
            for i in range(B):
                per_sample_mem.append(torch.empty((0, C), device=device, dtype=dtype))
                mem_lengths.append(0)
        else:
            # Group by Owner
            per_sample_indices = [[] for _ in range(B)]
            for idx, owner in enumerate(flat_owner):
                per_sample_indices[owner].append(idx)

            for i in range(B):
                idxs = per_sample_indices[i]
                if len(idxs) == 0:
                    per_sample_mem.append(torch.empty((0, C), device=device, dtype=dtype))
                    mem_lengths.append(0)
                else:
                    # 1. Extract clips: (T, N_compressed, C)
                    clips_tensor = compressed_flat[idxs]

                    # 2. Add Time Embedding
                    if self.use_time_embed:
                        # Stack scalar tensors -> (T,)
                        ts_tensor = torch.stack([flat_timesteps[j] for j in idxs])
                        t_embs = self.time_embed(ts_tensor).unsqueeze(1)
                        clips_tensor = clips_tensor + t_embs
                    
                    # 3. Add Position Embedding
                    if self.use_pos_embed:
                        # Stack odom vectors -> (T, 3)
                        od_tensor = torch.stack([flat_odoms[j] for j in idxs])
                        p_embs = self.pos_embed(od_tensor).unsqueeze(1)
                        clips_tensor = clips_tensor + p_embs

                    # 4. Flatten to (T*N_compressed, C)
                    mem_tensor = clips_tensor.flatten(0, 1)

                    per_sample_mem.append(mem_tensor)
                    mem_lengths.append(mem_tensor.shape[0])
            
        # ------------------------------------------------------------------
        # 5. Deep Cross Attention & Gating
        # ------------------------------------------------------------------
        S_max = max(mem_lengths) if len(mem_lengths) > 0 else 0

        # Grounding the Static Query
        q_grounded = queries.clone()
        if self.use_pos_embed and len(current_p_embs) > 0:
            q_grounded += torch.cat(current_p_embs, dim=0).unsqueeze(1)
        if self.use_time_embed and len(current_t_embs) > 0:
            q_grounded += torch.cat(current_t_embs, dim=0).unsqueeze(1)
        
        if S_max > 0:
            K = torch.zeros((B, S_max, C), device=device, dtype=dtype)
            V = torch.zeros((B, S_max, C), device=device, dtype=dtype)
            key_padding_mask = torch.ones((B, S_max), dtype=torch.bool, device=device)
            
            for i in range(B):
                mlen = mem_lengths[i]
                if mlen > 0:
                    K[i, :mlen, :] = per_sample_mem[i]
                    V[i, :mlen, :] = per_sample_mem[i]
                    key_padding_mask[i, :mlen] = False
            
            # Deep Retrieval
            curr_query = q_grounded
            need_weights = not (
                (not self.training) and self.skip_memory_attention_weights
            )
            for block in self.retrieval_blocks:
                curr_query = block(
                    curr_query,
                    K,
                    V,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                )
            
            updated_latents = curr_query
            
            # Gate Fusion (q_grounded + retrieved)
            # updated_latents = self.gate_fusion(q_grounded, retrieved_info)

        else:
            updated_latents = q_grounded

        return updated_latents
