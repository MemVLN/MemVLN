import math
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
from torch.distributed import get_rank

from model.qwen.modeling_qwen3_vl import (
    VGGTEmbeddingMerger,
    VGGTEmbeddingMergerCamera,
    VGGTEmbeddingMergerConfig,
)
from model.qwen.factory import (
    QWEN2_5_VL,
    build_model,
    build_processor,
    initialize_memory_tokens,
    resolve_model_family,
)
from model.streamvggt.models.streamvggt import StreamVGGT
from model.utils import disabled_train
from model.memory import ToMeMemoryBank

logger = get_logger(__name__)
torch_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}
BASE_MODEL_INPUT_KEYS = (
    'input_ids',
    'attention_mask',
    'pixel_values',
    'pixel_values_videos',
    'image_grid_thw',
    'video_grid_thw',
    'vggt_input',
)
_QWEN25_STREAMVGGT_UNUSED_HEAD_PREFIXES = (
    "camera_head.",
    "point_head.",
    "depth_head.",
    "track_head.",
)


def validate_qwen25_streamvggt_checkpoint_keys(missing_keys, unexpected_keys):
    """Reject partial Qwen2.5 StreamVGGT backbone loads.

    The project instantiates only StreamVGGT's aggregator. Official checkpoints
    can therefore contain unused prediction heads, but every instantiated key
    must load and no other unexpected namespace is accepted.
    """
    missing_keys = tuple(sorted(str(key) for key in missing_keys))
    unexpected_keys = tuple(sorted(str(key) for key in unexpected_keys))
    invalid_unexpected = tuple(
        key
        for key in unexpected_keys
        if not key.startswith(_QWEN25_STREAMVGGT_UNUSED_HEAD_PREFIXES)
    )
    if missing_keys or invalid_unexpected:
        details = []
        if missing_keys:
            details.append(
                f"missing instantiated keys ({len(missing_keys)}): "
                f"{list(missing_keys[:8])}"
            )
        if invalid_unexpected:
            details.append(
                f"unexpected non-head keys ({len(invalid_unexpected)}): "
                f"{list(invalid_unexpected[:8])}"
            )
        raise RuntimeError(
            "Qwen2.5 StreamVGGT checkpoint is incompatible; " + "; ".join(details)
        )
    return unexpected_keys


def resolve_memory_param_dtype(cfg):
    """Resolve the storage dtype for memory-module parameters and buffers."""
    mem_cfg = cfg.llm.mem_module
    if "param_dtype" in mem_cfg:
        dtype_name = mem_cfg.param_dtype
    else:
        # Historical configs predate the dedicated option and constructed the
        # memory module in the same dtype as the LLM.
        dtype_name = getattr(cfg.llm, "torch_dtype", None)

    if dtype_name is None:
        return None
    if dtype_name not in torch_dtype_map:
        valid = ", ".join(sorted(torch_dtype_map))
        raise ValueError(
            "llm.mem_module.param_dtype must be one of "
            f"[{valid}] or null, got {dtype_name!r}"
        )
    return torch_dtype_map[dtype_name]


class LeoAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_out_len = cfg.llm.max_out_len
        self.model_input_keys = list(BASE_MODEL_INPUT_KEYS)
        self.model_family = resolve_model_family(cfg)
        self.use_vggt = 'vggt' in cfg.vision3d.name.lower()
        self.use_mem = cfg.llm.get("mem_module", None) is not None
        if self.model_family == QWEN2_5_VL:
            self.model_input_keys.append('second_per_grid_ts')

        model_kwargs = {}
        if getattr(cfg.llm, 'torch_dtype', None):
            model_kwargs['dtype'] = torch_dtype_map[cfg.llm.torch_dtype]
        if getattr(cfg.llm, 'attn_implementation', None):
            model_kwargs['attn_implementation'] = cfg.llm.attn_implementation

        self.processor = build_processor(cfg)
        self.model = build_model(
            cfg,
            use_mem=self.use_mem,
            use_vggt=self.use_vggt,
            **model_kwargs,
        )
        if self.use_mem:
            self.build_mem_modules(cfg)
        if self.use_vggt:
            self.build_vggt_modules(cfg)

        if self.use_vggt or self.use_mem:
            self.model_input_keys.extend(["episode_id", "clip_id"])
        if self.use_mem:
            self.model_input_keys.append("current_odom")

        # lora for llm
        if cfg.llm.lora.flag:
            logger.info(f"Apply LoRA with configs: {cfg.llm.lora}")
            target_modules = []
            layers = tuple(cfg.llm.lora.target_modules)
            for name, module in self.model.named_modules():
                if name.endswith(layers) and 'visual' not in name and 'vggt' not in name:
                    target_modules.append(name)
            lora_config = LoraConfig(
                task_type='CAUSAL_LM',
                target_modules=target_modules,
                r=cfg.llm.lora.rank,
                lora_alpha=cfg.llm.lora.alpha,
                lora_dropout=cfg.llm.lora.dropout,
                bias='none',
            )
            self.model = get_peft_model(self.model, peft_config=lora_config)

        # trainable parameters
        for n, p in self.model.named_parameters():
            if 'lora' in n or 'memory' in n:
                p.requires_grad = True
            elif 'visual' in n and 'merger' in n:
                p.requires_grad = getattr(cfg.llm, 'tune_proj', False)
            elif 'vggt_embedding_merger' in n:
                p.requires_grad = True
            elif "embed_tokens" in n and self.use_mem:
                p.requires_grad = True
                p.register_hook(
                    self.make_memory_embedding_grad_hook(self.memory_token_ids)
                )
            elif 'vggt_lambda' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        if not cfg.llm.lora.flag:
            for n, p in self.model.model.language_model.named_parameters():
                p.requires_grad = True

        if not getattr(cfg.llm, 'tune_proj', False):
            self.model.model.visual.eval()
            self.model.model.visual.train = disabled_train

    def build_mem_modules(self, cfg):
        self.model.model.memory = ToMeMemoryBank(
            cfg.llm.mem_module,
            inference_config=getattr(cfg, "inference", None),
        )
        memory_param_dtype = resolve_memory_param_dtype(cfg)
        if memory_param_dtype is not None:
            self.model.model.memory.to(dtype=memory_param_dtype)
            logger.info(
                "Initialized memory-module parameters and buffers with dtype "
                f"{memory_param_dtype}"
            )
        self.memory_token_ids = initialize_memory_tokens(
            self.processor,
            int(cfg.llm.mem_module.memory_token_num),
            getattr(self, "model_family", "qwen3_vl"),
        )
        if getattr(self, "model_family", "qwen3_vl") == QWEN2_5_VL:
            embedding_rows = self.model.get_input_embeddings().num_embeddings
            invalid_ids = [
                token_id
                for token_id in self.memory_token_ids
                if token_id >= embedding_rows
            ]
            if invalid_ids:
                raise ValueError(
                    "Qwen2.5 Memory token IDs exceed the official embedding "
                    f"table ({embedding_rows} rows): {invalid_ids}. The official "
                    "152064-row table must not be resized or shrunk."
                )
        else:
            # Preserve the historical Qwen3 path, whose vendored tokenizer
            # appends the three Memory rows beyond the original table.
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.model.model.memory_token_id = self.memory_token_ids[1]
        self.model.model.memory_merge_vggt_feat = getattr(cfg.llm.mem_module, 'memory_merge_vggt_feat', False)
        self.model.model.visual_obs_mask_ratio_nav = getattr(cfg.llm.mem_module, 'visual_obs_mask_ratio_nav', 0.0)
        self.model.model.visual_obs_mask_ratio_aux = getattr(cfg.llm.mem_module, 'visual_obs_mask_ratio_aux', 1.0)

    def build_vggt_modules(self, cfg):
        # init these modules outside qwen vl model to bypass deepspeed weights loading issues
        if 'stream' not in cfg.vision3d.name.lower():
            raise ValueError(
                f"Only StreamVGGT is supported, got vision3d.name={cfg.vision3d.name!r}"
            )
        if self.model_family == QWEN2_5_VL and (
            str(cfg.vision3d.merger_type).lower() != "camera"
            or str(cfg.vision3d.fusion_op).lower() != "add"
        ):
            raise ValueError(
                "Qwen2.5 Full MemVLN is locked to the existing "
                "VGGTEmbeddingMergerCamera with fusion_op=add."
            )
        self.model.model.vggt = StreamVGGT()
        self.model.model.vggt.stream = True
        self.model.model.vggt.aggregator.max_num_frames = cfg.data.vln.max_num_frames
        self.model.model.vggt.aggregator.cache_feat = cfg.data.vln.vggt_cache_feat
        vggt_ckpt = torch.load(cfg.vision3d.ckpt_path, weights_only=True)
        missing_keys, unexpected_keys = self.model.model.vggt.load_state_dict(vggt_ckpt, strict=False)
        self.model.model.vggt.to(self.model.model.visual.dtype)
        if self.model_family == QWEN2_5_VL:
            unused_head_keys = validate_qwen25_streamvggt_checkpoint_keys(
                missing_keys,
                unexpected_keys,
            )
            logger.info(
                f"Loaded VGGT (stream={self.model.model.vggt.stream}) weights "
                f"from {cfg.vision3d.ckpt_path}; all instantiated keys matched, "
                f"ignored {len(unused_head_keys)} unused official head keys"
            )
        else:
            logger.info(f"Loaded VGGT (stream={self.model.model.vggt.stream}) weights from {cfg.vision3d.ckpt_path}\n"
                        f"Missing keys: {missing_keys}\n"
                        f"Unexpected keys: {unexpected_keys}")
        self.model.model.vggt.eval()
        self.model.model.vggt.train = disabled_train

        vggt_merger_config = VGGTEmbeddingMergerConfig(
            input_dim=2*cfg.vision3d.embed_dim,   # vggt concat tokens after frame and global attention
            output_dim=cfg.llm.hidden_dim,
        )
        self.model.model.merger_type = cfg.vision3d.merger_type
        self.model.model.fusion_op = cfg.vision3d.fusion_op
        if self.model.model.merger_type.lower() == 'patch':
            self.model.model.vggt_embedding_merger = VGGTEmbeddingMerger(
                config=vggt_merger_config
            )
        elif self.model.model.merger_type.lower() == 'camera':
            self.model.model.vggt_embedding_merger = VGGTEmbeddingMergerCamera(
                config=vggt_merger_config
            )
        else:
            raise NotImplementedError(f"Unsupported merger type {self.model.model.merger_type}")

        if getattr(cfg.vision3d, 'fusion_learnable', False):
            # Add a learnable lambda for VGGT feature fusion (Zero-Initialization / ReZero)
            self.model.model.register_parameter('vggt_lambda', nn.Parameter(torch.zeros(1)))

    def reset_episode_state(self, episode_id=None):
        """
        Clear episode-scoped streaming state kept inside memory/VGGT modules.

        Real-world inference reuses a long-lived server process. Clear these
        banks before a new episode starts so clip_id can safely restart from 0.
        """
        candidate_ids = None
        if episode_id is not None:
            candidate_ids = {episode_id, str(episode_id)}

        cleared = {
            "memory_episodes": 0,
            "memory_grounded_episodes": 0,
            "vggt_past_key_values": 0,
            "vggt_past_clip_ids": 0,
            "vggt_past_feats": 0,
        }

        for module in self.modules():
            history_memory = getattr(module, "history_memory", None)
            if isinstance(history_memory, dict):
                if candidate_ids is None:
                    cleared["memory_episodes"] += len(history_memory)
                    history_memory.clear()
                else:
                    for ep_id in candidate_ids:
                        if ep_id in history_memory:
                            del history_memory[ep_id]
                            cleared["memory_episodes"] += 1

            grounded_memory = getattr(module, "grounded_history_memory", None)
            if isinstance(grounded_memory, dict):
                cache_meta = getattr(module, "_grounded_cache_meta", None)
                if candidate_ids is None:
                    cleared["memory_grounded_episodes"] += len(grounded_memory)
                    grounded_memory.clear()
                    if isinstance(cache_meta, dict):
                        cache_meta.clear()
                else:
                    for ep_id in candidate_ids:
                        if ep_id in grounded_memory:
                            del grounded_memory[ep_id]
                            cleared["memory_grounded_episodes"] += 1
                        if isinstance(cache_meta, dict):
                            cache_meta.pop(ep_id, None)

            for attr_name, cleared_key in (
                ("past_key_values", "vggt_past_key_values"),
                ("past_clip_ids", "vggt_past_clip_ids"),
                ("past_feats", "vggt_past_feats"),
            ):
                cache = getattr(module, attr_name, None)
                if isinstance(cache, dict):
                    if candidate_ids is None:
                        cleared[cleared_key] += len(cache)
                        cache.clear()
                    else:
                        for ep_id in candidate_ids:
                            if ep_id in cache:
                                del cache[ep_id]
                                cleared[cleared_key] += 1

        return cleared

    def make_grad_mask_hook(self, frozen_ids):
        def grad_mask_hook(grad):
            new_grad = grad.clone()
            new_grad[frozen_ids] = 0
            return new_grad
        return grad_mask_hook

    def make_memory_embedding_grad_hook(self, memory_token_ids):
        """Keep gradients only for the three dynamically resolved Memory rows."""
        memory_token_ids = tuple(int(token_id) for token_id in memory_token_ids)

        def grad_mask_hook(grad):
            new_grad = torch.zeros_like(grad)
            new_grad[list(memory_token_ids)] = grad[list(memory_token_ids)]
            return new_grad

        return grad_mask_hook

    @cached_property
    def dtype(self):
        return next(self.parameters()).dtype

    @cached_property
    def device(self):
        return next(self.parameters()).device

    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot

    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)

    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        # learnable_params_size = self.count_params(learnable_named_params.values())
        # frozen_params_size = self.count_params(frozen_named_params.values())
        # logger.info(
        #     f"Build LeoAgent with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
        #     f"{self.show_params_size(learnable_params_size)} learnable and " +
        #     f"{self.show_params_size(frozen_params_size)} frozen"
        # )
        # count_params does not work with deepspeed for frozen params
        if hasattr(self.model, 'print_trainable_parameters') and (not torch.distributed.is_initialized() or get_rank() == 0):
            self.model.print_trainable_parameters()

        logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")

        return learnable_named_params

    def compute_ce_loss(self, logits, targets, sample_weight=None):
        shifted_loss_mask = (targets[..., 1:] >= 0).to(logits.device)
        shifted_logits = logits[..., :-1, :].float()
        shifted_labels = targets[..., 1:].clone().to(logits.device)

        if sample_weight is None:
            return F.cross_entropy(
                shifted_logits[shifted_loss_mask].contiguous(),
                shifted_labels[shifted_loss_mask].contiguous(),
                reduction='mean',
            )

        shifted_labels[~shifted_loss_mask] = 0

        per_token_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction='none',
        ).view(shifted_labels.shape)
        token_loss_sum = (per_token_loss * shifted_loss_mask.to(per_token_loss.dtype)).sum(-1)
        token_count = shifted_loss_mask.sum(-1).to(per_token_loss.dtype)
        per_sample_loss = token_loss_sum / token_count.clamp_min(1)

        weight = sample_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
        return (per_sample_loss * weight).sum() / weight.sum().clamp_min(1e-6)

    def sft_forward(self, data_dict):
        inputs = {}
        for k in self.model_input_keys:
            if k in data_dict:
                inputs[k] = data_dict[k]

        # indicator of auxiliary task
        is_aux = []
        for src in data_dict['source']:
            if 'vlntrajsum' in src.lower():
                is_aux.append(True)
            else:
                is_aux.append(False)

        if self.use_mem:
            inputs['is_aux'] = is_aux
        outputs = self.model(**inputs, labels=None)
        logits = outputs.logits

        sample_weight = data_dict.get('sample_weight', None)
        data_dict['loss_nll'] = self.compute_ce_loss(logits, data_dict['labels'], sample_weight=sample_weight)
        if sample_weight is not None:
            data_dict['sample_weight'] = sample_weight.to(logits.device, dtype=torch.float32).mean()
        data_dict['loss'] = 0
        data_dict['loss'] += data_dict['loss_nll']
        # if len(data_dict['output_gt']) == 1:
        #     data_dict[f'loss_{data_dict["output_gt"][0]}'] = data_dict['loss_nll']

        return data_dict

    def forward(self, data_dict):
        return self.sft_forward(data_dict)

    @torch.no_grad()
    def generate(
        self,
        data_dict,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
    ):
        if (
            getattr(self, "model_family", None) == QWEN2_5_VL
            and getattr(self, "use_mem", False)
            and getattr(self, "use_vggt", False)
        ):
            if do_sample or num_beams != 1 or num_return_sequences != 1:
                raise ValueError(
                    "Qwen2.5 Full MemVLN generation supports greedy decoding "
                    "only (do_sample=False, num_beams=1, "
                    "num_return_sequences=1)."
                )
        inputs = {}
        for k in self.model_input_keys:
            if k in data_dict:
                inputs[k] = data_dict[k]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_out_len,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )

        input_seq_len = data_dict['input_ids'].shape[1]
        output_ids = outputs[:, input_seq_len:]
        output_txt = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_txt = [txt.strip() for txt in output_txt]
        data_dict['output_txt'] = output_txt

        return data_dict
