"""Factories and small compatibility helpers for supported Qwen-VL families.

Qwen2.5-VL deliberately uses the implementation shipped by Transformers.  The
helpers in this module only select project variants and adapt the surrounding
prompt/data plumbing; they do not copy or replace upstream model/processor
code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MethodType
from typing import Any

import torch


QWEN3_VL = "qwen3_vl"
QWEN2_5_VL = "qwen2_5_vl"
SUPPORTED_MODEL_FAMILIES = (QWEN3_VL, QWEN2_5_VL)

MEMORY_START_TOKEN = "<|memory_start|>"
MEMORY_TOKEN = "<|memory_pad|>"
MEMORY_END_TOKEN = "<|memory_end|>"
MEMORY_SPECIAL_TOKENS = (
    MEMORY_START_TOKEN,
    MEMORY_TOKEN,
    MEMORY_END_TOKEN,
)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _llm_config(cfg: Any) -> Any:
    """Return the LLM sub-config from either a root config or an LLM config."""
    llm_cfg = _get_field(cfg, "llm")
    return cfg if llm_cfg is None else llm_cfg


def resolve_model_family(cfg_or_family: Any) -> str:
    """Resolve and validate an explicit Qwen model family.

    Args:
        cfg_or_family: A family string, root config, or LLM config containing
            ``model_family``.
    """
    if isinstance(cfg_or_family, str):
        family = cfg_or_family
    else:
        family = _get_field(_llm_config(cfg_or_family), "model_family")
    if family is None:
        raise ValueError(
            "Missing required llm.model_family; expected one of "
            f"{SUPPORTED_MODEL_FAMILIES}."
        )
    family = str(family).strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported llm.model_family={family!r}; expected one of "
            f"{SUPPORTED_MODEL_FAMILIES}."
        )
    return family


def _infer_use_memory(cfg: Any) -> bool:
    return _get_field(_llm_config(cfg), "mem_module") is not None


def _infer_use_vggt(cfg: Any) -> bool:
    vision_cfg = _get_field(cfg, "vision3d")
    if vision_cfg is None:
        return False
    return "vggt" in str(_get_field(vision_cfg, "name", "")).lower()


def get_model_class(
    model_family: Any,
    *,
    use_mem: bool = False,
    use_vggt: bool = False,
):
    """Return the model class for a supported family/project variant.

    Qwen2.5 supports the two reviewer-control endpoints only: the official
    plain model and the combined Memory+StreamVGGT model.  Importing the Full
    project extension is deferred until that endpoint is selected.
    """
    family = resolve_model_family(model_family)
    if family == QWEN3_VL:
        if use_mem and use_vggt:
            from model.qwen.modeling_qwen3_vl_mem_vggt import (
                Qwen3VLForConditionalGeneration_Mem_VGGT,
            )

            return Qwen3VLForConditionalGeneration_Mem_VGGT
        if use_mem:
            from model.qwen.modeling_qwen3_vl_mem import (
                Qwen3VLForConditionalGeneration_Mem,
            )

            return Qwen3VLForConditionalGeneration_Mem
        if use_vggt:
            from model.qwen.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration_VGGT,
            )

            return Qwen3VLForConditionalGeneration_VGGT

        from model.qwen.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    if not use_mem and not use_vggt:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    if use_mem and use_vggt:
        from model.qwen.modeling_qwen2_5_vl_mem_vggt import (
            Qwen2_5_VLForConditionalGeneration_Mem_VGGT,
        )

        return Qwen2_5_VLForConditionalGeneration_Mem_VGGT
    raise ValueError(
        "Qwen2.5-VL exposes only the plain baseline and the combined "
        "Memory+StreamVGGT endpoint; use_mem and use_vggt must be both false "
        "or both true."
    )


def build_model(
    cfg: Any,
    *,
    use_mem: bool | None = None,
    use_vggt: bool | None = None,
    **from_pretrained_kwargs: Any,
):
    """Load a selected Qwen model from ``cfg.llm.cfg_path``."""
    llm_cfg = _llm_config(cfg)
    cfg_path = _get_field(llm_cfg, "cfg_path")
    if not cfg_path:
        raise ValueError("Missing required llm.cfg_path.")
    if use_mem is None:
        use_mem = _infer_use_memory(cfg)
    if use_vggt is None:
        use_vggt = _infer_use_vggt(cfg)
    revision = _get_field(llm_cfg, "revision")
    if revision:
        from_pretrained_kwargs.setdefault("revision", str(revision))
    model_cls = get_model_class(
        resolve_model_family(cfg),
        use_mem=bool(use_mem),
        use_vggt=bool(use_vggt),
    )
    return model_cls.from_pretrained(cfg_path, **from_pretrained_kwargs)


def build_processor(cfg: Any, **from_pretrained_kwargs: Any):
    """Load the native processor selected by ``llm.model_family``.

    Dataset-specific visual sizes and StreamVGGT glue are intentionally handled
    by :func:`configure_processor` after construction.
    """
    llm_cfg = _llm_config(cfg)
    cfg_path = _get_field(llm_cfg, "cfg_path")
    if not cfg_path:
        raise ValueError("Missing required llm.cfg_path.")

    from_pretrained_kwargs.setdefault(
        "padding_side", _get_field(llm_cfg, "padding_side", "left")
    )
    revision = _get_field(llm_cfg, "revision")
    if revision:
        from_pretrained_kwargs.setdefault("revision", str(revision))
    family = resolve_model_family(cfg)
    if family == QWEN3_VL:
        from model.qwen.processing_qwen3_vl import Qwen3VLProcessor

        return Qwen3VLProcessor.from_pretrained(
            cfg_path, **from_pretrained_kwargs
        )

    requested_fast = from_pretrained_kwargs.pop("use_fast", False)
    if requested_fast is not False:
        raise ValueError(
            "Qwen2.5-VL must use the official slow processor "
            "(use_fast=False) for reproducible preprocessing."
        )
    from transformers import Qwen2_5_VLProcessor

    return Qwen2_5_VLProcessor.from_pretrained(
        cfg_path,
        use_fast=False,
        **from_pretrained_kwargs,
    )


def _plain_size(size_cfg: Any, field_name: str) -> dict[str, Any]:
    if isinstance(size_cfg, Mapping):
        return {str(key): value for key, value in size_cfg.items()}
    raise TypeError(
        f"{field_name} must be a mapping, got {type(size_cfg).__name__}."
    )


def configure_processor(processor: Any, cfg: Any, use_vggt: bool) -> Any:
    """Apply dataset visual budgets and optional StreamVGGT glue in-place.

    Qwen3 retains its existing custom preprocessing path.  Qwen2.5 retains all
    official image/video preprocessing; when StreamVGGT is active, only the
    independent ``vggt_preprocess`` helper is attached to its video processor.
    """
    data_cfg = _get_field(cfg, "data")
    visual_size = _get_field(data_cfg, "visual_size")
    image_size = _get_field(visual_size, "image")
    video_size = _get_field(visual_size, "video")
    if image_size is None or video_size is None:
        raise ValueError(
            "Missing required data.visual_size.image or data.visual_size.video."
        )
    if not hasattr(processor, "image_processor") or not hasattr(
        processor, "video_processor"
    ):
        raise TypeError("Qwen processor must expose image_processor and video_processor.")

    processor.image_processor.size = _plain_size(
        image_size, "data.visual_size.image"
    )
    processor.video_processor.size = _plain_size(
        video_size, "data.visual_size.video"
    )
    processor.video_processor.do_sample_frames = False

    if use_vggt:
        vision_cfg = _get_field(cfg, "vision3d")
        fixed_grid_hw = _get_field(vision_cfg, "input_grid_hw")
        if fixed_grid_hw is not None:
            try:
                fixed_grid_hw = tuple(int(value) for value in fixed_grid_hw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "vision3d.input_grid_hw must contain two positive integers."
                ) from exc
            if len(fixed_grid_hw) != 2 or any(
                value <= 0 for value in fixed_grid_hw
            ):
                raise ValueError(
                    "vision3d.input_grid_hw must contain two positive integers, "
                    f"got {fixed_grid_hw}."
                )
            processor.vggt_input_grid_hw = fixed_grid_hw

    family = resolve_model_family(cfg)
    if family == QWEN3_VL:
        fix_video_preprocess = getattr(processor, "fix_video_preprocess", None)
        if not callable(fix_video_preprocess):
            raise TypeError(
                "The Qwen3 processor is missing its required "
                "fix_video_preprocess method."
            )
        fix_video_preprocess(use_vggt=bool(use_vggt))
    elif use_vggt:
        from model.qwen.processing_qwen3_vl import vggt_preprocess

        processor.video_processor.vggt_preprocess = MethodType(
            vggt_preprocess, processor.video_processor
        )
    return processor


def initialize_memory_tokens(
    processor: Any,
    memory_token_num: int,
    model_family: Any,
) -> tuple[int, int, int]:
    """Initialize the three project Memory tokens without resizing a model."""
    if (
        isinstance(memory_token_num, bool)
        or not isinstance(memory_token_num, int)
        or memory_token_num <= 0
    ):
        raise ValueError(
            f"memory_token_num must be a positive integer, got {memory_token_num!r}."
        )

    family = resolve_model_family(model_family)
    if family == QWEN3_VL:
        init_memory = getattr(processor, "__init_memory__", None)
        if not callable(init_memory):
            raise TypeError("The Qwen3 processor is missing __init_memory__.")
        init_memory(memory_token_num)
    else:
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(MEMORY_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
        token_ids = tuple(
            tokenizer.convert_tokens_to_ids(token)
            for token in MEMORY_SPECIAL_TOKENS
        )
        if any(token_id is None or int(token_id) < 0 for token_id in token_ids):
            raise ValueError(
                "Failed to register all Qwen2.5 Memory special tokens: "
                f"{dict(zip(MEMORY_SPECIAL_TOKENS, token_ids))}."
            )
        tokenizer.memory_token = MEMORY_TOKEN
        tokenizer.memory_token_id = int(token_ids[1])
        processor.memory_token = MEMORY_TOKEN
        processor.memory_token_id = int(token_ids[1])
        processor.memory_token_num = memory_token_num

    convert_tokens_to_ids = getattr(
        processor.tokenizer, "convert_tokens_to_ids", None
    )
    if callable(convert_tokens_to_ids):
        token_ids = tuple(
            int(convert_tokens_to_ids(token))
            for token in MEMORY_SPECIAL_TOKENS
        )
    elif family == QWEN3_VL and getattr(
        processor.tokenizer, "memory_token_id", None
    ) is not None:
        # Compatibility for lightweight processor doubles and legacy Qwen3
        # wrappers that expose only the Memory pad ID after initialization.
        memory_token_id = int(processor.tokenizer.memory_token_id)
        token_ids = (
            memory_token_id - 1,
            memory_token_id,
            memory_token_id + 1,
        )
    else:
        raise TypeError(
            "Tokenizer must expose convert_tokens_to_ids for Memory tokens."
        )
    if len(set(token_ids)) != len(MEMORY_SPECIAL_TOKENS):
        raise ValueError(f"Memory special-token IDs must be distinct, got {token_ids}.")
    return token_ids


_MEMORY_TOKEN_RUN = re.compile(r"(?:" + re.escape(MEMORY_TOKEN) + r")+")


def _expand_memory_text(text: str, memory_token_num: int) -> str:
    def replace_run(match: re.Match[str]) -> str:
        run = match.group(0)
        count = run.count(MEMORY_TOKEN)
        if count == memory_token_num:
            return run
        if count != 1:
            raise ValueError(
                "Memory marker runs must contain either one placeholder or "
                f"the configured {memory_token_num} placeholders; got {count}."
            )
        return MEMORY_TOKEN * memory_token_num

    return _MEMORY_TOKEN_RUN.sub(replace_run, text)


def _copy_and_expand(value: Any, memory_token_num: int) -> Any:
    if isinstance(value, str):
        return _expand_memory_text(value, memory_token_num)
    if isinstance(value, list):
        return [_copy_and_expand(item, memory_token_num) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_and_expand(item, memory_token_num) for item in value)
    if isinstance(value, Mapping):
        return {
            key: _copy_and_expand(item, memory_token_num)
            for key, item in value.items()
        }
    # Reuse image/tensor/path-like leaves.  Only the surrounding message
    # containers and text strings need to be copied.
    return value


def expand_memory_markers(messages: Any, memory_token_num: int) -> Any:
    """Return a non-mutating, idempotent expansion of Memory placeholders."""
    if (
        isinstance(memory_token_num, bool)
        or not isinstance(memory_token_num, int)
        or memory_token_num <= 0
    ):
        raise ValueError(
            f"memory_token_num must be a positive integer, got {memory_token_num!r}."
        )
    return _copy_and_expand(messages, memory_token_num)


def apply_chat_template(
    processor: Any,
    messages: Any,
    model_family: Any,
    **kwargs: Any,
) -> Any:
    """Apply the family-native chat template with minimal project glue."""
    family = resolve_model_family(model_family)
    if family == QWEN3_VL:
        return processor.apply_chat_template(messages, **kwargs)

    memory_token_num = getattr(processor, "memory_token_num", None)
    if memory_token_num is not None:
        messages = expand_memory_markers(messages, int(memory_token_num))

    if "fps" in kwargs and float(kwargs["fps"]) != 2.0:
        raise ValueError("Qwen2.5-VL experiments require fps=2.0.")
    if "do_sample_frames" in kwargs and kwargs["do_sample_frames"] is not False:
        raise ValueError(
            "Qwen2.5-VL experiments require do_sample_frames=False."
        )
    kwargs["fps"] = 2.0
    kwargs["do_sample_frames"] = False
    return processor.apply_chat_template(messages, **kwargs)


def mask_labels_for_last_assistant(
    input_ids: torch.Tensor,
    tokenizer: Any,
    *,
    attention_mask: torch.Tensor | None = None,
    pad_token_id: int | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build labels supervised only after the last assistant header.

    The header token IDs are obtained from the active tokenizer instead of
    relying on Qwen-version-specific numeric IDs.
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim not in (1, 2):
        raise TypeError("input_ids must be a rank-1 or rank-2 torch.Tensor.")
    squeeze = input_ids.ndim == 1
    batch_input_ids = input_ids.unsqueeze(0) if squeeze else input_ids
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor when provided.")
        batch_attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        )
        if batch_attention_mask.shape != batch_input_ids.shape:
            raise ValueError(
                "attention_mask and input_ids must have identical shapes, got "
                f"{tuple(batch_attention_mask.shape)} and "
                f"{tuple(batch_input_ids.shape)}."
            )
    else:
        batch_attention_mask = None

    header_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    if not header_ids:
        raise ValueError("Tokenizer produced an empty assistant header.")
    header = torch.tensor(
        header_ids, dtype=batch_input_ids.dtype, device=batch_input_ids.device
    )
    if batch_input_ids.shape[1] < header.numel():
        raise ValueError("Input sequence is shorter than the assistant header.")

    labels = batch_input_ids.clone()
    header_size = int(header.numel())
    for sample_index, row in enumerate(batch_input_ids):
        matches = row.unfold(0, header_size, 1).eq(header).all(dim=1)
        if batch_attention_mask is not None:
            active_windows = (
                batch_attention_mask[sample_index]
                .bool()
                .unfold(0, header_size, 1)
                .all(dim=1)
            )
            matches &= active_windows
        starts = matches.nonzero(as_tuple=False).flatten()
        if starts.numel() == 0:
            raise ValueError(
                "Could not find an assistant header in tokenized sample "
                f"{sample_index}."
            )
        output_start = int(starts[-1].item()) + header_size
        labels[sample_index, :output_start] = ignore_index

    if batch_attention_mask is not None:
        labels.masked_fill_(~batch_attention_mask.bool(), ignore_index)
    else:
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels.masked_fill_(batch_input_ids == int(pad_token_id), ignore_index)
    return labels.squeeze(0) if squeeze else labels


def count_multimodal_tokens(
    input_ids: torch.Tensor,
    processor: Any,
) -> dict[str, torch.Tensor]:
    """Count visual and Memory placeholders for each tokenized sample."""
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim not in (1, 2):
        raise TypeError("input_ids must be a rank-1 or rank-2 torch.Tensor.")
    squeeze = input_ids.ndim == 1
    rows = input_ids.unsqueeze(0) if squeeze else input_ids
    tokenizer = processor.tokenizer

    def resolve_token_id(attribute: str, token: str) -> int | None:
        token_id = getattr(processor, attribute, None)
        if token_id is None:
            token_id = getattr(tokenizer, attribute, None)
        if token_id is None:
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                token_id = convert(token)
        return None if token_id is None else int(token_id)

    image_token_id = resolve_token_id("image_token_id", "<|image_pad|>")
    video_token_id = resolve_token_id("video_token_id", "<|video_pad|>")
    memory_token_id = resolve_token_id("memory_token_id", MEMORY_TOKEN)

    visual_mask = torch.zeros_like(rows, dtype=torch.bool)
    if image_token_id is not None:
        visual_mask |= rows == image_token_id
    if video_token_id is not None:
        visual_mask |= rows == video_token_id
    visual_counts = visual_mask.sum(dim=-1)
    if memory_token_id is None:
        memory_counts = torch.zeros_like(visual_counts)
    else:
        memory_counts = (rows == memory_token_id).sum(dim=-1)

    if squeeze:
        visual_counts = visual_counts.squeeze(0)
        memory_counts = memory_counts.squeeze(0)
    return {
        "visual_token_count": visual_counts,
        "memory_token_count": memory_counts,
    }


__all__ = [
    "QWEN3_VL",
    "QWEN2_5_VL",
    "SUPPORTED_MODEL_FAMILIES",
    "MEMORY_SPECIAL_TOKENS",
    "resolve_model_family",
    "get_model_class",
    "build_model",
    "build_processor",
    "configure_processor",
    "initialize_memory_tokens",
    "expand_memory_markers",
    "apply_chat_template",
    "mask_labels_for_last_assistant",
    "count_multimodal_tokens",
]
