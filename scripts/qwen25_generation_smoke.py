#!/usr/bin/env python3
"""Run a bounded real-weight Qwen2.5 Baseline or Full generation smoke test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from accelerate.state import PartialState
from hydra import compose, initialize_config_dir
from PIL import Image
import transformers


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.reproducibility import count_trainable_parameters, write_json_atomic
from data.data_utils import dict_to_cuda
from model.leo_agent import LeoAgent
from model.qwen.factory import (
    MEMORY_END_TOKEN,
    MEMORY_START_TOKEN,
    MEMORY_TOKEN,
    apply_chat_template,
    configure_processor,
    count_multimodal_tokens,
)
from model.streamvggt.processing import build_online_vggt_input


EXPECTED_TRANSFORMERS_VERSION = "4.57.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load real Qwen2.5 weights and generate two short synthetic "
            "navigation steps. Full also exercises Memory/VGGT episode state "
            "and reset."
        )
    )
    parser.add_argument(
        "--endpoint",
        choices=("baseline", "full"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--streamvggt-checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Optional model.safetensors/pytorch_model.bin or Accelerate "
            "checkpoint directory produced by the one-batch smoke run."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen2.5 reviewer-control smoke tests require transformers=="
            f"{EXPECTED_TRANSFORMERS_VERSION}, found {transformers.__version__}."
        )
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Qwen2.5 model directory not found: {args.model_path}")
    index_path = args.model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Qwen2.5 checkpoint index not found: {index_path}"
        )
    if args.endpoint == "full":
        if args.streamvggt_checkpoint is None:
            raise ValueError("Full smoke requires --streamvggt-checkpoint.")
        if not args.streamvggt_checkpoint.is_file():
            raise FileNotFoundError(
                "StreamVGGT checkpoint not found: "
                f"{args.streamvggt_checkpoint}"
            )
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(f"Smoke checkpoint not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable; cannot use {args.device!r}.")


def compose_config(args: argparse.Namespace):
    os.environ["QWEN25_VL_7B_PATH"] = str(args.model_path.resolve())
    if args.streamvggt_checkpoint is not None:
        os.environ["STREAMVGGT_CKPT"] = str(
            args.streamvggt_checkpoint.resolve()
        )
    config_name = (
        "qwen2_5_baseline" if args.endpoint == "baseline" else "qwen2_5_full"
    )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs"),
    ):
        return compose(
            config_name=config_name,
            overrides=[
                f"llm.max_out_len={args.max_new_tokens}",
                "data.vln.max_num_frames=1",
            ],
        )


def synthetic_image(step: int) -> Image.Image:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = 30 + step * 50
    image[120:360, 220:420, 1] = 180
    image[200:280, 300:340, 2] = 255
    return Image.fromarray(image)


def build_messages(image: Image.Image, *, use_memory: bool):
    content = []
    if use_memory:
        content.append(
            {
                "type": "text",
                "text": MEMORY_START_TOKEN + MEMORY_TOKEN + MEMORY_END_TOKEN,
            }
        )
    content.extend(
        [
            {"type": "text", "text": "Current observation: "},
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "You are a navigation robot. What is your next action? "
                    "Answer with one short navigation command."
                ),
            },
        ]
    )
    return [{"role": "user", "content": content}]


def current_git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / filename
        for filename in ("model.safetensors", "pytorch_model.bin")
        if (path / filename).is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one model checkpoint in {path}, found {candidates}."
        )
    return candidates[0]


def load_smoke_checkpoint(agent: LeoAgent, checkpoint: Path | None) -> dict:
    if checkpoint is None:
        return {
            "status": "base_only_untrained_project_modules",
            "path": None,
        }
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path), device="cpu")
    else:
        state_dict = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location="cpu",
        )
    missing_keys, unexpected_keys = agent.load_state_dict(
        state_dict,
        strict=False,
    )
    if unexpected_keys:
        raise RuntimeError(
            "Smoke checkpoint contains keys unknown to the current endpoint: "
            f"{unexpected_keys}"
        )
    trainable_keys = {
        name for name, parameter in agent.named_parameters()
        if parameter.requires_grad
    }
    missing_trainable_keys = sorted(trainable_keys.intersection(missing_keys))
    if missing_trainable_keys:
        raise RuntimeError(
            "Smoke checkpoint is missing trainable endpoint parameters: "
            f"{missing_trainable_keys}"
        )
    return {
        "status": "loaded",
        "path": str(checkpoint_path.resolve()),
        "saved_key_count": len(state_dict),
        "missing_frozen_key_count": len(missing_keys),
        "missing_trainable_keys": missing_trainable_keys,
        "unexpected_keys": list(unexpected_keys),
    }


def build_agent(cfg, *, cpu: bool) -> LeoAgent:
    """Construct the agent after initializing Accelerate's shared state."""
    PartialState(cpu=cpu)
    return LeoAgent(cfg)


def validate_episode_reset(agent: LeoAgent, reset_counts: dict) -> None:
    """Require Full smoke runs to prove both episode caches were cleared."""
    if agent.use_mem and int(reset_counts.get("memory_episodes", 0)) < 1:
        raise RuntimeError(
            "Full smoke did not clear an episode from the Memory bank."
        )
    if agent.use_vggt and int(reset_counts.get("vggt_past_key_values", 0)) < 1:
        raise RuntimeError(
            "Full smoke did not clear an episode from the StreamVGGT cache."
        )


def run() -> dict:
    args = parse_args()
    validate_args(args)
    cfg = compose_config(args)
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    agent = build_agent(cfg, cpu=device.type == "cpu")
    configure_processor(agent.processor, cfg, use_vggt=agent.use_vggt)
    checkpoint = load_smoke_checkpoint(agent, args.checkpoint)
    parameter_counts = count_trainable_parameters(
        agent,
        memory_token_ids=getattr(agent, "memory_token_ids", ()),
        hidden_size=int(cfg.llm.hidden_dim),
    )
    load_seconds = time.perf_counter() - started

    agent.to(device)
    agent.eval()
    endpoint_steps = []
    episode_id = f"qwen25-{args.endpoint}-generation-smoke"

    for clip_id in range(2):
        messages = build_messages(
            synthetic_image(clip_id),
            use_memory=agent.use_mem,
        )
        inputs = apply_chat_template(
            agent.processor,
            messages,
            cfg,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        token_counts = count_multimodal_tokens(
            inputs["input_ids"], agent.processor
        )
        vggt_shape = None
        if agent.use_vggt:
            inputs["vggt_input"] = build_online_vggt_input(
                agent.processor,
                messages,
                inputs["image_grid_thw"],
                dense_history=False,
            )
            vggt_shape = list(inputs["vggt_input"][0].shape)
        if agent.use_mem or agent.use_vggt:
            inputs["episode_id"] = [episode_id]
            inputs["clip_id"] = torch.tensor([clip_id])
        if agent.use_mem:
            inputs["current_odom"] = torch.tensor(
                [[0.25 * clip_id, 0.0, 0.0]], dtype=torch.float32
            )

        device_inputs = dict_to_cuda(dict(inputs), device)
        step_started = time.perf_counter()
        output = agent.generate(device_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        endpoint_steps.append(
            {
                "clip_id": clip_id,
                "generated_text": output["output_txt"][0],
                "seconds": time.perf_counter() - step_started,
                "visual_token_count": int(
                    token_counts["visual_token_count"][0].item()
                ),
                "memory_token_count": int(
                    token_counts["memory_token_count"][0].item()
                ),
                "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "vggt_input_shape": vggt_shape,
            }
        )

    reset_counts = agent.reset_episode_state(episode_id)
    validate_episode_reset(agent, reset_counts)
    report = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "endpoint": args.endpoint,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_path": str(args.model_path.resolve()),
        "streamvggt_checkpoint": (
            str(args.streamvggt_checkpoint.resolve())
            if args.streamvggt_checkpoint is not None
            else None
        ),
        "lora": {
            "rank": int(cfg.llm.lora.rank),
            "alpha": int(cfg.llm.lora.alpha),
            "target_modules": list(cfg.llm.lora.target_modules),
        },
        "load_seconds": load_seconds,
        "parameter_counts": parameter_counts,
        "checkpoint": checkpoint,
        "steps": endpoint_steps,
        "episode_reset_counts": reset_counts,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    if args.report is not None:
        write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
