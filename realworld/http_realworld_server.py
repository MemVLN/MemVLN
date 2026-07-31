import os
import sys
import time
import json
import torch
import torch.nn as nn
import argparse
import threading
import uuid

from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.misc import rgetattr, CustomAccelerator
from model import LeoAgent
from realworld.agent import VLNEvaluator


model_parallel_classes = (
    nn.parallel.DistributedDataParallel,
    nn.DataParallel,
)

logger = get_logger(__name__)

app = Flask(__name__)
idx = 0
terminate = False
start_time = time.time()
output_dir = ''
cfg = None
evaluator = None
active_episode_id = None
evaluation_lock = threading.Lock()
idx2actions = ["stop", "move forward", "turn left", "turn right"]


def load_ckpt(model, ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load
        ckpt = safe_load(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if isinstance(model, model_parallel_classes):
        missing_keys, unexpected_keys = model.module.load_state_dict(ckpt, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    
    print(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")


def parse_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--config", default="../configs/default.yaml", type=str, help="Path to the config file.")
    parser.add_argument("--mixed_precision", default="bf16", type=str, help="Mixed precision training, options (no | fp16 | bf16)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    return parser.parse_args()


def annotate_image(idx, image, start_time, llm_output, output_dir):
    # image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Running  Time: {time.time() - start_time:.2f} s")
    text_content.append(f"Actions      : {llm_output}")
    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height
    
    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

    text_color = 'white'
    y_position = box_y + padding

    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height
    
    image.save(f'{output_dir}/rgb_{idx}_annotation.png')


@app.route('/eval_vln', methods=['POST'])
def eval_vln():
    global idx, terminate, start_time, output_dir, evaluator, active_episode_id

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    # image = np.asarray(image)

    policy_init = data.get('reset', False)
    instruction = data.get('instruction', '')

    with evaluation_lock:
        if evaluator is None:
            return jsonify({"error": "VLN evaluator is not initialized."}), 503

        if policy_init:
            if not isinstance(instruction, str) or not instruction.strip():
                return jsonify({
                    "error": "Instruction cannot be empty when reset is true."
                }), 400
            start_time = time.time()
            terminate = False
            idx = 0
            new_episode_id = uuid.uuid4().hex
            print(
                "Init reset model! "
                f"Episode {new_episode_id}, instruction: '{instruction}'"
            )
            evaluator.reset_memory()
            evaluator.reset_instruction(instruction)
            active_episode_id = new_episode_id
        elif active_episode_id is None:
            return jsonify({
                "error": "No active episode. Send reset=true with an instruction first."
            }), 409

        idx += 1
        if terminate:
            print("!!!!!!!!!!Task Finished!!!!!!!!!!")
            return jsonify({
                "action": [0],
                "action_sequence": [0],
                "atomic_action": [0],
                "is_inference": False,
                "pending_count": 0,
                "raw_model_output": "terminate",
            })

        t1 = time.time()
        action, pending_count_after, is_inference, raw_output, action_sequence = (
            evaluator.step_with_metadata(image, active_episode_id)
        )
        print(f"one evaluate step time: {time.time() - t1:.2f} s")

        atomic_action = action if isinstance(action, list) else [action]
        action_sequence = (
            action_sequence if isinstance(action_sequence, list) else atomic_action
        )

        if 0 in atomic_action:
            terminate = True
        readable_actions = [
            idx2actions[a] if 0 <= a < len(idx2actions) else "unknown"
            for a in action_sequence
        ]
        str_action = ", ".join(readable_actions)

        annotate_image(idx, image, start_time, str_action, output_dir)

        print(str_action)

        return jsonify({
            "action": atomic_action,
            "action_sequence": action_sequence,
            "atomic_action": atomic_action,
            "is_inference": is_inference,
            "pending_count": pending_count_after,
            "raw_model_output": str(raw_output),
        })


if __name__ == "__main__":
    args = parse_args()
    opts = args.opts
    full_cfg_path = Path(args.config)
    cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
    
    with initialize(config_path=cfg_path, version_base=None):
        cfg = compose(config_name=cfg_file, overrides=opts)
    naming_keys = [cfg.name]
    for name in cfg.naming_keywords:
        key = str(rgetattr(cfg, name))
        if key:
            naming_keys.append(key)
    exp_name = '_'.join(naming_keys)

    # Record the experiment
    cfg.exp_dir = os.path.join(
        cfg.base_dir,
        exp_name,
        f"{datetime.now().strftime('%m-%d-%H%M%S')}" if "time" in cfg.naming_keywords else ""
    )
    os.makedirs(cfg.exp_dir, exist_ok=True)
    output_dir = cfg.exp_dir
    OmegaConf.save(cfg, os.path.join(cfg.exp_dir, "config.yaml"), resolve=True)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = CustomAccelerator(
        project_config=ProjectConfiguration(
            project_dir=cfg.exp_dir,
            automatic_checkpoint_naming=True,
            total_limit=1
        ),
        log_with=cfg.logger.name,
        kwargs_handlers=[kwargs]
    )

    # Prepare model
    model = LeoAgent(cfg)

    # Load checkpoint
    assert cfg.pretrained_ckpt_path != "", "Please provide a pretrained checkpoint path!"
    assert os.path.exists(cfg.pretrained_ckpt_path), f"Checkpoint path {cfg.pretrained_ckpt_path} does not exist!"
    print(f"Loading pretrained checkpoint from {cfg.pretrained_ckpt_path}")
    load_ckpt(model, cfg.pretrained_ckpt_path)
    model.requires_grad_(False)
    model.to('cuda')
    model.eval()

    evaluator = VLNEvaluator(model=model, cfg=cfg)

    # while 1:
    #     evaluator.step(np.zeros((480, 640, 3), dtype=np.uint8), 0)

    app.run(host='0.0.0.0', port=5801)
