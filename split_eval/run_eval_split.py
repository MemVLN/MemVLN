import os

import sys
import hydra
import torch
import argparse

from accelerate.utils import set_seed
from accelerate import PartialState
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up one level to 'LEO-R'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluator.vlnce_eval_split import VLNCESplitEvaluator
from model.leo_agent import LeoAgent
from common.misc import rgetattr
from split_eval.checkpoint_utils import require_checkpoint_file


def main():
    parser = argparse.ArgumentParser(description="Run Split Evaluation")
    parser.add_argument("--split-id", type=int, default=0, help="Split ID for this process")
    parser.add_argument("--split-num", type=int, default=1, help="Total number of splits")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch tag for saving")
    parser.add_argument("--config-name", type=str, default="default", help="Hydra config name")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")

    args, hydra_overrides = parser.parse_known_args()

    print(f"[Run Eval Split] ID={args.split_id}/{args.split_num}, Epoch={args.epoch}")
    print(f"Loading Config: {args.config_name}")
    print(f"Overrides: {hydra_overrides}")

    PartialState()

    config_dir = os.path.join(project_root, "configs")

    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name=args.config_name, overrides=hydra_overrides)

    checkpoint_path = require_checkpoint_file(args.model_path)
    cfg.pretrained_ckpt_path = str(checkpoint_path)

    naming_keys = [cfg.name]
    for name in cfg.naming_keywords:
        key = str(rgetattr(cfg, name))
        if key:
            naming_keys.append(key)
    exp_name = '_'.join(naming_keys)

    # Record the experiment
    cfg.exp_dir = os.path.join(
        cfg.base_dir, exp_name,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" if 'time' in cfg.naming_keywords else ""
    )
    print(f"Experiment Directory: {cfg.exp_dir}")
    
    set_seed(cfg.rng_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    
    print("Building model...")
    model = LeoAgent(cfg)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as safe_load
        model_state_dict = safe_load(str(checkpoint_path), device="cpu")
    else:
        model_state_dict = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location='cpu',
        )
    missing_keys, unexpected_keys = model.load_state_dict(
        model_state_dict,
        strict=False,
    )
    print(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")
    
    model.to(device)
    model.eval()

    dataset_list = []
    
    if "vlnce" in cfg.task:
        dataset_list = list(cfg.task.vlnce.keys())
        print(f"Found VLNCE datasets in config: {dataset_list}")
    
    if not dataset_list:
        print("Warning: 'vlnce' key not found in cfg.task, trying to search")
        for task_key in cfg.task.keys():
            if task_key in ['training', 'task_metrics_weights', 'leomix']:
                continue
            ds_keys = list(cfg.task[task_key].keys())
            if ds_keys:
                dataset_list.extend(ds_keys)

    if not dataset_list:
        raise ValueError("Could not find any valid dataset name in cfg.task! Check your config!")

    for dataset in dataset_list:
        print(f"Initializing VLNCESplitEvaluator for dataset: {dataset}...")
        evaluator = VLNCESplitEvaluator(cfg, dataset, model=model)
        evaluator.eval_action(split_id=args.split_id, total_splits=args.split_num, epoch=args.epoch)
    
    print(f"\nAll datasets processed for Split {args.split_id}.")

if __name__ == "__main__":
    main()
