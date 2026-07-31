import copy as cp
import math
import os
import re
from datetime import datetime
from pathlib import Path

import torch.optim as optim
import wandb
from fvcore.common.registry import Registry
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR


TRAINER_REGISTRY = Registry('Trainer')
_CHECKPOINT_DIRECTORY_PATTERN = re.compile(r"^checkpoint_(\d+)$")


class Tracker():
    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1
        self.loader_step = 0

    def step_loader(self):
        self.loader_step += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.note}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        self.run_id = wandb.util.generate_id()
        self.epoch = 0
        self.loader_step = 0
        self.val_count = 0
        self.overall_best_result = 0

    def state_dict(self):
        return {
            'run_id': self.run_id, 'exp_name': self.exp_name,
            'epoch': self.epoch, 'loader_step': self.loader_step,
            'val_count': self.val_count, 'overall_best_result': self.overall_best_result,
        }

    def load_state_dict(self, state_dict):
        state_dict = cp.deepcopy(state_dict)
        self.exp_name = state_dict['exp_name']
        self.run_id = state_dict['run_id']
        self.epoch = state_dict['epoch']
        self.loader_step = state_dict['loader_step']
        self.val_count = state_dict['val_count']
        self.overall_best_result = state_dict['overall_best_result']


def linear_warmup_cosine_decay(step, warmup_step, total_step):
    if step <= warmup_step:
        return 1e-3 + step / warmup_step * (1 - 1e-3)
    return max(0.5 * (1 + math.cos((step - warmup_step) / (total_step - warmup_step) * math.pi)), 1e-5)


def get_scheduler(cfg, optimizer, total_steps):
    lambda_func = lambda step: globals()[cfg.training.schedule.name](step, cfg.training.schedule.args.warmup_steps, total_steps)
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda_func)


def build_optim(cfg, params, total_steps):
    cfg_dict = OmegaConf.to_container(cfg.training.optim.args, resolve=True)
    optimizer = getattr(optim, cfg.training.optim.name)(params, **cfg_dict)
    scheduler = get_scheduler(cfg, optimizer, total_steps)
    return optimizer, scheduler


def checkpoint_index(path):
    """Return the numeric suffix from an Accelerate checkpoint directory."""
    checkpoint_name = Path(path).name
    match = _CHECKPOINT_DIRECTORY_PATTERN.fullmatch(checkpoint_name)
    if match is None:
        raise ValueError(
            "Expected an Accelerate checkpoint directory named "
            f"'checkpoint_<number>', got {checkpoint_name!r}"
        )
    return int(match.group(1))


def latest_checkpoint(path):
    """Return the checkpoint directory with the largest numeric suffix."""
    if not os.path.isdir(path):
        return ""

    checkpoints = []
    for name in os.listdir(path):
        candidate = os.path.join(path, name)
        if not os.path.isdir(candidate):
            continue
        try:
            index = checkpoint_index(candidate)
        except ValueError:
            continue
        checkpoints.append((index, candidate))

    if not checkpoints:
        return ""
    return max(checkpoints, key=lambda item: item[0])[1]


def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
