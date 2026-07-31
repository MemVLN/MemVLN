import os
from contextlib import contextmanager
from math import ceil
from pathlib import Path

import deepspeed
import torch
import torch.nn as nn
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from omegaconf import OmegaConf
from tqdm import trange

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler])

from common.misc import CustomAccelerator
from data.build import build_dataloader_leo
from evaluator.build import build_eval_leo
from model.leo_agent import LeoAgent
from trainer.build import (
    TRAINER_REGISTRY,
    Tracker,
    build_optim,
    checkpoint_index,
    latest_checkpoint,
)
from trl.models.utils import unwrap_model_for_generation

logger = get_logger(__name__)

gather_keys = [
    'source', 'scene_id', 'input_txt', 'output_gt', 'output_txt', 'corpus_key',   # non-tensor
    'iou_flag', 'type',   # tensor
]

_DEPRECATED_CHECKPOINT_KEY_COMPONENTS = (
    "memory_" + "repa_adapter",
)
_MODEL_CHECKPOINT_FILENAMES = ("model.safetensors", "pytorch_model.bin")


class _UnpreparedDataLoaderState:
    """Minimal state Accelerate reads when synchronizing at loader exhaustion."""

    def __init__(self, end_of_dataloader):
        self.end_of_dataloader = end_of_dataloader
        self.remainder = 0


@contextmanager
def track_unprepared_dataloader_batch(
    accelerator,
    *,
    enabled,
    end_of_dataloader,
):
    """Expose the final custom-sampler batch to Accelerate accumulation.

    The custom streaming sampler is already sharded across ranks, so its
    DataLoader intentionally bypasses ``accelerator.prepare``. Registering this
    lightweight state only while entering ``accelerator.accumulate`` preserves
    Accelerate's end-of-loader gradient synchronization without re-sharding the
    sampler.
    """
    gradient_state = accelerator.gradient_state
    if not enabled or not gradient_state.sync_with_dataloader:
        yield
        return

    loader_state = _UnpreparedDataLoaderState(end_of_dataloader)
    gradient_state._add_dataloader(loader_state)
    try:
        yield
    finally:
        gradient_state._remove_dataloader(loader_state)


def resolve_checkpoint_total_limit(training_cfg):
    """Validate the number of automatic checkpoints Accelerate may retain."""
    total_limit = training_cfg.get("checkpoint_total_limit", 1)
    if total_limit is None:
        return None
    if isinstance(total_limit, bool) or not isinstance(total_limit, int):
        raise TypeError(
            "training.checkpoint_total_limit must be a positive integer or "
            f"null, got {total_limit!r}"
        )
    if total_limit < 1:
        raise ValueError(
            "training.checkpoint_total_limit must be at least 1 or null, "
            f"got {total_limit}"
        )
    return total_limit


def resolve_model_checkpoint_path(path):
    """Resolve a model-only checkpoint from a file, checkpoint, or run path."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise ValueError(
            f"Model checkpoint must be a file or directory: {checkpoint_path}"
        )

    candidates = [
        checkpoint_path / filename
        for filename in _MODEL_CHECKPOINT_FILENAMES
        if (checkpoint_path / filename).is_file()
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous model checkpoint directory {checkpoint_path}; found "
            f"{[candidate.name for candidate in candidates]}. Pass the exact "
            "checkpoint file."
        )
    if candidates:
        return candidates[0]

    # Accept either the Accelerate ``checkpoints`` directory or an experiment
    # root containing it. Selection is numeric, never based on mtime.
    checkpoint_directory = checkpoint_path
    nested_checkpoint_directory = checkpoint_path / "checkpoints"
    if nested_checkpoint_directory.is_dir():
        checkpoint_directory = nested_checkpoint_directory
    latest = latest_checkpoint(checkpoint_directory)
    if latest:
        return resolve_model_checkpoint_path(latest)

    expected = ", ".join(_MODEL_CHECKPOINT_FILENAMES)
    raise FileNotFoundError(
        f"No model checkpoint found in {checkpoint_path}; expected exactly "
        f"one of [{expected}] directly, or a checkpoints/checkpoint_<number> "
        "tree."
    )


def report_model_load_incompatibilities(missing_keys, unexpected_keys, *, log=None):
    """Report every non-strict model load mismatch without hiding legacy keys."""
    log = logger if log is None else log
    missing_keys = tuple(missing_keys)
    unexpected_keys = tuple(unexpected_keys)
    deprecated_unexpected_keys = tuple(
        key
        for key in unexpected_keys
        if any(
            component in _DEPRECATED_CHECKPOINT_KEY_COMPONENTS
            for component in key.split(".")
        )
    )
    unexpected_keys = tuple(
        key
        for key in unexpected_keys
        if key not in deprecated_unexpected_keys
    )

    if deprecated_unexpected_keys:
        log.warning(
            "Deprecated unexpected checkpoint keys from removed components: "
            f"{list(deprecated_unexpected_keys)}"
        )
    if missing_keys:
        log.warning(f"Missing model checkpoint keys: {list(missing_keys)}")
    if unexpected_keys:
        log.warning(f"Unexpected model checkpoint keys: {list(unexpected_keys)}")
    if not missing_keys and not unexpected_keys:
        log.info("No active model checkpoint key mismatches.")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "deprecated_unexpected_keys": deprecated_unexpected_keys,
    }


@TRAINER_REGISTRY.register()
class LeoTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.exp_dir = cfg.exp_dir
        self.mode = cfg.mode

        self.epochs = cfg.training.epochs
        self.save_interval = cfg.training.save_interval
        self.val_interval = cfg.eval.val_interval
        self.num_batch_val = cfg.eval.num_batch_val

        self.debug = getattr(cfg, 'debug', False)
        self.grad_norm = getattr(cfg.training, 'grad_norm', 1.0)
        self.train_infer = getattr(cfg, 'train_infer', False)
        self.use_custom_sampler = getattr(cfg.task.leomix.LeoMix, 'use_custom_sampler', False)
        self.vggt_fusion_learnable = getattr(cfg.vision3d, 'fusion_learnable', False)

        # initialize accelerator
        gradient_accumulation_steps = cfg.training.get('gradient_accumulation_steps', 1)
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=24))

        self.accelerator = CustomAccelerator(
            project_config=ProjectConfiguration(
                project_dir=self.exp_dir,
                automatic_checkpoint_naming=True,
                total_limit=resolve_checkpoint_total_limit(cfg.training),
            ),
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=[process_group_kwargs]
        )

        # dataset, dataloader, evaluator
        self.data_loaders = {'train': {}, 'val': {}, 'test': {}}
        self.evaluators = {}
        self.eval_metrics = {}
        self.task_metrics_weights = getattr(cfg.task, 'task_metrics_weights', {})
        for task_name in cfg.task.keys():
            if task_name in ['training', 'task_metrics_weights']:   # hyperparameters
                continue
            for dataset_name in cfg.task[task_name].keys():
                for mode in cfg.task[task_name][dataset_name].mode:
                    if task_name not in self.data_loaders[mode]:
                        self.data_loaders[mode][task_name] = {}
                    self.data_loaders[mode][task_name][dataset_name] = build_dataloader_leo(
                        cfg=cfg, split=mode, dataset_name=dataset_name,
                        dataloader_args=cfg.task[task_name][dataset_name].train_dataloader_args if mode == 'train' \
                                        else cfg.task[task_name][dataset_name].eval_dataloader_args,
                    )
                if 'evaluator' in cfg.task[task_name][dataset_name]:
                    if task_name not in self.evaluators:
                        self.evaluators[task_name] = {}
                    self.evaluators[task_name][dataset_name] = build_eval_leo(
                        cfg, dataset_name=dataset_name, evaluator_name=cfg.task[task_name][dataset_name].evaluator
                    )
                    if task_name not in self.eval_metrics:
                        self.eval_metrics[task_name] = {}

        assert len(self.data_loaders['train']['leomix']) <= 1, "LEO requires only one training set"

        # prepare dataloaders
        all_loaders, all_loader_keys = [], []
        for mode, mode_loaders in self.data_loaders.items():
            if mode == 'train':
                continue   # left to prepare with model
            for task, task_loaders in mode_loaders.items():
                for dataset, dataloader in task_loaders.items():
                    all_loader_keys.append((mode, task, dataset))
                    all_loaders.append(dataloader)
        if len(all_loaders) == 0:
            logger.info("No eval dataloader to prepare")
        else:
            if len(all_loaders) > 1:
                accelerate_loaders = self.accelerator.prepare(*all_loaders)
            elif len(all_loaders) == 1:
                accelerate_loaders = [self.accelerator.prepare(*all_loaders)]
            for k, v in zip(all_loader_keys, accelerate_loaders):
                self.data_loaders[k[0]][k[1]][k[2]] = v

        # build model
        self.model = LeoAgent(cfg)
        learnable_named_params = self.model.get_learnable_named_params()
        self.accelerator.learn_params_list = list(learnable_named_params.keys())

        # group optim params
        # optim_params = self.get_grouped_params(learnable_named_params, cfg)
        if cfg.llm.lora.flag:
            optim_params = list(learnable_named_params.values())
        else:
            optim_params = self.get_grouped_params(learnable_named_params, cfg)

        # load pretrained model weights
        if cfg.pretrained_ckpt_path:
            logger.info(f"Load pretrained model from {cfg.pretrained_ckpt_path}")
            self.load(path=cfg.pretrained_ckpt_path, model_only=True, prepared=False)
            pretrained_ckpt_loaded = True
        else:
            pretrained_ckpt_loaded = False

        # prepare model, optimizer and scheduler
        if self.use_custom_sampler:
            loader = list(self.data_loaders['train']['leomix'].values())[0]
            current_len = len(loader)
            skipped_len = getattr(loader.sampler, 'skipped_batches', 0)
            true_len = current_len + skipped_len
            logger.info(f"Scheduler Init: Active={current_len}, Skipped={skipped_len}, Total={true_len}")
            
            per_rank_train_size = true_len
            tensor = torch.tensor([per_rank_train_size], device=self.accelerator.device)

            if self.accelerator.num_processes <= 1:
                per_rank_list = [per_rank_train_size]
            else:
                gathered = self.accelerator.gather(tensor)
                per_rank_list = [int(x.item()) for x in gathered]
            train_size = sum(per_rank_list)
            total_steps = ceil(train_size / gradient_accumulation_steps) * cfg.task.training.epochs
        else:
            train_size = len(list(self.data_loaders['train']['leomix'].values())[0])
            total_steps = ceil(train_size / gradient_accumulation_steps) * cfg.task.training.epochs
        
        if getattr(cfg.task.training, 'dummy_lr_decay', False):
            total_steps *= 5
        if len(optim_params) > 0:
            self.optimizer, self.scheduler = build_optim(cfg, optim_params, total_steps=total_steps)
        
        objects_to_prepare = [self.model]
        if len(optim_params) > 0:
            objects_to_prepare.extend([self.optimizer, self.scheduler])
        if not self.use_custom_sampler:
            objects_to_prepare.append( self.data_loaders['train']['leomix']['LeoMix'])
        
        if self.accelerator.distributed_type == "DEEPSPEED":
            try:
                bs = cfg.dataloader.train.batchsize
            except:
                bs = 4
            
            self.accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = bs

        prepared_objects = self.accelerator.prepare(*objects_to_prepare)
        prepared_iter = iter(prepared_objects)
        self.model = next(prepared_iter)
        if len(optim_params) > 0:
            self.optimizer = next(prepared_iter)
            self.scheduler = next(prepared_iter)
        if not self.use_custom_sampler:
            self.data_loaders['train']['leomix']['LeoMix'] = next(prepared_iter)

        # Bind the prepared model to VLN evaluators
        for task_name in self.evaluators:
            for dataset_name in self.evaluators[task_name]:
                eval_inst = self.evaluators[task_name][dataset_name]
                if hasattr(eval_inst, "update_model"):
                    eval_inst.update_model(self.model)

        self.exp_tracker = Tracker(cfg)
        self.accelerator.register_for_checkpointing(self.exp_tracker)
        if self.accelerator.distributed_type == 'DEEPSPEED' and self.model.zero_optimization_partition_weights():
            self.zero3 = True
        else:
            self.zero3 = False

        # resume training
        if self.mode == 'train':
            resume_ckpt = latest_checkpoint(os.path.join(self.exp_dir, 'checkpoints'))
            if resume_ckpt:
                logger.info(f"Train: resume from {resume_ckpt}")
                self.load(path=resume_ckpt, model_only=False, prepared=True)
            elif pretrained_ckpt_loaded:
                logger.info(f"Train: start from {cfg.pretrained_ckpt_path}")
            else:
                logger.info("Train: start from scratch")
        else:
            self_best_ckpt = os.path.join(self.exp_dir, 'best.pth')
            if os.path.exists(self_best_ckpt):
                logger.info(f"Eval: load model from {self_best_ckpt}")
                self.load(path=self_best_ckpt, model_only=True, prepared=True)
            elif pretrained_ckpt_loaded:
                logger.info(f"Eval: load model from {cfg.pretrained_ckpt_path}")
            else:
                logger.info(f"Eval: no checkpoint to load")

        if not self.debug:
            self.accelerator.init_trackers(
                project_name=cfg.name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                init_kwargs={
                    'wandb': {
                        'name': self.exp_tracker.exp_name, 'entity': cfg.logger.entity,
                        'id': self.exp_tracker.run_id, 'resume': True,
                    }
                }
            )
    
    def get_grouped_params(self, named_params, cfg):
        # 1. Get Base Hyperparameters
        base_lr = float(cfg.training.optim.args.lr)
        weight_decay = float(cfg.training.optim.args.weight_decay)

        # 2. Get Specific LRs (Use .get() to make them optional)
        projector_lr = getattr(cfg.task.training, 'mm_projector_lr', base_lr)
        vision_lr = getattr(cfg.task.training, 'vision_tower_lr', base_lr)

        # 3. Define Group Logic
        groups = {
            'projector': {'params': [], 'lr': projector_lr, 'keywords': ['merger', 'embed_tokens', 'vggt_embedding_merger']},
            'vision':    {'params': [], 'lr': vision_lr,    'keywords': ['visual']},
            'base':      {'params': [], 'lr': base_lr,      'keywords': []} # Fallback
        }

        def get_group_key(name):
            # Check projector first
            for kw in groups['projector']['keywords']:
                if kw in name:
                    return 'projector'
            
            # Check vision (ensure it's not a merger insdie vision)
            for kw in groups['vision']['keywords']:
                if kw in name and 'merger' not in name and 'embed_tokens' not in name:
                    return 'vision'
            
            # Default
            return 'base'
        
        no_decay_keywords = {'bias', 'LayerNorm.weight', 'layernorm.weight', 'norm.weight'}

        # 4. Bucket Parameters
        final_param_groups = []
        
        # Temporary buckets to hold decay/no_decay split per group
        buckets = {
            k: {'decay': [], 'no_decay': []} for k in groups.keys()
        }

        for name, p in named_params.items():
            if not p.requires_grad: 
                continue # Should not happen if input is learnable_named_params, but safe check
            
            group_key = get_group_key(name)
            
            if any(nd in name for nd in no_decay_keywords):
                buckets[group_key]['no_decay'].append(p)
            else:
                buckets[group_key]['decay'].append(p)

        # 5. Construct Final Optimizer List
        for key, bucket in buckets.items():
            lr = groups[key]['lr']
            
            # Only add groups that actually have parameters
            # This handles the case where you froze the vision tower or projector
            if bucket['decay']:
                final_param_groups.append({
                    'params': bucket['decay'],
                    'weight_decay': weight_decay,
                    'lr': lr,
                    'name': f"{key}_decay"
                })
                logger.info(f"Optim Group [{key}_decay]: {len(bucket['decay'])} params, lr={lr}")
                
            if bucket['no_decay']:
                final_param_groups.append({
                    'params': bucket['no_decay'],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'name': f"{key}_no_decay"
                })
                logger.info(f"Optim Group [{key}_no_decay]: {len(bucket['no_decay'])} params, lr={lr}")

        return final_param_groups

    def forward(self, model, data_dict, inference=False, **kwargs):
        if inference:
            if isinstance(model, nn.Module):
                return model.generate(data_dict, **kwargs)
            else:
                with unwrap_model_for_generation(model, self.accelerator, gather_deepspeed3_params=self.zero3) as unwrapped_model:
                    return unwrapped_model.generate(data_dict, **kwargs)
        else:
            return model(data_dict)

    def backward(self, loss):
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def one_step_sft(self, data_dict):
        # forward
        data_dict = self.forward(self.model, data_dict, inference=False)

        # calculate loss and optimize
        loss = data_dict['loss'].mean()
        self.backward(loss)

        # record
        loss_dict = {'overall': self.accelerator.gather_for_metrics(loss).mean()}
        for k, v in data_dict.items():
            if k.startswith('loss_'):
                loss_dict[k] = self.accelerator.gather_for_metrics(v).mean()

        return loss_dict

    def train_step(self, epoch):
        logger.info(f"Start training epoch {epoch+1}")
        self.model.train()
        loader = list(self.data_loaders['train']['leomix'].values())[0]

        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        
        skipped_batches = getattr(loader.sampler, 'skipped_batches', 0)
        total_batches = len(loader) + skipped_batches

        pbar = trange(total_batches, disable=(not self.accelerator.is_main_process))

        if skipped_batches > 0:
            logger.info(f"Resuming: Skipping {skipped_batches} batches in tqdm bar.")
            pbar.update(skipped_batches)

        checkpoint_due = False
        validation_due = False
        active_batches = min(len(loader), 501) if self.debug else len(loader)

        for i, data_dict in enumerate(loader):
            if i >= active_batches:
                break
            if self.use_custom_sampler:
                data_dict = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
            if self.use_custom_sampler and 'episode_id' not in data_dict:
                raise KeyError(
                    "Custom-sampler training batches must include episode_id "
                    "so checkpoint progress can follow consumed data."
                )
            batch_episode_ids = data_dict.get('episode_id')
            if isinstance(batch_episode_ids, torch.Tensor):
                batch_episode_ids = (
                    batch_episode_ids.detach().cpu().reshape(-1).tolist()
                )
            elif isinstance(batch_episode_ids, (str, bytes)):
                batch_episode_ids = (batch_episode_ids,)
            elif batch_episode_ids is not None:
                batch_episode_ids = tuple(batch_episode_ids)

            with track_unprepared_dataloader_batch(
                self.accelerator,
                enabled=self.use_custom_sampler,
                end_of_dataloader=(i + 1 == active_batches),
            ):
                with self.accelerator.accumulate(self.model):
                    loss_dict = self.one_step_sft(data_dict)

                    if self.vggt_fusion_learnable:
                        if isinstance(self.model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
                            base_vlm = self.model.module.model
                        else:
                            base_vlm = self.model.model
                        if hasattr(base_vlm, 'base_model'):
                            base_vlm = base_vlm.base_model.model
                        loss_dict['vggt_lambda'] = base_vlm.model.vggt_lambda.detach().cpu().item()

                    self.log(loss_dict, mode='train', note='loss')
                    self.exp_tracker.step_loader()
                    pbar.update(1)

                    checkpoint_due = checkpoint_due or (
                        self.exp_tracker.loader_step % self.save_interval == 0
                    )
                    validation_due = validation_due or (
                        self.exp_tracker.loader_step % self.val_interval == 0
                    )

            if self.use_custom_sampler:
                if not hasattr(loader.sampler, 'mark_batch_consumed'):
                    raise TypeError(
                        "Custom training sampler must implement "
                        "mark_batch_consumed(episode_ids)."
                    )
                loader.sampler.mark_batch_consumed(batch_episode_ids)

            # A full-state checkpoint does not contain partial accumulated
            # gradients, so interval hooks run only after an optimizer boundary.
            if self.accelerator.sync_gradients:
                if checkpoint_due:
                    self.save(model_only=False)
                    if hasattr(loader.sampler, 'sync_progress'):
                        loader.sampler.sync_progress()
                    checkpoint_due = False

                if validation_due:
                    is_best = self.val_step(epoch)
                    if is_best:
                        self.save('best.pth', model_only=True)
                    self.model.train()
                    validation_due = False

        logger.info(f"Finish training epoch {epoch+1}")

    def val_step(self, epoch):
        self.exp_tracker.val_count += 1
        logger.info(f"Start validation {self.exp_tracker.val_count}")
        self.model.eval()
        with unwrap_model_for_generation(self.model, self.accelerator, gather_deepspeed3_params=self.zero3) as unwrapped_model:
            for task_name in self.evaluators:
                for dataset_name, evaluator in self.evaluators[task_name].items():
                    if hasattr(evaluator, 'update_model'):
                        evaluator.update_model(unwrapped_model)
                    evaluator.eval_action(rank=self.accelerator.process_index, epoch=epoch)
                    self.accelerator.wait_for_everyone()
                    _, results = evaluator.record(is_main_process=self.accelerator.is_main_process)

                    self.log(results, mode='val', note=f'{task_name}/{dataset_name}')
                    logger.info(f"{dataset_name}: {results}")
                    self.accelerator.wait_for_everyone()
                    evaluator.reset()

        return True

    def test_step(self):
        logger.info("Start final testing")
        self.model.eval()
        with unwrap_model_for_generation(self.model, self.accelerator, gather_deepspeed3_params=self.zero3) as unwrapped_model:
            for task_name in self.data_loaders['test']:
                for dataset_name in self.data_loaders['test'][task_name]:
                    loader = self.data_loaders['test'][task_name][dataset_name]
                    pbar = trange(len(loader), disable=(not self.accelerator.is_main_process))
                    for data_dict in loader:
                        data_dict = self.forward(unwrapped_model, data_dict, inference=True)

                        # gather, ignore tensors, convert tensors required by evaluator to list
                        gather_dict = {}
                        for k in gather_keys:
                            if k in data_dict:
                                v = data_dict[k]
                                if isinstance(v, torch.Tensor):
                                    gather_dict[k] = v.cpu().tolist()
                                else:
                                    gather_dict[k] = v

                        gather_dict = self.accelerator.gather_for_metrics(gather_dict)

                        self.evaluators[task_name][dataset_name].update(gather_dict)
                        pbar.update(1)

                    _, results = self.evaluators[task_name][dataset_name].record(
                        split='test', is_main_process=self.accelerator.is_main_process
                    )

                    self.log(results, mode='test', note=f'{task_name}/{dataset_name}')
                    logger.info(f"{dataset_name}: {results}")
                    self.evaluators[task_name][dataset_name].reset()

        logger.info("Finish testing")

    def log(self, results, mode='train', note='default'):
        log_dict = {}
        for key, val in results.items():
            log_dict[f'{mode}/{note}/{key}'] = val

        if mode == 'train':
            lrs = self.scheduler.get_lr()
            for i, lr in enumerate(lrs):
                log_dict[f'train/lr/group_{i}'] = lr

        self.accelerator.log(log_dict)

    def save(self, name='best.pth', model_only=False):
        self.accelerator.wait_for_everyone()
        if model_only:
            path = os.path.join(self.exp_dir, name)
            os.makedirs(path, exist_ok=True)
            model_state_dict = self.accelerator.get_state_dict(self.model)
            # automatically filter non-learnable params, and save on main_process
            self.accelerator.save(model_state_dict, os.path.join(path, 'pytorch_model.bin'))
        else:
            self.accelerator.save_state(exclude_frozen_parameters=True)   # self.exp_dir / checkpoints
        self.accelerator.wait_for_everyone()

    def load(self, path, model_only=False, prepared=True):
        if model_only:
            model_ckpt_path = resolve_model_checkpoint_path(path)
            if model_ckpt_path.suffix == ".safetensors":
                from safetensors.torch import load_file as safe_load
                model_state_dict = safe_load(str(model_ckpt_path), device="cpu")
            else:
                model_state_dict = torch.load(model_ckpt_path, weights_only=True, map_location='cpu')
            if prepared:
                # training finished
                if self.zero3:
                    params_to_fetch = [
                        p for p in self.model.module.parameters()
                        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
                    ]
                    with deepspeed.zero.GatheredParameters(params_to_fetch, modifier_rank=0):
                        missing_keys, unexpected_keys = self.model.module.load_state_dict(model_state_dict, strict=False)
                        # assume no more training after loading weights, as this would be problematic if continue to training
                else:
                    if isinstance(
                        self.model, (nn.parallel.DistributedDataParallel, nn.DataParallel)
                    ):
                        missing_keys, unexpected_keys = self.model.module.load_state_dict(model_state_dict, strict=False)
                    else:
                        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            else:
                # before training
                missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            report_model_load_incompatibilities(missing_keys, unexpected_keys)
        else:
            # resume training (called after accelerator.prepare)
            if self.accelerator.distributed_type == 'DEEPSPEED':
                self.accelerator.load_state(path, load_module_strict=False)
            else:
                self.accelerator.load_state(path, strict=False)
            self.accelerator.project_configuration.iteration = (
                checkpoint_index(path) + 1
            )
        logger.info(f"Successfully loaded from {str(path)}, load_model_only = {model_only}, prepared = {prepared}")

    def run(self):
        if self.mode == 'train':
            start_epoch = self.exp_tracker.epoch
            for epoch in range(start_epoch, self.epochs):
                self.train_step(epoch)
                self.exp_tracker.step()
                self.save(model_only=False)   # automatic checkpointing
                loader = list(self.data_loaders['train']['leomix'].values())[0]
                if hasattr(loader.sampler, 'sync_progress'):
                    loader.sampler.sync_progress()
                self.val_step(epoch)
        else:
            self.val_step(0)

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
