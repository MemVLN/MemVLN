"""Fixed-update trainer contracts using real CPU PyTorch/Accelerate components.

The production trainer methods are loaded unchanged via AST so these tests do
not import Habitat, the VLM, or DeepSpeed. Optimizer, scheduler and accumulation
behavior use the actual installed Accelerate implementation.
"""

import ast
import copy
import importlib.util
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState


ROOT = Path(__file__).resolve().parents[1]


def _load_symbols(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class _Progress:
    def update(self, count):
        pass


_namespace = {
    "contextmanager": contextmanager,
    "torch": torch,
    "nn": nn,
    "Path": Path,
    "os": __import__("os"),
    "logger": Mock(),
    "trange": lambda *args, **kwargs: _Progress(),
    "checkpoint_index": lambda path: int(Path(path).name.rsplit("_", 1)[1]),
}
_load_symbols("trainer/memvln_trainer.py", {
    "_UnpreparedDataLoaderState", "track_unprepared_dataloader_batch",
    "resolve_max_updates", "validate_fixed_update_backend",
    "configure_fixed_update_scheduler", "fixed_update_remaining_batches",
    "MemVLNTrainer",
}, _namespace)
Trainer = _namespace["MemVLNTrainer"]
resolve_max_updates = _namespace["resolve_max_updates"]
configure_scheduler = _namespace["configure_fixed_update_scheduler"]

_tracker_namespace = {
    "cp": copy, "datetime": datetime,
    "wandb": SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "test")),
    "math": __import__("math"),
}
_load_symbols("trainer/build.py", {"Tracker", "linear_warmup_cosine_decay"}, _tracker_namespace)
Tracker = _tracker_namespace["Tracker"]

_sampler_spec = importlib.util.spec_from_file_location(
    "memvln_training_test_sampler", ROOT / "data/probabilistic_episode_sampler.py",
)
_sampler_module = importlib.util.module_from_spec(_sampler_spec)
_sampler_spec.loader.exec_module(_sampler_module)
ProbabilisticEpisodeSampler = _sampler_module.ProbabilisticEpisodeSampler


class _EpisodeFixtureDataset(Dataset):
    """Small real index-token dataset with episodes longer than the checkpoint prefix."""

    def __init__(self):
        self.stream_episode_to_indices = {
            "r2r-a": list(range(9)), "rxr-a": list(range(9, 20)),
        }
        self.stream_episode_to_domain = {"r2r-a": "r2r", "rxr-a": "rxr"}
        self.clip_ids = list(range(9)) + list(range(11))

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, token):
        index, occurrence_id = token
        return {
            "input": torch.ones(1), "episode_id": occurrence_id,
            "clip_id": self.clip_ids[index],
        }


class _Sampler:
    is_probabilistic_episode = True

    def __init__(self, budget):
        self.budget = budget
        self.consumed = 0
        self.set_epoch = Mock(side_effect=AssertionError("must not reset fixed sampler"))

    def mark_batch_consumed(self, episode_ids):
        if len(episode_ids) != 1:
            raise AssertionError("expected one item")
        self.consumed += 1

    def get_progress_metrics(self):
        return {"consumed_items": self.consumed}

    def state_dict(self):
        return {"budget": self.budget, "consumed": self.consumed}

    def load_state_dict(self, state):
        self.budget = state["budget"]
        self.consumed = state["consumed"]


class _Loader:
    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return self.sampler.budget - self.sampler.consumed

    def __iter__(self):
        for _ in range(len(self)):
            yield {"input": torch.ones(1, 1), "episode_id": ["episode"]}


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.update_count = 0

    def step(self, closure=None):
        self.update_count += 1
        return super().step(closure)


def _make_trainer(max_updates=4, accumulation=3, consumed_updates=0):
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=accumulation)
    model = nn.Linear(1, 1, bias=False)
    model.reset_episode_state = Mock()
    optimizer = _CountingSGD(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.01, 1 - step / max_updates))
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = accelerator
    trainer.model, trainer.optimizer, trainer.scheduler = accelerator.prepare(model, optimizer, scheduler)
    configure_scheduler(trainer.scheduler)
    trainer.max_updates = max_updates
    trainer.fixed_update_schedule_contract = {"name": "test", "warmup_steps": 0, "base_lrs": [0.1]}
    trainer.gradient_accumulation_steps = accumulation
    trainer.exp_tracker = Tracker(SimpleNamespace(note="test"))
    trainer.exp_tracker.optimizer_step = consumed_updates
    trainer.exp_tracker.loader_step = consumed_updates * accumulation
    trainer.use_custom_sampler = True
    trainer.debug = False
    trainer.vggt_fusion_learnable = False
    trainer.grad_norm = None
    trainer.save_interval = 5  # deliberately not divisible by accumulation
    trainer.val_interval = 1  # fixed mode must defer every in-flight validation
    trainer.log = Mock()
    trainer.save = Mock()
    trainer.val_step = Mock(return_value=False)
    trainer.mode = "train"
    trainer.epochs = 99  # fixed mode must ignore this
    sampler = _Sampler(max_updates * accumulation)
    sampler.consumed = consumed_updates * accumulation
    trainer.data_loaders = {"train": {"dataset_mix": {"VLNDatasetMix": _Loader(sampler)}}}

    def one_step(self, batch):
        loss = self.model(batch["input"]).square().mean()
        self.backward(loss)
        return {"overall": loss.detach()}

    trainer.one_step_sft = MethodType(one_step, trainer)
    return trainer, optimizer, sampler


class FixedUpdateTrainingTests(unittest.TestCase):
    def setUp(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()

    def tearDown(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()

    def test_budget_validation(self):
        self.assertIsNone(resolve_max_updates({}))
        self.assertEqual(resolve_max_updates({"max_updates": 12}), 12)
        for invalid in (True, 1.5, "12"):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                resolve_max_updates({"max_updates": invalid})
        for invalid in (0, -1):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                resolve_max_updates({"max_updates": invalid})

    def test_backend_and_precision_are_explicit(self):
        validate = _namespace["validate_fixed_update_backend"]
        for backend in ("NO", "MULTI_CPU", "MULTI_GPU"):
            validate(SimpleNamespace(distributed_type=backend, mixed_precision="bf16"))
        for backend in ("DEEPSPEED", "FSDP"):
            with self.assertRaises(ValueError):
                validate(SimpleNamespace(distributed_type=backend, mixed_precision="bf16"))
        with self.assertRaises(ValueError):
            validate(SimpleNamespace(distributed_type="NO", mixed_precision="fp16"))

    def test_accumulation_consumes_exact_budget_and_scheduler_steps_once(self):
        trainer, optimizer, sampler = _make_trainer()
        with patch("accelerate.scheduler.AcceleratorState", return_value=SimpleNamespace(num_processes=8)):
            trainer.train_step(0)
        self.assertEqual(optimizer.update_count, 4)
        self.assertEqual(trainer.scheduler.scheduler.last_epoch, 4)
        self.assertEqual(trainer.exp_tracker.optimizer_step, 4)
        self.assertEqual(sampler.consumed, 12)
        self.assertEqual(trainer.exp_tracker.loader_step, 12)
        self.assertEqual(trainer.save.call_count, 2)
        trainer.val_step.assert_not_called()
        sampler.set_epoch.assert_not_called()

    def test_run_ignores_epochs_and_only_validates_after_budget(self):
        trainer, optimizer, sampler = _make_trainer()
        trainer.run()
        self.assertEqual(optimizer.update_count, 4)
        self.assertEqual(sampler.consumed, 12)
        trainer.val_step.assert_called_once_with(0)
        self.assertEqual(trainer.exp_tracker.epoch, 0)

    def test_resume_consumes_only_remaining_updates(self):
        trainer, optimizer, sampler = _make_trainer(consumed_updates=2)
        trainer.train_step(0)
        self.assertEqual(optimizer.update_count, 2)
        self.assertEqual(trainer.exp_tracker.optimizer_step, 4)
        self.assertEqual(sampler.consumed, 12)

    def test_inconsistent_sampler_budget_fails_before_forward(self):
        trainer, optimizer, sampler = _make_trainer()
        sampler.budget -= 1
        with self.assertRaisesRegex(ValueError, "remaining batches"):
            trainer.train_step(0)
        self.assertEqual(optimizer.update_count, 0)

    def test_loader_exhaustion_cannot_silently_shorten_training(self):
        trainer, optimizer, sampler = _make_trainer()

        class ShortLoader(_Loader):
            def __iter__(self):
                return iter(list(super().__iter__())[:-3])

        trainer.data_loaders["train"]["dataset_mix"]["VLNDatasetMix"] = ShortLoader(sampler)
        with self.assertRaisesRegex(RuntimeError, "sampler ended early"):
            trainer.train_step(0)

    def test_skipped_optimizer_update_is_rejected(self):
        trainer, optimizer, sampler = _make_trainer()
        with patch.object(Accelerator, "optimizer_step_was_skipped", new_callable=PropertyMock, return_value=True):
            with self.assertRaisesRegex(RuntimeError, "optimizer update was skipped"):
                trainer.train_step(0)
        self.assertEqual(trainer.exp_tracker.optimizer_step, 0)

    def test_completed_run_does_not_train_or_save_again(self):
        trainer, optimizer, sampler = _make_trainer(consumed_updates=4)
        trainer.run()
        self.assertEqual(optimizer.update_count, 0)
        trainer.save.assert_not_called()
        trainer.val_step.assert_not_called()

    def test_debug_stop_preserves_accumulation_boundary(self):
        trainer, optimizer, sampler = _make_trainer(max_updates=200, accumulation=3)
        trainer.debug = True
        trainer.train_step(0)
        self.assertEqual(optimizer.update_count, 167)
        self.assertEqual(sampler.consumed, 501)

    def test_zero_warmup_is_valid(self):
        schedule = _tracker_namespace["linear_warmup_cosine_decay"]
        self.assertEqual(schedule(0, 0, 10), 1.0)
        self.assertAlmostEqual(schedule(5, 0, 10), 0.5)
        self.assertEqual(schedule(10, 0, 10), 1e-5)

    def test_sampling_logs_include_every_rank_and_global_exposure(self):
        trainer = Trainer.__new__(Trainer)
        trainer.log = Mock()
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"), num_processes=2, is_main_process=True,
            gather=Mock(return_value=torch.tensor([10, 3, 7, 10, 8, 2], dtype=torch.float64)),
        )
        sampler = SimpleNamespace(get_progress_metrics=lambda: {
            "rxr/items": 7, "consumed_items": 10, "r2r/items": 3,
        })
        trainer.log_sampling_progress(sampler)
        sent = trainer.accelerator.gather.call_args.args[0]
        self.assertTrue(torch.equal(sent, torch.tensor([10, 3, 7], dtype=torch.float64)))
        results = trainer.log.call_args.args[0]
        self.assertEqual(results["rank_0/r2r/items"], 3)
        self.assertEqual(results["rank_1/r2r/items"], 8)
        self.assertEqual(results["global/consumed_items"], 20)
        self.assertEqual(results["global/r2r/items"], 11)
        self.assertEqual(results["global/rxr/items"], 9)
        trainer.log.assert_called_once_with(results, mode="train", note="sampling")

        trainer.log.reset_mock()
        trainer.accelerator.is_main_process = False
        trainer.log_sampling_progress(sampler)
        self.assertEqual(trainer.accelerator.gather.call_count, 2)
        trainer.log.assert_not_called()

    def test_tracker_round_trip_and_legacy_checkpoint(self):
        tracker = Tracker(SimpleNamespace(note="test"))
        tracker.step_optimizer()
        tracker.step_loader()
        tracker.step()
        self.assertEqual(tracker.optimizer_step, 1)
        checkpoint = tracker.state_dict()
        restored = Tracker(SimpleNamespace(note="restored"))
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.optimizer_step, 1)
        del checkpoint["optimizer_step"]
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.optimizer_step, 0)

    def test_checkpoint_sidecar_matches_actual_checkpoint_directory(self):
        trainer, optimizer, sampler = _make_trainer()
        trainer.train_step(0)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_7"
            checkpoint.mkdir()
            trainer.accelerator.save_state = Mock(return_value=str(checkpoint))
            Trainer.save(trainer, model_only=False)
            saved = torch.load(checkpoint / "sampler_rank_0.pt", weights_only=True)
            self.assertEqual(saved["optimizer_step"], 4)
            self.assertEqual(saved["sampler"]["consumed"], 12)
            self.assertEqual(saved["gradient_accumulation_steps"], 3)
            self.assertFalse((checkpoint / "sampler_rank_0.pt.tmp").exists())

    def test_resume_requires_every_rank_sidecar(self):
        trainer, optimizer, sampler = _make_trainer()
        with TemporaryDirectory() as directory:
            trainer.accelerator.load_state = Mock()
            with self.assertRaisesRegex(FileNotFoundError, "missing sampler state"):
                trainer.load(directory, model_only=False)
            trainer.accelerator.load_state.assert_not_called()

    def test_missing_other_rank_also_invalidates_checkpoint(self):
        trainer, optimizer, sampler = _make_trainer()
        with TemporaryDirectory() as directory:
            (Path(directory) / "sampler_rank_0.pt").touch()
            trainer.accelerator.load_state = Mock()
            with patch.object(Accelerator, "num_processes", new_callable=PropertyMock, return_value=2):
                with self.assertRaisesRegex(FileNotFoundError, "ranks \\[1\\]"):
                    trainer.load(directory, model_only=False)
            trainer.accelerator.load_state.assert_not_called()

    def test_resume_validates_metadata_and_clears_episode_cache(self):
        trainer, optimizer, sampler = _make_trainer(consumed_updates=2)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_7"
            checkpoint.mkdir()
            trainer.accelerator.save_state = Mock(return_value=str(checkpoint))
            Trainer.save(trainer, model_only=False)
            trainer.accelerator.load_state = Mock()
            sampler.consumed = 0
            trainer.load(checkpoint, model_only=False)
            self.assertEqual(sampler.consumed, 6)
            trainer.model.reset_episode_state.assert_called_once_with()
            self.assertEqual(trainer.accelerator.project_configuration.iteration, 8)
            trainer.max_updates = 5
            with self.assertRaisesRegex(ValueError, "max_updates"):
                trainer.load(checkpoint, model_only=False)

    def test_epoch_path_keeps_final_partial_accumulation_and_validation(self):
        trainer, optimizer, sampler = _make_trainer(max_updates=4, accumulation=3)
        trainer.max_updates = None
        sampler.budget = 11
        sampler.set_epoch = Mock()
        trainer.train_step(0)
        sampler.set_epoch.assert_called_once_with(0)
        self.assertEqual(optimizer.update_count, 4)
        self.assertEqual(trainer.val_step.call_count, 4)

    def test_real_accelerate_checkpoint_restores_optimizer_scheduler_and_tracker(self):
        trainer, optimizer, sampler = _make_trainer(consumed_updates=2)
        trainer.accelerator.register_for_checkpointing(trainer.exp_tracker)
        trainer.train_step(0)
        with TemporaryDirectory() as directory:
            trainer.accelerator.project_configuration.project_dir = directory
            trainer.accelerator.project_configuration.automatic_checkpoint_naming = True
            Trainer.save(trainer, model_only=False)
            saved_model = copy.deepcopy(trainer.model.state_dict())
            saved_lr = trainer.scheduler.get_last_lr()
            trainer.exp_tracker.optimizer_step = 0
            sampler.consumed = 0
            with torch.no_grad():
                trainer.model.weight.add_(99)
            trainer.load(Path(directory) / "checkpoints" / "checkpoint_0", model_only=False)
            self.assertEqual(trainer.exp_tracker.optimizer_step, 4)
            self.assertEqual(sampler.consumed, 12)
            self.assertEqual(trainer.scheduler.get_last_lr(), saved_lr)
            self.assertTrue(torch.equal(trainer.model.weight, saved_model["weight"]))

    def test_real_sampler_dataloader_trainer_checkpoint_replays_within_update_budget(self):
        dataset = _EpisodeFixtureDataset()

        def make_streaming_trainer():
            trainer, optimizer, _ = _make_trainer(max_updates=16, accumulation=2)
            sampler = ProbabilisticEpisodeSampler(
                dataset, num_samples=32, dataset_probabilities={"r2r": 0.5, "rxr": 0.5},
                seed=42, rank=0, num_replicas=1,
            )
            # A worker exercises real DataLoader prefetch. Checkpoint state
            # must still include only the six acknowledged microbatches.
            loader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=1)
            trainer.data_loaders["train"]["dataset_mix"]["VLNDatasetMix"] = loader
            trainer.accelerator.register_for_checkpointing(trainer.exp_tracker)
            records = []
            original_step = trainer.one_step_sft

            def record_step(batch):
                records.append((batch["episode_id"][0], batch["clip_id"].item()))
                return original_step(batch)

            trainer.one_step_sft = record_step
            return trainer, optimizer, sampler, loader, records

        class InterruptAfterCheckpoint(Exception):
            pass

        trainer, first_optimizer, sampler, loader, first_records = make_streaming_trainer()
        trainer.save_interval = 6
        with TemporaryDirectory() as directory:
            trainer.accelerator.project_configuration.project_dir = directory
            trainer.accelerator.project_configuration.automatic_checkpoint_naming = True

            def save_and_interrupt(*args, **kwargs):
                Trainer.save(trainer, *args, **kwargs)
                raise InterruptAfterCheckpoint()

            trainer.save = save_and_interrupt
            with self.assertRaises(InterruptAfterCheckpoint):
                trainer.train_step(0)
            checkpoint = Path(directory) / "checkpoints" / "checkpoint_0"
            saved = torch.load(checkpoint / "sampler_rank_0.pt", weights_only=True)
            self.assertEqual(first_optimizer.update_count, 3)
            self.assertEqual(saved["optimizer_step"], 3)
            self.assertEqual(saved["sampler"]["progress"]["consumed"], 6)
            self.assertEqual(saved["sampler"]["progress"]["active"]["offset"], 6)
            self.assertEqual([clip for _, clip in first_records], list(range(6)))
            self.assertEqual(trainer.scheduler.scheduler.last_epoch, 3)
            checkpoint_lr = trainer.scheduler.get_last_lr()

            restored, second_optimizer, restored_sampler, restored_loader, replay_records = make_streaming_trainer()
            restored.accelerator.project_configuration.project_dir = directory
            restored.accelerator.project_configuration.automatic_checkpoint_naming = True
            restored.save_interval = 10_000  # save the second checkpoint at the end only
            restored.save = MethodType(Trainer.save, restored)
            restored.load(checkpoint, model_only=False)
            self.assertEqual(restored.exp_tracker.optimizer_step, 3)
            self.assertEqual(restored.scheduler.get_last_lr(), checkpoint_lr)
            self.assertEqual(restored.scheduler.scheduler.last_epoch, 3)
            self.assertEqual(len(restored_loader), 26)
            self.assertEqual(restored_sampler.state_dict()["progress"]["active"]["offset"], 0)
            restored.model.reset_episode_state.assert_called_once_with()

            restored.run()
            self.assertEqual(second_optimizer.update_count, 13)
            self.assertEqual(first_optimizer.update_count + second_optimizer.update_count, 16)
            self.assertEqual(restored.exp_tracker.optimizer_step, 16)
            self.assertEqual(restored.exp_tracker.loader_step, 32)
            self.assertEqual(restored.scheduler.scheduler.last_epoch, 16)
            self.assertAlmostEqual(restored.scheduler.get_last_lr()[0], 0.001)
            self.assertEqual(len(replay_records), 26)
            self.assertEqual(replay_records[:6], first_records)
            self.assertEqual(restored_sampler.get_progress_metrics()["replayed_items"], 6)
            self.assertEqual(restored_sampler.get_progress_metrics()["consumed_items"], 32)
            self.assertEqual(len(restored_loader), 0)
            for previous, current in zip(replay_records, replay_records[1:]):
                if current[0] == previous[0]:
                    self.assertEqual(current[1], previous[1] + 1)
                else:
                    self.assertEqual(current[1], 0)

            final_state = torch.load(
                Path(directory) / "checkpoints" / "checkpoint_1" / "sampler_rank_0.pt",
                weights_only=True,
            )
            self.assertEqual(final_state["optimizer_step"], 16)
            self.assertEqual(final_state["sampler"]["progress"]["consumed"], 32)
            self.assertEqual(final_state["sampler"]["progress"]["replayed_items"], 6)


if __name__ == "__main__":
    unittest.main()
