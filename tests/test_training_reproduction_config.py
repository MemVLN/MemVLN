import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from model.memvln_agent import MemVLNAgent, resolve_memory_param_dtype


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    memory_token_id = 151669

    def __len__(self):
        return 151672


class _TinyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.register_buffer("floating_buffer", torch.ones(2))
        self.register_buffer("integer_buffer", torch.ones(2, dtype=torch.long))


def _memory_cfg(param_dtype="bf16"):
    mem_module = {
        "memory_token_num": 300,
        "memory_merge_vggt_feat": False,
        "visual_obs_mask_ratio_nav": 0.0,
        "visual_obs_mask_ratio_aux": 1.0,
    }
    if param_dtype is not ...:
        mem_module["param_dtype"] = param_dtype
    return OmegaConf.create(
        {
            "llm": {
                "torch_dtype": "bf16",
                "mem_module": mem_module,
            },
            "inference": {},
        }
    )


class MemoryParameterDtypeTests(unittest.TestCase):
    def test_default_config_matches_april_training_and_current_runtime_policy(self):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(ROOT / "configs"),
        ):
            cfg = compose(config_name="default")

        self.assertEqual(cfg.training.gradient_accumulation_steps, 1)
        self.assertEqual(cfg.training.checkpoint_total_limit, 5)
        self.assertEqual(cfg.llm.mem_module.param_dtype, "bf16")
        self.assertIs(cfg.data.vln.aug_stop_steps, False)
        self.assertEqual(
            cfg.note,
            "sft_r2r_rxr_memory_16frame_w_traj_sum_"
            "streamvggt_cache_feat",
        )
        self.assertEqual(
            list(cfg.task.dataset_mix.VLNDatasetMix.mix.vln),
            [
                "mp3d_r2r",
                "mp3d_rxr",
                "traj_sum_mp3d_r2r",
                "traj_sum_mp3d_rxr",
            ],
        )
        self.assertTrue(cfg.inference.cache_grounded_memory)
        self.assertTrue(cfg.inference.skip_memory_attention_weights)
        self.assertTrue(cfg.inference.fast_habitat)

    def test_legacy_config_falls_back_to_llm_dtype(self):
        cfg = _memory_cfg(param_dtype=...)
        self.assertIs(resolve_memory_param_dtype(cfg), torch.bfloat16)

    def test_explicit_memory_dtype_overrides_llm_dtype(self):
        cfg = _memory_cfg(param_dtype="fp32")
        self.assertIs(resolve_memory_param_dtype(cfg), torch.float32)

    def test_invalid_memory_dtype_fails_fast(self):
        cfg = _memory_cfg(param_dtype="fp8")
        with self.assertRaisesRegex(
            ValueError,
            r"llm\.mem_module\.param_dtype.*fp8",
        ):
            resolve_memory_param_dtype(cfg)

    def test_build_casts_only_memory_parameters_and_floating_buffers(self):
        cfg = _memory_cfg(param_dtype="bf16")
        memory = _TinyMemory()
        unrelated = nn.Linear(2, 2)
        inner_model = SimpleNamespace(unrelated=unrelated)
        outer_model = SimpleNamespace(
            model=inner_model,
            resize_token_embeddings=mock.Mock(),
        )
        processor = SimpleNamespace(
            tokenizer=_Tokenizer(),
            __init_memory__=mock.Mock(),
        )
        agent = SimpleNamespace(model=outer_model, processor=processor)

        with (
            mock.patch("model.memvln_agent.ToMeMemoryBank", return_value=memory),
            mock.patch("model.memvln_agent.logger.info"),
        ):
            MemVLNAgent.build_mem_modules(agent, cfg)

        floating_tensors = [
            *memory.parameters(),
            *(
                buffer
                for buffer in memory.buffers()
                if buffer.is_floating_point()
            ),
        ]
        self.assertTrue(floating_tensors)
        self.assertTrue(
            all(tensor.dtype == torch.bfloat16 for tensor in floating_tensors)
        )
        self.assertEqual(memory.integer_buffer.dtype, torch.long)
        self.assertEqual(unrelated.weight.dtype, torch.float32)


class GradientAccumulationRegressionTests(unittest.TestCase):
    def test_unprepared_loader_syncs_the_final_partial_accumulation_window(self):
        trainer_path = ROOT / "trainer" / "memvln_trainer.py"
        tree = ast.parse(
            trainer_path.read_text(encoding="utf-8"),
            filename=str(trainer_path),
        )
        selected_nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "_UnpreparedDataLoaderState"
            )
            or (
                isinstance(node, ast.FunctionDef)
                and node.name == "track_unprepared_dataloader_batch"
            )
        ]
        namespace = {"contextmanager": contextmanager}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=selected_nodes, type_ignores=[])
                ),
                str(trainer_path),
                "exec",
            ),
            namespace,
        )

        accelerator = Accelerator(gradient_accumulation_steps=2)
        model = nn.Linear(1, 1)
        sync_boundaries = []
        for batch_index in range(3):
            with namespace["track_unprepared_dataloader_batch"](
                accelerator,
                enabled=True,
                end_of_dataloader=(batch_index == 2),
            ):
                with accelerator.accumulate(model):
                    sync_boundaries.append(accelerator.sync_gradients)

        self.assertEqual(sync_boundaries, [False, True, True])

    def test_backward_zeroes_gradients_after_the_accumulated_update(self):
        trainer_path = ROOT / "trainer" / "memvln_trainer.py"
        tree = ast.parse(
            trainer_path.read_text(encoding="utf-8"),
            filename=str(trainer_path),
        )
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MemVLNTrainer"
        )
        backward_method = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "backward"
        )
        harness_class = ast.ClassDef(
            name="_BackwardHarness",
            bases=[],
            keywords=[],
            body=[backward_method],
            decorator_list=[],
        )
        namespace = {}
        harness_module = ast.fix_missing_locations(
            ast.Module(body=[harness_class], type_ignores=[])
        )
        exec(
            compile(
                harness_module,
                str(trainer_path),
                "exec",
            ),
            namespace,
        )
        backward_harness = namespace["_BackwardHarness"]

        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()

        accelerator = SimpleNamespace(sync_gradients=False)

        def backward(loss):
            (loss / 2).backward()

        accelerator.backward = backward
        accelerator.clip_grad_norm_ = mock.Mock()

        class SyncAwareOptimizer:
            def __init__(self, parameter):
                self.parameter = parameter

            def step(self):
                if accelerator.sync_gradients:
                    self.parameter.data.add_(self.parameter.grad, alpha=-1.0)

            def zero_grad(self):
                if accelerator.sync_gradients:
                    self.parameter.grad = None

        trainer = backward_harness()
        trainer.model = model
        trainer.accelerator = accelerator
        trainer.optimizer = SyncAwareOptimizer(model.weight)
        trainer.scheduler = SimpleNamespace(step=mock.Mock())
        trainer.grad_norm = None

        first_loss = 0.5 * (model.weight - 1).square().sum()
        trainer.backward(first_loss)
        accelerator.sync_gradients = True
        second_loss = 0.5 * (model.weight - 3).square().sum()
        trainer.backward(second_loss)

        torch.testing.assert_close(
            model.weight.detach(),
            torch.tensor([[2.0]]),
        )
        self.assertIsNone(model.weight.grad)


class LauncherFailurePropagationTests(unittest.TestCase):
    @staticmethod
    def _load_launch_helpers():
        launch_utils_path = ROOT / "common" / "launch_utils.py"
        tree = ast.parse(
            launch_utils_path.read_text(encoding="utf-8"),
            filename=str(launch_utils_path),
        )
        selected_nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id in {
                        "_SUBMITIT_PORT_BASE",
                        "_SUBMITIT_PORT_COUNT",
                    }
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.FunctionDef)
                and node.name == "resolve_submitit_port"
            )
        ]
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=selected_nodes, type_ignores=[])
                ),
                str(launch_utils_path),
                "exec",
            ),
            namespace,
        )
        return launch_utils_path, tree, namespace["resolve_submitit_port"]

    def test_submitit_fallback_port_is_shared_and_requested_port_is_validated(self):
        _, _, resolve_submitit_port = self._load_launch_helpers()

        self.assertEqual(resolve_submitit_port(None, "12345"), 27345)
        self.assertEqual(
            resolve_submitit_port(None, "12345_0"),
            resolve_submitit_port(None, "12345_0"),
        )
        self.assertEqual(resolve_submitit_port(3012, "different-job"), 3012)
        for invalid_port in (-1, 65536):
            with self.subTest(invalid_port=invalid_port):
                with self.assertRaises(ValueError):
                    resolve_submitit_port(invalid_port, "12345")

    def test_submitit_task_rank_is_used_as_accelerate_machine_rank(self):
        launch_utils_path, tree, _ = self._load_launch_helpers()
        launcher_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SubmititLauncher"
        )
        set_gpu_args = next(
            node
            for node in launcher_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_set_gpu_args"
        )
        node_id_assignment = next(
            node
            for node in ast.walk(set_gpu_args)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "node_id"
                for target in node.targets
            )
        )

        self.assertIsInstance(node_id_assignment.value, ast.Attribute)
        self.assertEqual(node_id_assignment.value.attr, "global_rank")
        self.assertFalse(
            any(
                isinstance(node, (ast.Div, ast.FloorDiv))
                for node in ast.walk(node_id_assignment.value)
            ),
            msg=f"machine rank in {launch_utils_path} must not divide task rank",
        )

    def test_every_launcher_checks_child_process_exit_status(self):
        launch_utils_path = ROOT / "common" / "launch_utils.py"
        tree = ast.parse(
            launch_utils_path.read_text(encoding="utf-8"),
            filename=str(launch_utils_path),
        )
        subprocess_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
        ]

        self.assertEqual(len(subprocess_calls), 3)
        for call in subprocess_calls:
            check_keyword = next(
                (
                    keyword
                    for keyword in call.keywords
                    if keyword.arg == "check"
                ),
                None,
            )
            self.assertIsNotNone(check_keyword)
            self.assertIsInstance(check_keyword.value, ast.Constant)
            self.assertIs(check_keyword.value.value, True)


if __name__ == "__main__":
    unittest.main()
