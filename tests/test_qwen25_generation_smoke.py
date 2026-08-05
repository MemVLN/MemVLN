import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

import scripts.qwen25_generation_smoke as generation_smoke
from scripts.qwen25_generation_smoke import (
    build_messages,
    load_smoke_checkpoint,
    resolve_checkpoint_path,
    synthetic_image,
)
from model.qwen.factory import MEMORY_TOKEN


class Qwen25GenerationSmokeHarnessTests(unittest.TestCase):
    def test_agent_construction_initializes_accelerate_state_first(self):
        calls = mock.Mock()
        partial_state = mock.Mock(return_value=mock.sentinel.state)
        leo_agent = mock.Mock(return_value=mock.sentinel.agent)
        calls.attach_mock(partial_state, "partial_state")
        calls.attach_mock(leo_agent, "leo_agent")

        with (
            mock.patch.object(generation_smoke, "PartialState", partial_state),
            mock.patch.object(generation_smoke, "LeoAgent", leo_agent),
        ):
            agent = generation_smoke.build_agent(mock.sentinel.cfg, cpu=True)

        self.assertIs(agent, mock.sentinel.agent)
        self.assertEqual(
            calls.mock_calls,
            [
                mock.call.partial_state(cpu=True),
                mock.call.leo_agent(mock.sentinel.cfg),
            ],
        )

    def test_synthetic_image_matches_control_resolution(self):
        image = synthetic_image(1)

        self.assertEqual(image.size, (640, 480))
        self.assertEqual(image.mode, "RGB")

    def test_full_prompt_contains_one_expandable_memory_marker(self):
        messages = build_messages(synthetic_image(0), use_memory=True)
        content = messages[0]["content"]

        self.assertEqual(content[0]["text"].count(MEMORY_TOKEN), 1)
        self.assertEqual(
            sum(item.get("type") == "image" for item in content),
            1,
        )

    def test_baseline_prompt_has_no_memory_marker(self):
        messages = build_messages(synthetic_image(0), use_memory=False)

        self.assertFalse(
            any(MEMORY_TOKEN in item.get("text", "") for item in messages[0]["content"])
        )

    def test_checkpoint_directory_loads_without_unknown_keys(self):
        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            torch.save(source.state_dict(), checkpoint)

            self.assertEqual(resolve_checkpoint_path(Path(directory)), checkpoint)
            report = load_smoke_checkpoint(target, Path(directory))

        self.assertEqual(report["status"], "loaded")
        self.assertEqual(report["unexpected_keys"], [])
        self.assertEqual(report["missing_trainable_keys"], [])
        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)

    def test_checkpoint_rejects_missing_trainable_keys(self):
        target = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            torch.save({"weight": target.weight.detach().clone()}, checkpoint)

            with self.assertRaisesRegex(
                RuntimeError, "missing trainable endpoint parameters"
            ):
                load_smoke_checkpoint(target, checkpoint)

    def test_checkpoint_allows_intentionally_excluded_frozen_keys(self):
        target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        target[1].requires_grad_(False)
        trainable_state = {
            key: value.detach().clone()
            for key, value in target.state_dict().items()
            if key.startswith("0.")
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            torch.save(trainable_state, checkpoint)

            report = load_smoke_checkpoint(target, checkpoint)

        self.assertEqual(report["missing_trainable_keys"], [])
        self.assertEqual(report["missing_frozen_key_count"], 2)

    def test_full_reset_requires_memory_and_vggt_episode_state(self):
        full = SimpleNamespace(use_mem=True, use_vggt=True)
        valid = {
            "memory_episodes": 1,
            "vggt_past_key_values": 1,
        }
        generation_smoke.validate_episode_reset(full, valid)

        with self.assertRaisesRegex(RuntimeError, "Memory bank"):
            generation_smoke.validate_episode_reset(
                full,
                {**valid, "memory_episodes": 0},
            )
        with self.assertRaisesRegex(RuntimeError, "StreamVGGT cache"):
            generation_smoke.validate_episode_reset(
                full,
                {**valid, "vggt_past_key_values": 0},
            )

    def test_baseline_reset_accepts_empty_state(self):
        baseline = SimpleNamespace(use_mem=False, use_vggt=False)

        generation_smoke.validate_episode_reset(baseline, {})


if __name__ == "__main__":
    unittest.main()
