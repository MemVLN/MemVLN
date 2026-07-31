import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from data.datasets import VLNBase, VLNTrajSumBase
from model.leo_agent import LeoAgent


ROOT = Path(__file__).resolve().parents[1]


def make_dataset(dataset_cls, split):
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.split = split
    dataset.domain = "MP3D"
    dataset.realworld_resize_size = None
    dataset.prompt_format = "v2"
    dataset.memory_prompt = ""
    dataset.use_vggt = False
    return dataset


class DatasetSFTContractTests(unittest.TestCase):
    def test_navigation_train_adds_assistant_target_only_for_train(self):
        metadata = {
            "scene_id": "scene-1",
            "episode_id": "episode-1",
            "clip_id": 0,
            "video_path": ["frame.jpg"],
            "instruction": "Walk forward.",
            "past_actions": [],
            "pred_actions": ["move forward"],
            "current_odom": [0.0, 0.0, 0.0],
        }

        for split, expected_roles in (
            ("train", ["user", "assistant"]),
            ("val", ["user"]),
        ):
            with self.subTest(split=split):
                dataset = make_dataset(VLNBase, split)
                dataset.meta_data = [metadata]
                sample = dataset[0]
                self.assertEqual(
                    [message["role"] for message in sample["messages"]],
                    expected_roles,
                )
                if split == "train":
                    self.assertEqual(
                        sample["messages"][-1]["content"][0]["text"],
                        sample["output_gt"],
                    )
                self.assertNotIn("qwen_feat", sample)
                self.assertNotIn("vggt_feat", sample)

    def test_trajectory_summary_adds_assistant_target_only_for_train(self):
        metadata = {
            "scene_id": "scene-1",
            "episode_id": "episode-1",
            "clip_id": 0,
            "video_path": ["frame.jpg"],
            "instruction": "Walk forward.",
            "current_odom": [0.0, 0.0, 0.0],
        }

        for split, expected_roles in (
            ("train", ["user", "assistant"]),
            ("val", ["user"]),
        ):
            with self.subTest(split=split):
                dataset = make_dataset(VLNTrajSumBase, split)
                dataset.meta_data = [metadata]
                sample = dataset[0]
                self.assertEqual(
                    [message["role"] for message in sample["messages"]],
                    expected_roles,
                )
                if split == "train":
                    self.assertEqual(
                        sample["messages"][-1]["content"][0]["text"],
                        sample["output_gt"],
                    )
                self.assertNotIn("qwen_feat", sample)
                self.assertNotIn("vggt_feat", sample)


class LeoAgentSFTContractTests(unittest.TestCase):
    def test_forward_dispatches_directly_to_sft(self):
        expected = {"loss": torch.tensor(1.0)}
        agent = SimpleNamespace(sft_forward=Mock(return_value=expected))
        data_dict = {"input_ids": torch.tensor([[1, 2]])}

        result = LeoAgent.forward(agent, data_dict)

        self.assertIs(result, expected)
        agent.sft_forward.assert_called_once_with(data_dict)

    def test_generate_returns_text_without_full_prompt_completion_tokens(self):
        generated_ids = torch.tensor(
            [
                [11, 12, 21, 22],
                [11, 12, 31, 32],
            ]
        )
        generate = Mock(return_value=generated_ids)
        processor = SimpleNamespace(
            batch_decode=Mock(return_value=[" move forward ", " stop "])
        )
        agent = SimpleNamespace(
            model=SimpleNamespace(generate=generate),
            processor=processor,
            max_out_len=8,
            model_input_keys=["input_ids"],
        )
        data_dict = {"input_ids": torch.tensor([[11, 12]])}

        result = LeoAgent.generate(
            agent,
            data_dict,
            num_beams=2,
            num_return_sequences=2,
        )

        self.assertEqual(result["output_txt"], ["move forward", "stop"])
        self.assertNotIn("prompt_completion_ids", result)
        call_kwargs = generate.call_args.kwargs
        self.assertEqual(call_kwargs["num_beams"], 2)
        self.assertEqual(call_kwargs["num_return_sequences"], 2)

    def test_weighted_ce_path_remains_distinct_from_unweighted_ce(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
                [[2.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
            ]
        )
        targets = torch.tensor(
            [
                [-100, 0, 1],
                [-100, 1, 1],
            ]
        )
        weights = torch.tensor([1.0, 3.0])

        per_token_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 2),
            targets[:, 1:].reshape(-1),
            reduction="none",
        ).view(2, 2)
        per_sample_loss = per_token_loss.mean(-1)
        expected_unweighted = per_token_loss.mean()
        expected_weighted = (per_sample_loss * weights).sum() / weights.sum()

        unweighted = LeoAgent.compute_ce_loss(None, logits, targets)
        weighted = LeoAgent.compute_ce_loss(
            None,
            logits,
            targets,
            sample_weight=weights,
        )

        torch.testing.assert_close(unweighted, expected_unweighted)
        torch.testing.assert_close(weighted, expected_weighted)
        self.assertFalse(torch.isclose(unweighted, weighted))


class LegacyRemovalContractTests(unittest.TestCase):
    def test_grpo_and_vsibench_surface_is_removed(self):
        self.assertFalse((ROOT / "data" / "vsibench.py").exists())

        checked_files = [
            ROOT / "data" / "data_utils.py",
            ROOT / "data" / "datasets.py",
            ROOT / "trainer" / "leo_trainer.py",
            ROOT / "model" / "leo_agent.py",
        ]
        forbidden = (
            "use_grpo",
            "one_step_grpo",
            "grpo_forward",
            "expand_inputs_for_grpo",
            "calculate_grpo_rewards",
            "prompt_completion_ids",
            "use_dpo",
        )
        combined_source = "\n".join(
            path.read_text(encoding="utf-8") for path in checked_files
        )
        for symbol in forbidden:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, combined_source)

    def test_pandas_is_removed_but_trl_is_retained(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        requirement_names = {
            line.split("=", 1)[0].split(">", 1)[0].strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertNotIn("pandas", requirement_names)
        self.assertIn("trl", requirement_names)


if __name__ == "__main__":
    unittest.main()
