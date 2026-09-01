import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from realworld.agent import VLNEvaluator


ROOT = Path(__file__).resolve().parents[1]


def solid_image(value):
    return Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8))


class FakeVideoProcessor:
    def vggt_preprocess(self, videos, **kwargs):
        return [video.float() / 255 for video in videos]


class FakeProcessor:
    def __init__(self):
        self.video_processor = FakeVideoProcessor()
        self.image_processor = SimpleNamespace(patch_size=14)

    def apply_chat_template(self, messages, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "image_grid_thw": torch.tensor([[1, 2, 3]]),
        }


class FakeModel:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []
        self.reset_count = 0

    def generate(self, data_dict, **kwargs):
        self.calls.append((data_dict, kwargs))
        return {"output_txt": [self.outputs.popleft()]}

    def reset_episode_state(self):
        self.reset_count += 1
        return {"reset": True}


def make_evaluator(outputs):
    evaluator = VLNEvaluator.__new__(VLNEvaluator)
    evaluator.device = torch.device("cpu")
    evaluator.model = FakeModel(outputs)
    evaluator.processor = FakeProcessor()
    evaluator.use_vggt = True
    evaluator.use_mem = False
    evaluator.dense_vggt_input = True
    evaluator.obs_hist_sample = "nearest"
    evaluator.max_num_frames = 16
    evaluator.merge_action = True
    evaluator.max_act_hist = 0
    evaluator.memory_prompt = ""
    evaluator.instruction = "Walk to the door."
    evaluator.reset_memory()
    return evaluator


class RealWorldStreamVGGTTests(unittest.TestCase):
    def test_pending_steps_fill_dense_stream_and_reset_starts_new_episode(self):
        evaluator = make_evaluator([
            "move forward 75 cm",
            "stop",
            "stop",
        ])

        first = evaluator.step_with_metadata(solid_image(0), "episode-a")
        self.assertEqual(first[:3], (1, 2, True))
        self.assertEqual(first[4], [1, 1, 1])
        self.assertEqual(len(evaluator.model.calls), 1)
        first_call, first_generation_args = evaluator.model.calls[0]
        self.assertEqual(first_call["episode_id"], ["episode-a"])
        self.assertEqual(first_call["clip_id"].tolist(), [0])
        self.assertEqual(first_call["vggt_input"][0].shape[0], 1)
        self.assertEqual(first_generation_args["num_beams"], 1)
        self.assertEqual(first_generation_args["num_return_sequences"], 1)

        second = evaluator.step_with_metadata(solid_image(1), "episode-a")
        third = evaluator.step_with_metadata(solid_image(2), "episode-a")
        self.assertEqual(second[:3], (1, 1, False))
        self.assertEqual(third[:3], (1, 0, False))
        self.assertEqual(len(evaluator.model.calls), 1)

        fourth = evaluator.step_with_metadata(solid_image(3), "episode-a")
        self.assertEqual(fourth[0], 0)
        self.assertEqual(len(evaluator.model.calls), 2)
        fourth_call, _ = evaluator.model.calls[1]
        self.assertEqual(fourth_call["clip_id"].tolist(), [3])
        dense_values = fourth_call["vggt_input"][0].mean(dim=(1, 2, 3))
        torch.testing.assert_close(
            dense_values,
            torch.tensor([1, 2, 3], dtype=dense_values.dtype) / 255,
        )

        evaluator.reset_memory()
        after_reset = evaluator.step_with_metadata(solid_image(4), "episode-b")
        self.assertEqual(after_reset[0], 0)
        reset_call, _ = evaluator.model.calls[2]
        self.assertEqual(reset_call["episode_id"], ["episode-b"])
        self.assertEqual(reset_call["clip_id"].tolist(), [0])
        self.assertEqual(reset_call["vggt_input"][0].shape[0], 1)
        self.assertEqual(evaluator.model.reset_count, 2)

    def test_repeat_count_is_clamped_to_project_limit(self):
        evaluator = make_evaluator(["move forward 200 cm"])

        result = evaluator.step_with_metadata(solid_image(1), "episode-a")

        self.assertEqual(result[4], [1, 1, 1])
        self.assertEqual(result[1], 2)

    def test_invalid_model_action_uses_fail_safe_stop(self):
        evaluator = make_evaluator(["dance in place"])

        result = evaluator.step_with_metadata(solid_image(1), "episode-a")

        self.assertEqual(result[0], 0)
        self.assertEqual(result[4], [0])

    def test_http_and_clients_expose_session_safety_contract(self):
        server_source = (
            ROOT / "realworld" / "http_realworld_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("with evaluation_lock:", server_source)
        self.assertIn("uuid.uuid4().hex", server_source)
        self.assertIn("}), 409", server_source)

        for filename in ("g1_vln_client.py", "go2_vln_client.py"):
            source = (ROOT / "realworld" / filename).read_text(encoding="utf-8")
            self.assertIn("response.raise_for_status()", source)


if __name__ == "__main__":
    unittest.main()
