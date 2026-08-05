import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from model.streamvggt.processing import (
    build_online_vggt_input,
    resolve_dense_vggt_input,
)


def solid_image(value):
    return Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8))


def messages_for(history, current):
    content = []
    if history:
        content.append({"type": "video", "video": history})
    content.append({"type": "image", "image": current})
    return [{"role": "user", "content": content}]


class FakeVideoProcessor:
    def vggt_preprocess(self, videos, **kwargs):
        self.videos = videos
        self.kwargs = kwargs
        return [video.float() / 255 for video in videos]


class OnlineVGGTProcessingTests(unittest.TestCase):
    def setUp(self):
        self.video_processor = FakeVideoProcessor()
        self.processor = SimpleNamespace(
            image_processor=SimpleNamespace(patch_size=16),
            video_processor=self.video_processor,
        )

    def build(self, messages, dense=True):
        batch_size = 1 if isinstance(messages[0], dict) else len(messages)
        image_grid_thw = torch.tensor([[1, 2, 3]] * batch_size)
        return build_online_vggt_input(
            self.processor,
            messages,
            image_grid_thw,
            dense_history=dense,
        )

    def test_first_frame_contains_only_current_observation(self):
        output = self.build(messages_for([], solid_image(7)))

        self.assertEqual(len(output), 1)
        self.assertEqual(tuple(output[0].shape), (1, 3, 4, 5))
        self.assertEqual(output[0].dtype, torch.float32)
        self.assertEqual(self.video_processor.kwargs["patch_size"], 14)
        self.assertGreaterEqual(output[0].min().item(), 0.0)
        self.assertLessEqual(output[0].max().item(), 1.0)
        torch.testing.assert_close(
            output[0].mean(),
            torch.tensor(7 / 255, dtype=output[0].dtype),
        )

    def test_dense_history_keeps_last_two_frames_in_temporal_order(self):
        output = self.build(
            messages_for(
                [solid_image(10), solid_image(20), solid_image(30)],
                solid_image(40),
            )
        )

        self.assertEqual(tuple(output[0].shape), (3, 3, 4, 5))
        frame_values = output[0].mean(dim=(1, 2, 3))
        torch.testing.assert_close(
            frame_values,
            torch.tensor([20, 30, 40], dtype=output[0].dtype) / 255,
        )

    def test_dense_history_includes_one_and_two_available_frames(self):
        cases = (
            ([solid_image(10)], [10, 30]),
            ([solid_image(10), solid_image(20)], [10, 20, 30]),
        )
        for history, expected in cases:
            with self.subTest(history_frames=len(history)):
                output = self.build(messages_for(history, solid_image(30)))
                frame_values = output[0].mean(dim=(1, 2, 3))
                torch.testing.assert_close(
                    frame_values,
                    torch.tensor(expected, dtype=output[0].dtype) / 255,
                )

    def test_non_dense_mode_uses_only_current_observation(self):
        output = self.build(
            messages_for([solid_image(10), solid_image(20)], solid_image(30)),
            dense=False,
        )

        self.assertEqual(output[0].shape[0], 1)
        torch.testing.assert_close(
            output[0].mean(),
            torch.tensor(30 / 255, dtype=output[0].dtype),
        )

    def test_batched_inputs_keep_one_group_per_sample(self):
        messages = [
            messages_for([solid_image(1)], solid_image(2)),
            messages_for([solid_image(3)], solid_image(4)),
        ]
        output = self.build(messages)

        self.assertEqual(len(output), 2)
        torch.testing.assert_close(
            output[0].mean(dim=(1, 2, 3)),
            torch.tensor([1, 2], dtype=output[0].dtype) / 255,
        )
        torch.testing.assert_close(
            output[1].mean(dim=(1, 2, 3)),
            torch.tensor([3, 4], dtype=output[1].dtype) / 255,
        )

    def test_fixed_geometry_grid_is_independent_of_qwen_image_grid(self):
        self.processor.vggt_input_grid_hw = (30, 40)

        self.build(messages_for([], solid_image(7)))

        torch.testing.assert_close(
            self.video_processor.kwargs["video_grid_thw"],
            torch.tensor([[1, 30, 40]]),
        )

    def test_missing_current_image_fails_fast(self):
        messages = [{
            "role": "user",
            "content": [{"type": "video", "video": [solid_image(1)]}],
        }]
        with self.assertRaisesRegex(ValueError, "exactly one current image"):
            self.build(messages)

    def test_grid_batch_mismatch_fails_fast(self):
        messages = [
            messages_for([], solid_image(1)),
            messages_for([], solid_image(2)),
        ]
        with self.assertRaisesRegex(ValueError, "batch sizes differ"):
            build_online_vggt_input(
                self.processor,
                messages,
                torch.tensor([[1, 2, 3]]),
            )

    def test_dense_mode_requires_nearest_sampling(self):
        dense_cfg = {
            "merge_action": True,
            "skip_merge_step_obs": False,
            "vggt_cache_feat": True,
            "obs_hist_sample": "nearest",
            "max_num_frames": 16,
        }
        self.assertTrue(resolve_dense_vggt_input(dense_cfg))

        dense_cfg["obs_hist_sample"] = "uniform"
        with self.assertRaisesRegex(ValueError, "requires.*nearest"):
            resolve_dense_vggt_input(dense_cfg)

        dense_cfg["obs_hist_sample"] = "nearest"
        dense_cfg["vggt_cache_feat"] = False
        with self.assertRaisesRegex(ValueError, "vggt_cache_feat=true"):
            resolve_dense_vggt_input(dense_cfg)

        dense_cfg["merge_action"] = False
        self.assertFalse(resolve_dense_vggt_input(dense_cfg))

    def test_dense_mode_requires_room_for_two_pending_frames(self):
        dense_cfg = {
            "merge_action": True,
            "skip_merge_step_obs": False,
            "vggt_cache_feat": True,
            "obs_hist_sample": "nearest",
            "max_num_frames": 2,
        }

        with self.assertRaisesRegex(ValueError, "max_num_frames>=3"):
            resolve_dense_vggt_input(dense_cfg)

    def test_current_image_count_is_validated_per_sample(self):
        malformed_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": solid_image(1)},
                    {"type": "image", "image": solid_image(2)},
                ],
            }],
            [{
                "role": "user",
                "content": [{"type": "text", "text": "no current image"}],
            }],
        ]

        with self.assertRaisesRegex(ValueError, "sample 0 has 2"):
            build_online_vggt_input(
                self.processor,
                malformed_batch,
                torch.tensor([[1, 2, 3], [1, 2, 3]]),
            )


if __name__ == "__main__":
    unittest.main()
