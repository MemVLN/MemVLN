import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts.qwen25_generation_smoke import (
    build_messages,
    load_smoke_checkpoint,
    resolve_checkpoint_path,
    synthetic_image,
)
from model.qwen.factory import MEMORY_TOKEN


class Qwen25GenerationSmokeHarnessTests(unittest.TestCase):
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
        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)


if __name__ == "__main__":
    unittest.main()
