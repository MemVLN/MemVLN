import unittest
from pathlib import Path

from common.vln_prompt import MEMORY_PROMPT, resolve_memory_prompt


ROOT = Path(__file__).resolve().parents[1]


class VLNPromptTest(unittest.TestCase):
    def test_memory_prompt_enabled(self):
        self.assertEqual(resolve_memory_prompt(True), MEMORY_PROMPT)

    def test_memory_prompt_disabled_is_empty_text(self):
        prompt = resolve_memory_prompt(False)

        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "")
        self.assertNotIn("False", prompt)

    def test_disabled_prompt_is_safe_for_v1_and_v2(self):
        memory_prompt = resolve_memory_prompt(False)
        prompt_v1 = (
            "Instruction. {memory_prompt}What is your next action?"
        ).format(memory_prompt=memory_prompt)
        prompt_v2_content = [
            {"type": "text", "text": memory_prompt},
            {"type": "text", "text": "Current observation: "},
        ]

        self.assertNotIn("False", prompt_v1)
        self.assertTrue(
            all(
                isinstance(item["text"], str) and "False" not in item["text"]
                for item in prompt_v2_content
            )
        )

    def test_non_boolean_config_fails_fast(self):
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            resolve_memory_prompt("false")

    def test_all_vln_input_paths_share_the_resolver(self):
        for relative_path in (
            "data/datasets.py",
            "evaluator/vlnce_eval.py",
            "evaluator/vlnce_eval_split.py",
            "realworld/agent.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                self.assertIn(
                    "resolve_memory_prompt(cfg.data.vln.memory_prompt)",
                    source,
                )
                self.assertNotIn(
                    "self.memory_prompt = cfg.data.vln.memory_prompt",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
