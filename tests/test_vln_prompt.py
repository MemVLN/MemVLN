import unittest
from pathlib import Path

from common.vln_prompt import (
    MEMORY_PROMPT,
    NAVIGATION_PROMPT,
    TRAJECTORY_SUMMARY_PROMPT,
    build_navigation_user_message,
    build_trajectory_summary_user_message,
    resolve_memory_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


class VLNPromptTest(unittest.TestCase):
    def test_memory_prompt_enabled(self):
        self.assertEqual(resolve_memory_prompt(True), MEMORY_PROMPT)

    def test_memory_prompt_disabled_is_empty_text(self):
        prompt = resolve_memory_prompt(False)

        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "")
        self.assertNotIn("False", prompt)

    def test_single_observation_navigation_message_is_exact(self):
        message, input_txt = build_navigation_user_message(
            MEMORY_PROMPT,
            ["current.jpg"],
            "Find the door.",
        )

        self.assertEqual(
            input_txt,
            "You are a navigation robot to perform the instruction: Find the "
            "door. What is your next action?",
        )
        self.assertEqual(
            message,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MEMORY_PROMPT},
                    {"type": "text", "text": "Current observation: "},
                    {"type": "image", "image": "current.jpg"},
                    {"type": "text", "text": input_txt},
                ],
            },
        )

    def test_history_current_order_and_action_text_are_exact(self):
        message, input_txt = build_navigation_user_message(
            "",
            ["history-0.jpg", "history-1.jpg", "current.jpg"],
            "Find the door.",
            "Your previous actions are: move forward. ",
        )

        self.assertEqual(
            input_txt,
            "You are a navigation robot to perform the instruction: Find the "
            "door. Your previous actions are: move forward. What is your next "
            "action?",
        )
        self.assertEqual(
            message["content"],
            [
                {"type": "text", "text": ""},
                {"type": "text", "text": "Historical observation: "},
                {
                    "type": "video",
                    "video": ["history-0.jpg", "history-1.jpg"],
                },
                {"type": "text", "text": "Current observation: "},
                {"type": "image", "image": "current.jpg"},
                {"type": "text", "text": input_txt},
            ],
        )

    def test_trajectory_summary_message_is_exact(self):
        message, input_txt = build_trajectory_summary_user_message(
            MEMORY_PROMPT,
            ["history.jpg", "current.jpg"],
        )

        self.assertEqual(input_txt, TRAJECTORY_SUMMARY_PROMPT)
        self.assertEqual(
            [item["type"] for item in message["content"]],
            ["text", "text", "video", "text", "image", "text"],
        )
        self.assertEqual(message["content"][-1]["text"], input_txt)

    def test_disabled_memory_is_safe_in_canonical_message(self):
        memory_prompt = resolve_memory_prompt(False)
        message, _ = build_navigation_user_message(
            memory_prompt,
            ["current.jpg"],
            "Find the door.",
        )

        self.assertEqual(message["content"][0], {"type": "text", "text": ""})
        self.assertTrue(
            all(
                isinstance(item["text"], str) and "False" not in item["text"]
                for item in message["content"]
                if item["type"] == "text"
            )
        )

    def test_empty_observation_list_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            build_navigation_user_message("", [], "Find the door.")

    def test_non_boolean_config_fails_fast(self):
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            resolve_memory_prompt("false")

    def test_all_vln_input_paths_share_the_canonical_builder(self):
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
                self.assertIn("build_navigation_user_message(", source)
                self.assertNotIn(
                    "self.memory_prompt = cfg.data.vln.memory_prompt",
                    source,
                )

    def test_prompt_constants_do_not_contain_layout_placeholders(self):
        layout_placeholders = (
            "{memory_" + "prompt}",
            "{visual_" + "prompt}",
        )
        for placeholder in layout_placeholders:
            self.assertNotIn(placeholder, NAVIGATION_PROMPT)
            self.assertNotIn(placeholder, TRAJECTORY_SUMMARY_PROMPT)


if __name__ == "__main__":
    unittest.main()
