import ast
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from evaluator.vlnce_actions import STOP_ACTION, parse_navigation_action
from split_eval.checkpoint_utils import require_checkpoint_file
from split_eval.merge_results import merge_results


def result_record(scene_id, episode_id, *, ndtw=None):
    record = {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "steps": 10,
        "success": 1.0,
        "oracle_success": 1.0,
        "spl": 0.5,
        "distance_to_goal": 2.0,
    }
    if ndtw is not None:
        record["ndtw"] = ndtw
    return record


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class NavigationActionSafetyTests(unittest.TestCase):
    def test_merged_action_parser_requires_a_magnitude(self):
        self.assertEqual(parse_navigation_action("stop"), (STOP_ACTION, None))
        self.assertEqual(
            parse_navigation_action("move forward 50 cm"),
            (1, 50.0),
        )
        self.assertEqual(
            parse_navigation_action("turn left -15 degrees"),
            (2, -15.0),
        )
        self.assertEqual(parse_navigation_action("move forward"), (None, None))

    def test_unmerged_parser_accepts_direction_without_magnitude(self):
        self.assertEqual(
            parse_navigation_action(
                "move forward", require_magnitude=False
            ),
            (1, None),
        )
        self.assertEqual(
            parse_navigation_action("turn right", require_magnitude=False),
            (3, None),
        )
        self.assertEqual(
            parse_navigation_action("unknown", require_magnitude=False),
            (None, None),
        )
        self.assertEqual(STOP_ACTION, 0)

    def test_split_evaluation_requires_an_exact_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.touch()

            self.assertEqual(require_checkpoint_file(checkpoint), checkpoint)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "requires an exact checkpoint file",
            ):
                require_checkpoint_file(root)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "requires an exact checkpoint file",
            ):
                require_checkpoint_file(root / "missing.safetensors")

    def test_both_evaluators_use_stop_instead_of_random_fallback(self):
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "evaluator/vlnce_eval.py",
            "evaluator/vlnce_eval_split.py",
        ):
            source = (repo_root / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("random.randint", source)
            self.assertIn("STOP_ACTION", source)
            self.assertIn("parse_navigation_action", source)


class SplitResultMergeSafetyTests(unittest.TestCase):
    def test_split_evaluator_delegates_to_validated_merge_loader(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "evaluator/vlnce_eval_split.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "merge_and_record"
        )
        method_source = ast.get_source_segment(source, method)

        self.assertIn("discover_split_files", method_source)
        self.assertIn("load_split_records", method_source)
        self.assertIn("validate_expected_episode_count", method_source)
        self.assertNotIn("except", method_source)

    def test_valid_splits_are_merged_after_count_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            first = result_record("scene-a", "episode-0", ndtw=0.4)
            second = result_record("scene-b", "episode-1", ndtw=0.8)
            second.update(
                steps=20,
                success=0.0,
                oracle_success=1.0,
                spl=0.25,
                distance_to_goal=4.0,
            )
            write_jsonl(target_dir / "results_split_0.json", [first])
            write_jsonl(target_dir / "results_split_1.json", [second])

            merged = merge_results(
                target_dir,
                expected_splits=2,
                expected_episodes=2,
            )
            persisted = json.loads(
                (target_dir / "final_merged.json").read_text(encoding="utf-8")
            )

        self.assertEqual(merged["count"], 2)
        self.assertEqual(merged["success"], 0.5)
        self.assertEqual(merged["steps"], 15.0)
        self.assertAlmostEqual(merged["ndtw"], 0.6)
        self.assertEqual(persisted, merged)
        self.assertTrue(
            all(
                math.isfinite(value)
                for key, value in merged.items()
                if key != "count"
            )
        )

    def test_no_split_files_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No results_split"):
                merge_results(directory)

    def test_expected_split_indices_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            write_jsonl(
                target_dir / "results_split_0.json",
                [result_record("scene-a", "episode-0")],
            )
            write_jsonl(
                target_dir / "results_split_2.json",
                [result_record("scene-b", "episode-1")],
            )

            with self.assertRaisesRegex(ValueError, "Split set mismatch"):
                merge_results(target_dir, expected_splits=2)

    def test_expected_episode_count_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            write_jsonl(
                target_dir / "results_split_0.json",
                [result_record("scene-a", "episode-0")],
            )

            with self.assertRaisesRegex(ValueError, "Episode count mismatch"):
                merge_results(target_dir, expected_episodes=2)

    def test_invalid_json_fails_with_file_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            path = target_dir / "results_split_0.json"
            path.write_text("{not-json}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, r"results_split_0\.json:1: invalid JSON"
            ):
                merge_results(target_dir)

    def test_missing_required_field_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            record = result_record("scene-a", "episode-0")
            del record["success"]
            write_jsonl(target_dir / "results_split_0.json", [record])

            with self.assertRaisesRegex(
                ValueError, "missing required fields.*success"
            ):
                merge_results(target_dir)

    def test_non_finite_values_are_skipped_independently_per_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            first = result_record("scene-a", "episode-0", ndtw=0.4)
            first.update(success=True, spl=float("nan"))
            second = result_record("scene-b", "episode-1", ndtw=0.8)
            second.update(
                steps=20,
                success=False,
                oracle_success=float("inf"),
                spl=0.25,
                distance_to_goal=float("-inf"),
            )
            third = result_record("scene-c", "episode-2", ndtw=float("inf"))
            third.update(
                steps=30,
                success=float("nan"),
                oracle_success=False,
                spl=0.75,
                distance_to_goal=6.0,
            )
            write_jsonl(target_dir / "results_split_0.json", [first, second])
            write_jsonl(target_dir / "results_split_1.json", [third])

            merged = merge_results(
                target_dir, expected_splits=2, expected_episodes=3
            )
            persisted = json.loads(
                (target_dir / "final_merged.json").read_text(encoding="utf-8")
            )

        self.assertEqual(merged["count"], 3)
        self.assertEqual(merged["success"], 0.5)
        self.assertEqual(merged["oracle_success"], 0.5)
        self.assertEqual(merged["spl"], 0.5)
        self.assertEqual(merged["distance_to_goal"], 4.0)
        self.assertEqual(merged["steps"], 20.0)
        self.assertAlmostEqual(merged["ndtw"], 0.6)
        self.assertEqual(persisted, merged)

    def test_metric_with_no_finite_values_is_written_as_null(self):
        fields = ("success", "oracle_success", "spl", "distance_to_goal", "ndtw")
        non_finite_values = (float("nan"), float("inf"), float("-inf"))
        for field in fields:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    target_dir = Path(directory)
                    records = []
                    for index, value in enumerate(non_finite_values):
                        record = result_record("scene-a", str(index), ndtw=0.5)
                        record[field] = value
                        records.append(record)
                    write_jsonl(target_dir / "results_split_0.json", records)

                    merged = merge_results(target_dir, expected_episodes=3)
                    output = (target_dir / "final_merged.json").read_text(
                        encoding="utf-8"
                    )

                self.assertIsNone(merged[field])
                self.assertEqual(merged["count"], 3)
                self.assertEqual(merged["steps"], 10.0)
                self.assertIn(f'"{field}": null', output)
                self.assertNotIn("NaN", output)
                self.assertNotIn("Infinity", output)

    def test_non_numeric_metric_still_fails_with_file_and_line(self):
        for value in ("nan", None, False):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory:
                    target_dir = Path(directory)
                    record = result_record("scene-a", "episode-0")
                    record["spl"] = value
                    write_jsonl(target_dir / "results_split_0.json", [record])

                    with self.assertRaisesRegex(
                        ValueError,
                        r"results_split_0\.json:1: field 'spl' must be numeric",
                    ):
                        merge_results(target_dir)

    def test_invalid_steps_still_fail(self):
        for value in (float("nan"), float("inf"), float("-inf"), 1.5, True, -1):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory:
                    target_dir = Path(directory)
                    record = result_record("scene-a", "episode-0")
                    record["steps"] = value
                    write_jsonl(target_dir / "results_split_0.json", [record])

                    with self.assertRaisesRegex(
                        ValueError, "field 'steps' must be a non-negative integer"
                    ):
                        merge_results(target_dir)

    def test_duplicate_episode_across_splits_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            duplicate = result_record("scene-a", "episode-0")
            write_jsonl(target_dir / "results_split_0.json", [duplicate])
            write_jsonl(target_dir / "results_split_1.json", [duplicate])

            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                merge_results(target_dir)

    def test_partially_missing_optional_metric_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            write_jsonl(
                target_dir / "results_split_0.json",
                [result_record("scene-a", "episode-0", ndtw=0.5)],
            )
            write_jsonl(
                target_dir / "results_split_1.json",
                [result_record("scene-b", "episode-1")],
            )

            with self.assertRaisesRegex(
                ValueError, "Optional metric 'ndtw' is present for 1/2"
            ):
                merge_results(target_dir)

    def test_empty_split_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            (target_dir / "results_split_0.json").write_text(
                "", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "split result file is empty"
            ):
                merge_results(target_dir)


class SplitLauncherSafetyTests(unittest.TestCase):
    def test_any_failed_shard_makes_launcher_fail(self):
        repo_root = Path(__file__).resolve().parents[1]
        launcher = repo_root / "split_eval/launch_eval_split.sh"
        with tempfile.TemporaryDirectory() as directory:
            fake_bin = Path(directory) / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            fake_python.write_text(
                """#!/bin/sh
previous=""
split_id=""
for argument in "$@"; do
    if [ "$previous" = "--split-id" ]; then
        split_id="$argument"
    fi
    previous="$argument"
done
if [ "$split_id" = "1" ]; then
    exit 7
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_sleep = fake_bin / "sleep"
            fake_sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_sleep.chmod(0o755)
            fake_checkpoint = Path(directory) / "model.safetensors"
            fake_checkpoint.touch()
            environment = os.environ.copy()
            environment["PATH"] = (
                str(fake_bin) + os.pathsep + environment["PATH"]
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--gpus",
                    "3",
                    "--model",
                    str(fake_checkpoint),
                ],
                cwd=repo_root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("Error: Split 1 failed.", completed.stdout)
        self.assertIn("results are incomplete", completed.stdout)
        self.assertNotIn("All splits finished", completed.stdout)

    def test_launcher_rejects_invalid_gpu_count_before_starting_shards(self):
        repo_root = Path(__file__).resolve().parents[1]
        launcher = repo_root / "split_eval/launch_eval_split.sh"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.touch()
            completed = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--gpus",
                    "0",
                    "--model",
                    str(checkpoint),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("GPU count must be a positive integer", completed.stdout)


if __name__ == "__main__":
    unittest.main()
