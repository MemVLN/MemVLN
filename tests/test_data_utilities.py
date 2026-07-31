import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from habitat_extensions.generate_obs import (
    CacheTrajectory,
    normalize_action,
    trajectory_directory_name,
    validate_reference_path,
)
from habitat_extensions.rxr_conversion import convert_dataset
from habitat_extensions.stat_vln_action_lengths import (
    _write_csv,
    merge_actions,
)
from split_eval.extract_vlnce_video_parts import (
    load_manifest,
    output_dir_for,
    summary_paths_for,
)


class RxRConversionTests(unittest.TestCase):
    def test_filters_language_and_preserves_metadata(self):
        source = {
            "metadata": {"split": "train"},
            "episodes": [
                {
                    "episode_id": 1,
                    "instruction": {
                        "language": "en-US",
                        "instruction_text": "Walk forward.",
                    },
                },
                {
                    "episode_id": 2,
                    "instruction": {
                        "language": "hi-IN",
                        "instruction_text": "आगे चलें।",
                    },
                },
            ],
        }

        converted, removed = convert_dataset(source, ["en"])

        self.assertEqual(removed, 1)
        self.assertEqual(converted["metadata"], source["metadata"])
        self.assertEqual(
            converted["episodes"][0]["instruction"],
            {
                "instruction_text": "Walk forward.",
                "instruction_tokens": [],
            },
        )
        self.assertEqual(converted["instruction_vocab"]["num_vocab"], 0)


class ObservationGenerationValidationTests(unittest.TestCase):
    def test_validates_trajectory_ids_reference_paths_and_actions(self):
        self.assertEqual(trajectory_directory_name(17), "17")
        self.assertEqual(validate_reference_path([[0, 0, 0]], "17"), [[0, 0, 0]])
        self.assertEqual(normalize_action(3, "17"), 3)

        for unsafe_id in ("../17", "scene/17", ".."):
            with self.subTest(unsafe_id=unsafe_id):
                with self.assertRaises(ValueError):
                    trajectory_directory_name(unsafe_id)

        with self.assertRaises(ValueError):
            validate_reference_path([], "17")
        with self.assertRaises(ValueError):
            normalize_action(4, "17")

    def test_generation_error_propagates_after_environment_cleanup(self):
        episode = SimpleNamespace(
            scene_id="/scenes/example/example.glb",
            trajectory_id=17,
        )
        environment = SimpleNamespace(
            episodes=[episode],
            closed=False,
        )

        def close_environment():
            environment.closed = True

        environment.close = close_environment
        original_error = RuntimeError("synthetic generation failure")

        with tempfile.TemporaryDirectory() as temporary_dir:
            generator = CacheTrajectory(
                config_path="unused.yaml",
                output_path=temporary_dir,
            )
            generator.config_env = Mock(return_value=environment)
            generator._generate_episode = Mock(side_effect=original_error)

            with patch(
                "tqdm.tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    generator.generate()

        self.assertIs(raised.exception, original_error)
        self.assertTrue(environment.closed)


class ActionStatisticsTests(unittest.TestCase):
    def test_merges_actions_with_project_limits(self):
        merged, source_indices = merge_actions(
            [
                "move forward",
                "move forward",
                "move forward",
                "move forward",
                "turn left",
                "turn left",
                "stop",
            ]
        )

        self.assertEqual(
            merged,
            [
                "move forward 75 cm",
                "move forward 25 cm",
                "turn left 30 degree",
                "stop",
            ],
        )
        self.assertEqual(source_indices, [0, 3, 4, 6])

    def test_csv_refuses_implicit_overwrite_and_quotes_values(self):
        records = [
            {
                "dataset": "mp3d_r2r",
                "episode_id": "r2r-1",
                "trajectory_id": "trajectory,one",
                "raw_len": 4,
                "merged_len": 2,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "nested" / "lengths.csv"
            _write_csv(records, output_path)
            self.assertIn('"trajectory,one"', output_path.read_text(encoding="utf-8"))

            with self.assertRaises(FileExistsError):
                _write_csv(records, output_path)

            _write_csv(records, output_path, overwrite=True)


class VideoManifestTests(unittest.TestCase):
    def test_resolves_relative_manifest_paths_and_summary_location(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest = root / "videos.txt"
            manifest.write_text(
                "# comment\nruns/video/example.mp4\n\n",
                encoding="utf-8",
            )

            videos = load_manifest(manifest)
            self.assertEqual(videos, [root / "runs" / "video" / "example.mp4"])

            output_dir = output_dir_for(videos[0], output_root=None)
            self.assertEqual(
                summary_paths_for(videos, output_root=None),
                {output_dir.parent / "split_summary.json"},
            )


if __name__ == "__main__":
    unittest.main()
