"""Exercise the real dataset-mixing methods without loading model dependencies.

Only the registry decorator, Dataset base, and concrete dataset mapping are
replaced. The constructor, selection, stream ordering, and item-access methods
are compiled unchanged from data/datasets.py.
"""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "datasets.py"


def _load_dataset_mix_class(source=None):
    if source is None:
        source = DATASET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(DATASET_PATH))
    dataset_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VLNDatasetMix"
    )
    dataset_class.decorator_list = []
    dataset_class.bases = []
    for node in dataset_class.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "mapping"
            for target in node.targets
        ):
            node.value = ast.Dict(keys=[], values=[])

    namespace = {
        "np": np,
        "logger": Mock(),
        "OmegaConf": SimpleNamespace(
            is_config=lambda value: False,
            to_yaml=lambda value, resolve=True: str(value),
        ),
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[dataset_class], type_ignores=[])
    )
    exec(compile(module, str(DATASET_PATH), "exec"), namespace)
    return namespace["VLNDatasetMix"]


DatasetMix = _load_dataset_mix_class()


class _FixtureDataset:
    def __init__(self, dataset_key, episode_clips):
        self.meta_data = [
            {
                "episode_id": episode_id,
                "clip_id": clip_id,
                "source": f"source-of-{dataset_key}",
                "path": f"/{dataset_key}/{episode_id}/{clip_id}/{index}.jpg",
                "dataset_key": dataset_key,
            }
            for index, (episode_id, clip_id) in enumerate(episode_clips)
        ]

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        # Returning the stored object also detects mutation by the mix wrapper.
        return self.meta_data[index]


def _make_mix(specifications, ratios, *, ratio_unit="episode"):
    datasets = {
        key: _FixtureDataset(key, episode_clips)
        for key, episode_clips in specifications.items()
    }
    DatasetMix.mapping = {
        key: (lambda cfg, split, dataset=dataset: dataset)
        for key, dataset in datasets.items()
    }
    mix = {}
    for key in specifications:
        task, source = key.split("_", 1)
        mix.setdefault(task, []).append(source)
    mix_config = SimpleNamespace(
        use_custom_sampler=True,
        mix=mix,
        ratio=ratios,
        ratio_unit=ratio_unit,
    )
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            dataset_mix=SimpleNamespace(VLNDatasetMix=mix_config)
        )
    )
    return DatasetMix(cfg, "train")


def _stream_records(mix):
    return {
        stream: [mix[index] for index in indices]
        for stream, indices in mix.stream_episode_to_indices.items()
    }


class EpisodeDatasetMixTests(unittest.TestCase):
    def assert_stream_partition(self, mix):
        consumed = []
        for stream, indices in mix.stream_episode_to_indices.items():
            records = [mix[index] for index in indices]
            consumed.extend(indices)
            self.assertEqual(
                [record["clip_id"] for record in records],
                mix.stream_episode_to_clip_ids[stream],
            )
            self.assertTrue(all(record["episode_id"] == stream for record in records))
            self.assertTrue(all(
                record["orig_episode_id"]
                == mix.stream_episode_to_orig_episode[stream]
                for record in records
            ))
        self.assertEqual(Counter(consumed), Counter(range(len(mix))))

    def test_navigation_and_summary_share_replays_in_temporal_order(self):
        for domain in ("r2r", "rxr"):
            with self.subTest(domain=domain):
                nav = f"vln_mp3d_{domain}"
                summary = f"vln_traj_sum_mp3d_{domain}"
                # Summary is deliberately configured first, and neither source
                # is already sorted. At a shared clip, navigation must run first.
                mix = _make_mix({
                    summary: [("a", 2), ("a", 1), ("b", 1)],
                    nav: [("a", 2), ("a", 0), ("a", 1), ("b", 1), ("b", 0)],
                }, [2, 2])
                self.assertEqual(len(mix), 16)
                streams = _stream_records(mix)
                self.assertEqual(len(streams), 4)
                expected = {
                    "a": [(0, nav), (1, nav), (1, summary), (2, nav), (2, summary)],
                    "b": [(0, nav), (1, nav), (1, summary)],
                }
                for records in streams.values():
                    original = records[0]["orig_episode_id"]
                    self.assertEqual(
                        [(record["clip_id"], record["dataset_key"]) for record in records],
                        expected[original],
                    )
                self.assertEqual(
                    Counter(mix.stream_episode_to_orig_episode.values()),
                    {"a": 2, "b": 2},
                )
                self.assert_stream_partition(mix)

    def test_more_navigation_replays_than_summaries_keep_complete_history(self):
        nav = "vln_mp3d_r2r"
        summary = "vln_traj_sum_mp3d_r2r"
        mix = _make_mix({
            nav: [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)],
            summary: [("a", 2), ("b", 1)],
        }, [3, 1])
        self.assertEqual(len(mix), 17)
        self.assertEqual(len(mix.stream_episode_to_indices), 6)
        for original, nav_clips in (("a", [0, 1, 2]), ("b", [0, 1])):
            streams = [
                records for records in _stream_records(mix).values()
                if records[0]["orig_episode_id"] == original
            ]
            self.assertEqual(len(streams), 3)
            for records in streams:
                self.assertEqual(
                    [record["clip_id"] for record in records if record["dataset_key"] == nav],
                    nav_clips,
                )
            self.assertEqual(sum(
                record["dataset_key"] == summary
                for records in streams for record in records
            ), 1)
        self.assert_stream_partition(mix)

    def test_fractional_repetition_repeats_whole_paired_episodes(self):
        nav = "vln_mp3d_rxr"
        summary = "vln_traj_sum_mp3d_rxr"
        mix = _make_mix({
            nav: [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
            summary: [("a", 1), ("b", 1)],
        }, [2.5, 2.5])
        self.assertEqual(len(mix), 15)
        self.assertEqual(len(mix.stream_episode_to_indices), 5)
        for records in _stream_records(mix).values():
            self.assertEqual(
                [(record["clip_id"], record["dataset_key"]) for record in records],
                [(0, nav), (1, nav), (1, summary)],
            )
        self.assertEqual(
            sorted(Counter(mix.stream_episode_to_orig_episode.values()).values()),
            [2, 3],
        )
        self.assert_stream_partition(mix)

    def test_domains_and_unrelated_datasets_do_not_share_history(self):
        keys = [
            "vln_mp3d_r2r", "vln_traj_sum_mp3d_r2r",
            "vln_mp3d_rxr", "vln_traj_sum_mp3d_rxr", "vln_realworld",
        ]
        mix = _make_mix({key: [("same-original-id", 0)] for key in keys}, 1)
        groups = {
            frozenset(record["dataset_key"] for record in records)
            for records in _stream_records(mix).values()
        }
        self.assertEqual(groups, {
            frozenset(keys[:2]), frozenset(keys[2:4]), frozenset(keys[4:]),
        })
        self.assertEqual(len(mix.stream_episode_to_indices), 3)
        self.assert_stream_partition(mix)

    def test_shared_stream_keeps_source_path_and_dataset_provenance(self):
        nav = "vln_mp3d_r2r"
        summary = "vln_traj_sum_mp3d_r2r"
        mix = _make_mix({nav: [("a", 0)], summary: [("a", 0)]}, 2)
        before = [dict(dataset.meta_data[0]) for dataset in mix.datasets]
        for index, entry in enumerate(mix.index_map):
            item = mix[index]
            original = before[entry["dataset_idx"]]
            for field in ("source", "path", "dataset_key"):
                self.assertEqual(item[field], original[field])
            self.assertEqual(item["orig_episode_id"], original["episode_id"])
        self.assertEqual(
            [dataset.meta_data[0] for dataset in mix.datasets], before
        )
        self.assert_stream_partition(mix)

    def assert_missing_navigation_rejected(self, specifications, ratios):
        with self.assertRaises(ValueError) as raised:
            _make_mix(specifications, ratios)
        message = str(raised.exception).lower()
        self.assertIn("summary", message)
        self.assertIn("navigation", message)

    def test_summary_replay_requires_a_selected_navigation_counterpart(self):
        for ratios in ([1, 2], [1, 1.5], [0, 1]):
            with self.subTest(ratios=ratios):
                self.assert_missing_navigation_rejected({
                    "vln_mp3d_r2r": [("a", 0), ("b", 0)],
                    "vln_traj_sum_mp3d_r2r": [("a", 0), ("b", 0)],
                }, ratios)

    def test_summary_original_episode_must_exist_in_selected_navigation(self):
        self.assert_missing_navigation_rejected({
            "vln_mp3d_rxr": [("navigation-only", 0)],
            "vln_traj_sum_mp3d_rxr": [("summary-only", 0)],
        }, [1, 1])

    def test_smaller_fractional_summary_ratio_can_still_select_orphan_history(self):
        # The navigation subset is {a, b, d}; the separately ordered summary
        # subset is {c, d}. Comparing only the numeric ratios misses orphan c.
        self.assert_missing_navigation_rejected({
            "vln_mp3d_r2r": [(episode, 0) for episode in ("a", "b", "c", "d")],
            "vln_traj_sum_mp3d_r2r": [(episode, 0) for episode in ("c", "a", "b", "d")],
        }, [0.75, 0.5])

    def test_summary_without_its_navigation_dataset_keeps_standalone_use(self):
        summary = "vln_traj_sum_mp3d_r2r"
        other_nav = "vln_mp3d_rxr"
        mix = _make_mix({
            summary: [("a", 2), ("a", 1)],
            other_nav: [("a", 0)],
        }, [2, 1])
        self.assertEqual(len(mix), 5)
        self.assertEqual(len(mix.stream_episode_to_indices), 3)
        for records in _stream_records(mix).values():
            self.assertEqual(len({record["dataset_key"] for record in records}), 1)
        self.assert_stream_partition(mix)

    def test_sample_ratio_one_preserves_original_items_and_order(self):
        mix = _make_mix({
            "vln_mp3d_r2r": [("a", 2), ("a", 0)],
            "vln_traj_sum_mp3d_r2r": [("a", 2)],
        }, 1, ratio_unit="sample")
        originals = [item for dataset in mix.datasets for item in dataset.meta_data]
        self.assertEqual(len(mix), len(originals))
        self.assertEqual([mix[index] for index in range(len(mix))], originals)
        self.assertTrue(all("orig_episode_id" not in item for item in originals))
        self.assertIsNone(mix.stream_episode_to_indices)
        self.assertIsNone(mix.stream_episode_to_clip_ids)
        self.assertIsNone(mix.stream_episode_to_orig_episode)


if __name__ == "__main__":
    unittest.main()
