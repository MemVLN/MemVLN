"""Real dataset-wrapper, sampler factory, and Hydra integration checks."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import unittest
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from data.build import build_training_sampler
from data.datasets import VLNDatasetMix
from data.probabilistic_episode_sampler import ProbabilisticEpisodeSampler


ROOT = Path(__file__).resolve().parents[1]


class _Records(Dataset):
    def __init__(self, key, clips):
        self.meta_data = [
            {"episode_id": "same-source-id", "clip_id": clip,
             "source": key, "path": f"/{key}/{clip}"}
            for clip in clips
        ]

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        return self.meta_data[index]


def _collate_records(rows):
    return rows[0]


def _make_mix(*, mode="probabilistic_episode", unit="episode", ratios=None,
              use_custom=True, summary_only=False):
    specifications = {
        "vln_traj_sum_mp3d_r2r": [2, 1],
        "vln_mp3d_rxr": [1, 0, 2],
        "vln_mp3d_r2r": [2, 0, 1],
        "vln_traj_sum_mp3d_rxr": [2, 1],
    }
    if summary_only:
        del specifications["vln_mp3d_r2r"]
    datasets = {key: _Records(key, clips) for key, clips in specifications.items()}
    cfg = OmegaConf.create({
        "rng_seed": 42, "exp_dir": "/unused-in-fixed-mode",
        "training": {"max_updates": 5, "gradient_accumulation_steps": 2},
        "task": {
            "training": {"dummy_lr_decay": False},
            "dataset_mix": {"VLNDatasetMix": {
                "use_custom_sampler": use_custom,
                "mix": {"vln": [key[len("vln_"):] for key in specifications]},
                "ratio_unit": unit,
                "ratio": [1] * len(datasets) if ratios is None else ratios,
                "sampling": {"mode": mode, "dataset_probabilities": {"r2r": .5, "rxr": .5}},
            }},
        },
    })
    mapping = {key: (lambda cfg, split, ds=ds: ds) for key, ds in datasets.items()}
    with mock.patch.object(VLNDatasetMix, "mapping", mapping), mock.patch("data.datasets.logger"):
        mix = VLNDatasetMix(cfg, "train")
    return mix, cfg


class ProbabilisticEpisodeIntegrationTests(unittest.TestCase):
    def test_real_wrapper_pairs_navigation_and_summary_with_unique_draw_ids(self):
        mix, cfg = _make_mix()
        self.assertEqual(set(mix.stream_episode_to_domain.values()), {"r2r", "rxr"})
        before = deepcopy([ds.meta_data for ds in mix.datasets])
        sampler = ProbabilisticEpisodeSampler(mix, 100, {"r2r": .5, "rxr": .5})
        occurrences = defaultdict(list)
        for token in sampler:
            record = mix[token]
            occurrences[record["episode_id"]].append(record)
            self.assertEqual(record["orig_episode_id"], "same-source-id")
            sampler.mark_batch_consumed([record["episode_id"]])
        self.assertEqual(len(occurrences), 20)
        for records in occurrences.values():
            self.assertEqual([r["clip_id"] for r in records], [0, 1, 1, 2, 2])
            self.assertEqual(["traj_sum" in r["dataset_key"] for r in records],
                             [False, False, True, False, True])
            domain = "r2r" if records[0]["dataset_key"].endswith("r2r") else "rxr"
            self.assertTrue(all(r["dataset_key"].endswith(domain) for r in records))
            self.assertTrue(all(r["source"] == r["dataset_key"] for r in records))
        self.assertEqual([ds.meta_data for ds in mix.datasets], before)

    def test_real_wrapper_is_safe_with_worker_prefetch(self):
        mix, cfg = _make_mix()
        reference = [mix[t] for t in ProbabilisticEpisodeSampler(mix, 23, {"r2r": .5, "rxr": .5})]
        sampler = ProbabilisticEpisodeSampler(mix, 23, {"r2r": .5, "rxr": .5})
        loader = DataLoader(mix, batch_size=1, sampler=sampler, num_workers=2,
                            prefetch_factor=4, collate_fn=_collate_records)
        iterator = iter(loader)
        rows = []
        try:
            for row in iterator:
                rows.append(row)
                sampler.mark_batch_consumed([row["episode_id"]])
        finally:
            iterator._shutdown_workers()
        self.assertEqual(rows, reference)
        self.assertEqual(len(sampler), 0)

    def test_factory_scales_microbatch_budget_by_accumulation_only(self):
        mix, cfg = _make_mix()
        sampler = build_training_sampler(cfg, mix, 1)
        self.assertIsInstance(sampler, ProbabilisticEpisodeSampler)
        self.assertEqual(len(sampler), 10)
        self.assertEqual(sampler.probabilities, [.5, .5])

    def test_new_mode_rejects_sample_unit_repeated_ratios_and_unpaired_summary(self):
        for kwargs in ({"unit": "sample"}, {"ratios": [2, 1, 2, 1]},
                       {"ratios": [.5, 1, .5, 1]}, {"use_custom": False},
                       {"summary_only": True}, {"mode": "typo"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _make_mix(**kwargs)

    def test_factory_rejects_incomplete_or_conflicting_fixed_budget_configs(self):
        for field, value in (("training.max_updates", None), ("training.max_updates", True),
                             ("training.max_updates", 0), ("training.max_updates", 1.5),
                             ("training.gradient_accumulation_steps", 0),
                             ("task.training.dummy_lr_decay", True),
                             ("task.dataset_mix.VLNDatasetMix.sampling.mode", "epoch")):
            mix, cfg = _make_mix()
            OmegaConf.update(cfg, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_training_sampler(cfg, mix, 1)
        mix, cfg = _make_mix()
        with self.assertRaisesRegex(ValueError, "batchsize=1"):
            build_training_sampler(cfg, mix, 2)

    def test_old_sample_mode_keeps_original_ids(self):
        mix, cfg = _make_mix(mode="epoch", unit="sample")
        self.assertIsNone(mix.stream_episode_to_indices)
        self.assertEqual(mix[0]["episode_id"], "same-source-id")
        with self.assertRaisesRegex(ValueError, "ratio_unit=episode"):
            mix[(0, "new-occurrence")]

    def test_all_public_task_configs_default_to_legacy_and_accept_new_mode(self):
        for task in ("sft_r2r", "sft_r2r_w_traj_sum", "sft_r2r_rxr", "sft_r2r_rxr_w_traj_sum"):
            with self.subTest(task=task), initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
                old = compose(config_name="default", overrides=[f"task={task}"])
                self.assertIsNone(old.training.max_updates)
                self.assertEqual(old.task.dataset_mix.VLNDatasetMix.sampling.mode, "epoch")
                self.assertEqual(old.task.dataset_mix.VLNDatasetMix.ratio_unit, "sample")
                cfg = compose(config_name="default", overrides=[
                    f"task={task}", "training.max_updates=100",
                    "task.training.dummy_lr_decay=false",
                    "task.dataset_mix.VLNDatasetMix.ratio_unit=episode",
                    "task.dataset_mix.VLNDatasetMix.sampling.mode=probabilistic_episode",
                ])
                self.assertEqual(cfg.training.max_updates, 100)
                probs = cfg.task.dataset_mix.VLNDatasetMix.sampling.dataset_probabilities
                self.assertEqual(dict(probs), {"r2r": .5, "rxr": .5} if "rxr" in task else {"r2r": 1.0})


if __name__ == "__main__":
    unittest.main()
