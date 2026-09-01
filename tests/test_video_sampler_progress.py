import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

from data.video_sampler import VideoDistributedSampler


ROOT = Path(__file__).resolve().parents[1]


class _EpisodeDataset(Dataset):
    def __init__(self, episode_clips):
        self.meta_data = [
            {"episode_id": episode_id, "clip_id": clip_id}
            for episode_id, clip_id in episode_clips
        ]

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        return dict(self.meta_data[index])


def _collate_episode_batch(batch):
    return {
        "episode_id": [item["episode_id"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
    }


def _build_sampler(dataset, log_dir, *, rank=0, num_replicas=1):
    return VideoDistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False,
        drop_last=True,
        num_streams=1,
        log_dir=log_dir,
    )


class VideoSamplerConsumptionProgressTests(unittest.TestCase):
    def test_prefetch_does_not_confirm_unconsumed_episodes(self):
        dataset = _EpisodeDataset(
            [("episode-0", 0), ("episode-1", 0), ("episode-2", 0)]
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            sampler = _build_sampler(dataset, temporary_dir)
            loader = DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=1,
                prefetch_factor=2,
                collate_fn=_collate_episode_batch,
            )
            loader_iterator = iter(loader)
            try:
                # Constructing the iterator dispatches prefetched indices to the
                # worker. Dispatch alone must not advance checkpoint progress.
                self.assertEqual(sampler.pending_episodes, [])

                first_batch = next(loader_iterator)
                self.assertEqual(sampler.pending_episodes, [])

                sampler.mark_batch_consumed(first_batch["episode_id"])
                self.assertEqual(sampler.pending_episodes, ["episode-0"])
                sampler.sync_progress()
            finally:
                shutdown_workers = getattr(
                    loader_iterator, "_shutdown_workers", None
                )
                if shutdown_workers is not None:
                    shutdown_workers()

            progress_path = (
                Path(temporary_dir) / "finished_eps_epoch_0_rank_0.txt"
            )
            self.assertEqual(
                progress_path.read_text(encoding="utf-8").splitlines(),
                ["episode-0"],
            )

    def test_episode_is_confirmed_only_after_every_clip_is_consumed(self):
        dataset = _EpisodeDataset(
            [("episode-0", 0), ("episode-0", 1), ("episode-1", 0)]
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            sampler = _build_sampler(dataset, temporary_dir)

            # Exhausting the sampler models arbitrarily deep DataLoader
            # prefetch. It must remain a side-effect-free dispatch operation.
            list(sampler)
            self.assertEqual(sampler.pending_episodes, [])

            sampler.mark_batch_consumed(["episode-0"])
            self.assertEqual(sampler.pending_episodes, [])
            sampler.mark_batch_consumed(["episode-0"])
            self.assertEqual(sampler.pending_episodes, ["episode-0"])

            with self.assertRaisesRegex(RuntimeError, "more than once"):
                sampler.mark_batch_consumed(["episode-0"])

            with self.assertRaisesRegex(RuntimeError, "not synchronized"):
                sampler.set_epoch(1)

            sampler.sync_progress()
            sampler.set_epoch(1)
            self.assertEqual(sampler.pending_episodes, [])

    def test_each_ddp_rank_writes_only_its_consumed_episodes(self):
        dataset = _EpisodeDataset(
            [
                ("episode-0", 0),
                ("episode-1", 0),
                ("episode-2", 0),
                ("episode-3", 0),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            gather_local_value = lambda _sampler, value: [value, value]
            with (
                mock.patch(
                    "data.video_sampler.dist.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "data.video_sampler.dist.is_initialized",
                    return_value=True,
                ),
                mock.patch.object(
                    VideoDistributedSampler,
                    "_all_gather_ints",
                    autospec=True,
                    side_effect=gather_local_value,
                ),
            ):
                rank_0_sampler = _build_sampler(
                    dataset,
                    temporary_dir,
                    rank=0,
                    num_replicas=2,
                )
                rank_1_sampler = _build_sampler(
                    dataset,
                    temporary_dir,
                    rank=1,
                    num_replicas=2,
                )

            rank_0_sampler.mark_batch_consumed(["episode-0"])
            rank_1_sampler.mark_batch_consumed(["episode-2"])

            with self.assertRaisesRegex(ValueError, "not active"):
                rank_0_sampler.mark_batch_consumed(["episode-2"])

            rank_0_sampler.sync_progress()
            rank_1_sampler.sync_progress()

            rank_0_path = (
                Path(temporary_dir) / "finished_eps_epoch_0_rank_0.txt"
            )
            rank_1_path = (
                Path(temporary_dir) / "finished_eps_epoch_0_rank_1.txt"
            )
            self.assertEqual(
                rank_0_path.read_text(encoding="utf-8").splitlines(),
                ["episode-0"],
            )
            self.assertEqual(
                rank_1_path.read_text(encoding="utf-8").splitlines(),
                ["episode-2"],
            )

    def test_nccl_length_sync_uses_local_cuda_device_not_global_rank(self):
        sampler = object.__new__(VideoDistributedSampler)
        gathered_values = [mock.Mock(), mock.Mock()]
        gathered_values[0].item.return_value = 7
        gathered_values[1].item.return_value = 5
        local_tensor = mock.Mock()

        with (
            mock.patch(
                "data.video_sampler.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "data.video_sampler.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "data.video_sampler.dist.get_backend",
                return_value="nccl",
            ),
            mock.patch(
                "data.video_sampler.dist.get_world_size",
                return_value=2,
            ),
            mock.patch("data.video_sampler.dist.all_gather") as all_gather,
            mock.patch(
                "data.video_sampler.torch.cuda.is_available",
                return_value=True,
            ),
            mock.patch(
                "data.video_sampler.torch.cuda.current_device",
                return_value=3,
            ),
            mock.patch(
                "data.video_sampler.torch.tensor",
                return_value=local_tensor,
            ) as tensor_factory,
            mock.patch(
                "data.video_sampler.torch.zeros_like",
                side_effect=gathered_values,
            ),
        ):
            result = sampler._all_gather_ints(7)

        tensor_factory.assert_called_once_with(
            [7],
            dtype=torch.long,
            device=torch.device("cuda", 3),
        )
        all_gather.assert_called_once_with(gathered_values, local_tensor)
        self.assertEqual(result, [7, 5])

    def test_trainer_confirms_consumption_before_checkpoint_sidecar_flush(self):
        trainer_path = ROOT / "trainer" / "memvln_trainer.py"
        tree = ast.parse(
            trainer_path.read_text(encoding="utf-8"),
            filename=str(trainer_path),
        )
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MemVLNTrainer"
        )
        train_step = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_step"
        )

        calls_by_name = {}
        for node in ast.walk(train_step):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                calls_by_name.setdefault(node.func.attr, []).append(node)

        mark_call = calls_by_name["mark_batch_consumed"][0]
        checkpoint_call = min(
            calls_by_name["save"],
            key=lambda node: node.lineno,
        )
        sync_call = calls_by_name["sync_progress"][0]

        self.assertLess(mark_call.lineno, checkpoint_call.lineno)
        self.assertLess(checkpoint_call.lineno, sync_call.lineno)
        self.assertIsInstance(mark_call.args[0], ast.Name)
        self.assertEqual(mark_call.args[0].id, "batch_episode_ids")


if __name__ == "__main__":
    unittest.main()
