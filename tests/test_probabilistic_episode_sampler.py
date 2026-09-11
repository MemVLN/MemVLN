"""Behavioral checks for independent, episode-level dataset sampling.

These tests exercise the real sampler and PyTorch DataLoader, using metadata-only
episodes so they do not load Qwen, StreamVGGT, Habitat, or navigation images.
Run the optional two-process CPU DDP check with MEMVLN_RUN_DDP_TESTS=1.
"""

from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from data.probabilistic_episode_sampler import ProbabilisticEpisodeSampler


class _ToyEpisodes(Dataset):
    def __init__(self, domains):
        """domains maps each domain to its list of episode lengths."""
        self.items = []
        self.stream_episode_to_indices = {}
        self.stream_episode_to_domain = {}
        for domain, lengths in domains.items():
            for episode_number, length in enumerate(lengths):
                episode_id = f"{domain}-episode-{episode_number}"
                self.stream_episode_to_indices[episode_id] = []
                self.stream_episode_to_domain[episode_id] = domain
                for clip_id in range(length):
                    index = len(self.items)
                    self.stream_episode_to_indices[episode_id].append(index)
                    self.items.append({
                        "episode_id": episode_id,
                        "orig_episode_id": episode_id,
                        "domain": domain,
                        "clip_id": clip_id,
                        "base_index": index,
                    })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        base_index, occurrence_id = index
        item = self.items[base_index].copy()
        item["episode_id"] = occurrence_id
        return item


def _collate_rows(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}


def _sampler(dataset, count=100, *, rank=0, seed=42, probabilities=None):
    return ProbabilisticEpisodeSampler(
        dataset,
        num_samples=count,
        dataset_probabilities=(
            {"r2r": 0.5, "rxr": 0.5}
            if probabilities is None else probabilities
        ),
        seed=seed,
        rank=rank,
        num_replicas=2,
    )


def _occurrences(dataset, tokens):
    """Group consecutive samples by occurrence, preserving every clip."""
    groups = []
    for token in tokens:
        item = dataset[token]
        if not groups or groups[-1][0]["episode_id"] != item["episode_id"]:
            groups.append([])
        groups[-1].append(item)
    return groups


def _shutdown_iterator(iterator):
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def _ddp_worker(rank, rendezvous, result_dir):
    """Actual synchronized backward calls with unequal episode durations."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous}",
        rank=rank, world_size=2,
    )
    try:
        dataset = _ToyEpisodes({"r2r": [1, 3], "rxr": [7, 11]})
        updates, accumulation = 31, 3
        sampler = _sampler(dataset, count=updates * accumulation, rank=rank)
        loader = DataLoader(
            dataset, batch_size=1, sampler=sampler, num_workers=0,
            collate_fn=_collate_rows,
        )
        model = DistributedDataParallel(torch.nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / (updates + 1))
        occurrences = []
        steps = 0
        for i, batch in enumerate(loader):
            value = torch.tensor([[float(batch["clip_id"][0] + 1)]])
            update_due = (i + 1) % accumulation == 0
            with nullcontext() if update_due else model.no_sync():
                loss = model(value).square().mean() / accumulation
                loss.backward()
            if update_due:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1
            sampler.mark_batch_consumed(batch["episode_id"])
            occurrences.append(batch["episode_id"][0])
        final_weight = model.module.weight.detach().clone()
        weights = [torch.empty_like(final_weight) for _ in range(2)]
        dist.all_gather(weights, final_weight)
        (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps({
            "steps": steps,
            "microbatches": len(occurrences),
            "scheduler_steps": scheduler.last_epoch,
            "remaining": len(sampler),
            "occurrences": occurrences,
            "synchronized_weights": bool(torch.equal(weights[0], weights[1])),
        }), encoding="utf-8")
    finally:
        dist.destroy_process_group()


class ProbabilisticEpisodeSamplerTests(unittest.TestCase):
    def test_consumes_complete_episode_before_selecting_another(self):
        dataset = _ToyEpisodes({"r2r": [2, 4], "rxr": [3, 7]})
        groups = _occurrences(dataset, list(_sampler(dataset, count=173)))
        for group in groups[:-1]:
            original = group[0]["orig_episode_id"]
            self.assertEqual(
                [item["base_index"] for item in group],
                dataset.stream_episode_to_indices[original],
            )
            self.assertEqual(len({item["orig_episode_id"] for item in group}), 1)
        final = groups[-1]
        expected = dataset.stream_episode_to_indices[final[0]["orig_episode_id"]]
        self.assertEqual([item["base_index"] for item in final], expected[:len(final)])

    def test_equal_dataset_probability_is_not_equal_item_probability(self):
        dataset = _ToyEpisodes({"r2r": [2], "rxr": [8]})
        groups = _occurrences(dataset, list(_sampler(dataset, count=50_000)))
        starts = Counter(group[0]["domain"] for group in groups)
        items = Counter(item["domain"] for group in groups for item in group)
        self.assertAlmostEqual(starts["r2r"] / sum(starts.values()), 0.5, delta=0.025)
        self.assertAlmostEqual(items["r2r"] / sum(items.values()), 0.2, delta=0.025)

    def test_domain_probability_does_not_scale_with_its_episode_count(self):
        dataset = _ToyEpisodes({"r2r": [1], "rxr": [1] * 9})
        groups = _occurrences(dataset, list(_sampler(dataset, count=20_000)))
        starts = Counter(group[0]["domain"] for group in groups)
        self.assertAlmostEqual(starts["r2r"] / len(groups), 0.5, delta=0.025)
        rxr_episodes = Counter(
            group[0]["orig_episode_id"] for group in groups
            if group[0]["domain"] == "rxr"
        )
        self.assertEqual(len(rxr_episodes), 9)
        for count in rxr_episodes.values():
            self.assertAlmostEqual(count / starts["rxr"], 1 / 9, delta=0.02)

    def test_rank_seed_reproduces_locally_but_does_not_synchronize_choices(self):
        dataset = _ToyEpisodes({"r2r": [1] * 5, "rxr": [2] * 5})
        rank0 = list(_sampler(dataset, count=401, rank=0))
        rank1 = list(_sampler(dataset, count=401, rank=1))
        self.assertEqual(rank0, list(_sampler(dataset, count=401, rank=0)))
        self.assertEqual(rank1, list(_sampler(dataset, count=401, rank=1)))
        self.assertNotEqual([token[0] for token in rank0], [token[0] for token in rank1])
        self.assertTrue({token[1] for token in rank0}.isdisjoint(token[1] for token in rank1))
        self.assertEqual(len(rank0), 401)
        self.assertEqual(len(rank1), 401)

    def test_every_rank_can_sample_every_episode_without_static_sharding(self):
        dataset = _ToyEpisodes({"r2r": [1, 1], "rxr": [1, 1]})
        for rank in (0, 1):
            with self.subTest(rank=rank):
                seen = {
                    dataset[token]["orig_episode_id"]
                    for token in _sampler(dataset, count=500, rank=rank)
                }
                self.assertEqual(seen, set(dataset.stream_episode_to_indices))

    def test_immediately_repeated_episode_has_fresh_occurrence_identity(self):
        dataset = _ToyEpisodes({"r2r": [3], "rxr": [1]})
        originals = deepcopy(dataset.items)
        tokens = list(_sampler(
            dataset, count=12, probabilities={"r2r": 1.0, "rxr": 0.0},
        ))
        groups = _occurrences(dataset, tokens)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len({group[0]["episode_id"] for group in groups}), 4)
        for group in groups:
            self.assertEqual([item["clip_id"] for item in group], [0, 1, 2])
            self.assertEqual({item["orig_episode_id"] for item in group}, {"r2r-episode-0"})
        self.assertEqual(dataset.items, originals)

    def test_repeated_episode_starts_with_a_fresh_real_memory_bank(self):
        from model.memory.memory_module import ToMeMemoryBank

        config = SimpleNamespace(
            feature_dim=8, frame_max_tokens_num=2,
            episode_max_frame_num=10, memory_consolidate_type="fifo",
            use_pos_embed=False, use_time_embed=False,
            retrieval_mha_heads_num=2, retrieval_layers_num=1,
            tome_mha_heads_num=2, tome_layers_num=1, memory_token_num=2,
        )
        bank = ToMeMemoryBank(config).train()
        fresh_reference = ToMeMemoryBank(config).train()
        fresh_reference.load_state_dict(bank.state_dict())
        features = torch.randn(3, 2, 8, generator=torch.Generator().manual_seed(11))
        dataset = _ToyEpisodes({"r2r": [3], "rxr": [1]})
        sampler = _sampler(
            dataset, count=9, probabilities={"r2r": 1.0, "rxr": 0.0},
        )
        with torch.no_grad():
            first_step_reference = fresh_reference(
                features[0:1], ["fresh-reference"], torch.tensor([0]),
                current_odom=torch.zeros(1, 3),
            )
            for token in sampler:
                item = dataset[token]
                clip = item["clip_id"]
                occurrence = item["episode_id"]
                output = bank(
                    features[clip:clip + 1], [occurrence], torch.tensor([clip]),
                    current_odom=torch.zeros(1, 3),
                )
                self.assertEqual(set(bank.history_memory), {occurrence})
                self.assertEqual(list(bank.history_memory[occurrence]), list(range(clip + 1)))
                if clip == 0:
                    torch.testing.assert_close(output, first_step_reference)
                sampler.mark_batch_consumed([occurrence])

    def test_repeated_episode_keeps_streamvggt_feature_and_kv_caches_separate(self):
        from model.streamvggt.models.aggregator import Aggregator

        # Exercise the real cache update/prune methods without constructing the
        # vision transformer. Cache ownership is exactly the episode ID supplied
        # by the sampler, including the first-frame duplicate-detection lookup.
        cache = SimpleNamespace(
            past_feats={}, past_clip_ids={}, past_key_values={},
            depth=1, max_num_frames=16,
        )
        dataset = _ToyEpisodes({"r2r": [3], "rxr": [1]})
        sampler = _sampler(
            dataset, count=9, probabilities={"r2r": 1.0, "rxr": 0.0},
        )
        for token in sampler:
            item = dataset[token]
            occurrence, clip = item["episode_id"], item["clip_id"]
            if clip == 0:
                self.assertNotIn(occurrence, cache.past_key_values)
                self.assertNotIn(clip, cache.past_clip_ids.get(occurrence, []))
            key = torch.full((1, 1, clip + 1, 1, 1), float(clip))
            Aggregator.update_past_key_values(cache, [occurrence], [(key, key.clone())])
            feature = torch.full((1, 1, 2, 8), float(clip))
            Aggregator.update_past_feats(cache, [occurrence], feature, clip_ids=[clip])
            self.assertEqual(set(cache.past_key_values), {occurrence})
            self.assertEqual(set(cache.past_feats), {occurrence})
            self.assertEqual(set(cache.past_clip_ids), {occurrence})
            self.assertEqual(cache.past_clip_ids[occurrence], list(range(clip + 1)))
            self.assertEqual(len(cache.past_feats[occurrence]), clip + 1)
            sampler.mark_batch_consumed([occurrence])

    def test_sampling_is_with_replacement_not_a_forced_episode_epoch(self):
        dataset = _ToyEpisodes({"r2r": [1] * 5, "rxr": [1]})
        groups = _occurrences(dataset, list(_sampler(
            dataset, count=100, probabilities={"r2r": 1.0, "rxr": 0.0},
        )))
        originals = [group[0]["orig_episode_id"] for group in groups]
        self.assertTrue(any(a == b for a, b in zip(originals, originals[1:])))

    def test_fixed_budget_truncates_only_the_final_episode(self):
        dataset = _ToyEpisodes({"r2r": [7], "rxr": [7]})
        sampler = _sampler(dataset, count=10)
        groups = _occurrences(dataset, list(sampler))
        self.assertEqual([len(group) for group in groups], [7, 3])
        self.assertEqual(len(sampler), 10, "Dispatch is not consumption.")

    def test_dispatch_and_deep_prefetch_do_not_change_checkpoint_state(self):
        dataset = _ToyEpisodes({"r2r": [2, 3], "rxr": [4, 5]})
        sampler = _sampler(dataset, count=100)
        initial = deepcopy(sampler.state_dict())
        list(sampler)
        self.assertEqual(sampler.state_dict(), initial)
        self.assertEqual(len(sampler), 100)

    def test_workers_and_prefetch_keep_the_same_sampling_sequence(self):
        dataset = _ToyEpisodes({"r2r": [2, 3], "rxr": [4, 5]})
        expected = [dataset[token] for token in _sampler(dataset, count=37)]
        for num_workers, prefetch_factor in ((0, None), (2, 1), (2, 4)):
            with self.subTest(workers=num_workers, prefetch=prefetch_factor):
                sampler = _sampler(dataset, count=37)
                initial = deepcopy(sampler.state_dict())
                loader = DataLoader(
                    dataset, batch_size=1, sampler=sampler,
                    num_workers=num_workers, prefetch_factor=prefetch_factor,
                    collate_fn=_collate_rows,
                )
                iterator = iter(loader)
                try:
                    self.assertEqual(sampler.state_dict(), initial)
                    actual = []
                    for batch in iterator:
                        actual.append({key: values[0] for key, values in batch.items()})
                        sampler.mark_batch_consumed(batch["episode_id"])
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(sampler), 0)
                finally:
                    _shutdown_iterator(iterator)

    def test_completed_episode_resume_preserves_the_future_draw_sequence(self):
        dataset = _ToyEpisodes({"r2r": [4, 4], "rxr": [4, 4]})
        sampler = _sampler(dataset, count=40)
        tokens = list(sampler)
        for _, occurrence in tokens[:8]:
            sampler.mark_batch_consumed([occurrence])
        restored = _sampler(dataset, count=40)
        restored.load_state_dict(deepcopy(sampler.state_dict()))
        self.assertEqual(len(restored), 32)
        self.assertEqual(list(restored), tokens[8:])

    def test_partial_episode_resume_replays_history_without_resetting_budget(self):
        dataset = _ToyEpisodes({"r2r": [4, 4], "rxr": [4, 4]})
        sampler = _sampler(dataset, count=20)
        tokens = list(sampler)
        for _, occurrence in tokens[:2]:
            sampler.mark_batch_consumed([occurrence])
        restored = _sampler(dataset, count=20)
        restored.load_state_dict(deepcopy(sampler.state_dict()))
        self.assertEqual(len(restored), 18)
        replayed = list(restored)
        self.assertEqual(replayed[:4], tokens[:4])
        self.assertEqual(replayed[4:], tokens[4:18])
        self.assertEqual(len(replayed), 18)
        for _, occurrence in replayed:
            restored.mark_batch_consumed([occurrence])
        self.assertEqual(len(restored), 0)

    def test_prefetch_depth_does_not_change_the_saved_consumed_cursor(self):
        dataset = _ToyEpisodes({"r2r": [3, 7], "rxr": [4, 8]})
        states = []
        for num_workers, prefetch_factor in ((0, None), (2, 4)):
            sampler = _sampler(dataset, count=100)
            loader = DataLoader(
                dataset, batch_size=1, sampler=sampler,
                num_workers=num_workers, prefetch_factor=prefetch_factor,
                collate_fn=_collate_rows,
            )
            iterator = iter(loader)
            try:
                for _ in range(5):
                    batch = next(iterator)
                    sampler.mark_batch_consumed(batch["episode_id"])
                states.append(deepcopy(sampler.state_dict()))
                self.assertEqual(len(sampler), 95)
            finally:
                _shutdown_iterator(iterator)
        self.assertEqual(states[0], states[1])

    def test_epoch_reset_cannot_silently_restart_the_fixed_training_budget(self):
        dataset = _ToyEpisodes({"r2r": [2], "rxr": [3]})
        sampler = _sampler(dataset, count=10)
        sampler.set_epoch(0)
        with self.assertRaises((ValueError, RuntimeError)):
            sampler.set_epoch(1)

    def test_invalid_probabilities_fail_before_iteration(self):
        dataset = _ToyEpisodes({"r2r": [2], "rxr": [3]})
        invalid = (
            {"r2r": -0.1, "rxr": 1.1},
            {"r2r": math.nan, "rxr": 0.5},
            {"r2r": math.inf, "rxr": 0.5},
            {"r2r": 0.0, "rxr": 0.0},
            {"r2r": 0.2, "rxr": 0.2},
            {"r2r": 1.0, "rxr": 1.0},
            {"r2r": 1.0},
            {"r2r": 0.5, "unknown": 0.5},
            {},
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises((ValueError, TypeError)):
                    _sampler(dataset, probabilities=probabilities)

    def test_positive_probability_for_empty_domain_fails_before_iteration(self):
        dataset = _ToyEpisodes({"r2r": [], "rxr": [3]})
        with self.assertRaises((ValueError, TypeError)):
            _sampler(dataset)

    def test_an_empty_episode_pool_is_rejected_even_at_zero_probability(self):
        dataset = _ToyEpisodes({"r2r": [0], "rxr": [3]})
        with self.assertRaises((ValueError, TypeError)):
            _sampler(dataset, probabilities={"r2r": 0.0, "rxr": 1.0})

    def test_invalid_budget_and_rank_are_rejected(self):
        dataset = _ToyEpisodes({"r2r": [2], "rxr": [3]})
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count):
                with self.assertRaises((ValueError, TypeError)):
                    _sampler(dataset, count=count)
        for rank in (-1, 2):
            with self.subTest(rank=rank):
                with self.assertRaises((ValueError, TypeError)):
                    _sampler(dataset, rank=rank)

    def test_resume_rejects_mismatched_run_identity_or_data_manifest(self):
        dataset = _ToyEpisodes({"r2r": [2, 2], "rxr": [3, 3]})
        original = _sampler(dataset, count=30)
        state = deepcopy(original.state_dict())
        changed_manifest = _ToyEpisodes({"r2r": [1, 3], "rxr": [2, 4]})
        incompatible = {
            "seed": _sampler(dataset, count=30, seed=43),
            "rank": _sampler(dataset, count=30, rank=1),
            "budget": _sampler(dataset, count=31),
            "probabilities": _sampler(
                dataset, count=30, probabilities={"r2r": 0.6, "rxr": 0.4},
            ),
            "manifest": _sampler(changed_manifest, count=30),
            "world size": ProbabilisticEpisodeSampler(
                dataset, num_samples=30,
                dataset_probabilities={"r2r": 0.5, "rxr": 0.5},
                seed=42, rank=0, num_replicas=3,
            ),
        }
        for change, restored in incompatible.items():
            with self.subTest(change=change):
                with self.assertRaises((ValueError, RuntimeError)):
                    restored.load_state_dict(deepcopy(state))

    def test_unknown_consumption_identity_is_rejected(self):
        dataset = _ToyEpisodes({"r2r": [2], "rxr": [3]})
        sampler = _sampler(dataset)
        next(iter(sampler))
        before = deepcopy(sampler.state_dict())
        with self.assertRaises((ValueError, RuntimeError, KeyError)):
            sampler.mark_batch_consumed(["never-dispatched-occurrence"])
        self.assertEqual(sampler.state_dict(), before)

    def test_progress_counts_consumed_items_and_episode_starts_separately(self):
        dataset = _ToyEpisodes({"r2r": [3], "rxr": [1]})
        sampler = _sampler(
            dataset, count=7, probabilities={"r2r": 1.0, "rxr": 0.0},
        )
        tokens = list(sampler)
        initial = sampler.get_progress_metrics()
        self.assertEqual(initial["consumed_items"], 0)
        self.assertEqual(initial["started_episodes"], 0)
        for _, occurrence in tokens[:4]:
            sampler.mark_batch_consumed([occurrence])
        progress = sampler.get_progress_metrics()
        self.assertEqual(progress["consumed_items"], 4)
        self.assertEqual(progress["started_episodes"], 2)
        self.assertEqual(progress["completed_episodes"], 1)
        self.assertEqual(progress["unfinished_episode"], 1)
        self.assertEqual(progress["r2r/items"], 4)
        self.assertEqual(progress["r2r/started"], 2)
        self.assertEqual(progress["r2r/completed"], 1)
        self.assertEqual(progress["rxr/items"], 0)
        restored = _sampler(
            dataset, count=7, probabilities={"r2r": 1.0, "rxr": 0.0},
        )
        restored.load_state_dict(sampler.state_dict())
        _, occurrence = next(iter(restored))
        restored.mark_batch_consumed([occurrence])
        self.assertEqual(restored.get_progress_metrics()["replayed_items"], 1)
        self.assertEqual(restored.get_progress_metrics()["consumed_items"], 5)

    @unittest.skipUnless(
        os.environ.get("MEMVLN_RUN_DDP_TESTS") == "1",
        "Enable MEMVLN_RUN_DDP_TESTS=1 for the two-process CPU/Gloo integration check.",
    )
    def test_two_actual_ddp_processes_finish_the_same_number_of_updates(self):
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            self.skipTest("PyTorch Gloo distributed backend is unavailable.")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            context = torch.multiprocessing.spawn(
                _ddp_worker, args=(rendezvous, directory), nprocs=2, join=False,
            )
            try:
                deadline = time.monotonic() + 60
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("Two-process DDP check exceeded its 60-second timeout.")
                results = [
                    json.loads((Path(directory) / f"rank{rank}.json").read_text())
                    for rank in range(2)
                ]
                self.assertEqual([result["steps"] for result in results], [31, 31])
                self.assertEqual([result["microbatches"] for result in results], [93, 93])
                self.assertEqual([result["scheduler_steps"] for result in results], [31, 31])
                self.assertEqual([result["remaining"] for result in results], [0, 0])
                self.assertTrue(all(result["synchronized_weights"] for result in results))
                self.assertNotEqual(results[0]["occurrences"], results[1]["occurrences"])
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
