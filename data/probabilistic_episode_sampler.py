"""Rank-local, fixed-budget sampling of complete navigation episodes.

Dataset probabilities apply at episode boundaries, not at individual steps.
Each draw samples an episode uniformly with replacement from the chosen domain.
Only the trainer's acknowledgements advance checkpoint progress: DataLoader
prefetch is allowed to plan ahead, but cannot confirm that data were trained.
"""

import copy
import hashlib
import json
import math
import random

import torch.distributed as dist
from torch.utils.data import Sampler


class ProbabilisticEpisodeSampler(Sampler):
    is_probabilistic_episode = True
    STATE_VERSION = 1

    def __init__(self, dataset, num_samples, dataset_probabilities, seed=0,
                 rank=None, num_replicas=None):
        distributed = dist.is_available() and dist.is_initialized()
        if num_replicas is None:
            num_replicas = dist.get_world_size() if distributed else 1
        if rank is None:
            rank = dist.get_rank() if distributed else 0
        for name, value in (("num_samples", num_samples), ("num_replicas", num_replicas)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank!r} must be in [0, {num_replicas})")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = num_samples
        self.seed = seed

        episodes = getattr(dataset, "stream_episode_to_indices", None)
        domains = getattr(dataset, "stream_episode_to_domain", None)
        if not episodes or domains is None or set(episodes) != set(domains):
            raise ValueError(
                "Probabilistic episode sampling requires nonempty, paired "
                "stream_episode_to_indices and stream_episode_to_domain maps. "
                "Use VLNDatasetMix ratio_unit=episode and ratios of 1."
            )
        self.episodes = {}
        self.domain_episodes = {}
        for episode_id, indices in episodes.items():
            indices = tuple(indices)
            if not isinstance(episode_id, str) or not indices:
                raise ValueError(f"Episode {episode_id!r} must have a string ID and nonempty steps")
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(dataset)
                   for i in indices):
                raise ValueError(f"Episode {episode_id!r} has invalid dataset indices")
            domain = domains[episode_id]
            if not isinstance(domain, str) or not domain:
                raise ValueError(f"Episode {episode_id!r} has an invalid domain")
            self.episodes[episode_id] = indices
            self.domain_episodes.setdefault(domain, []).append(episode_id)

        if set(dataset_probabilities) != set(self.domain_episodes):
            raise ValueError(
                "dataset_probabilities must specify exactly the available nonempty "
                f"domains {sorted(self.domain_episodes)}, got {list(dataset_probabilities)}"
            )
        self.domains = sorted(self.domain_episodes)
        self.probabilities = []
        for domain in self.domains:
            value = dataset_probabilities[domain]
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"Probability for {domain} must be a finite number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Probability for {domain} must be finite and nonnegative")
            self.probabilities.append(float(value))
            self.domain_episodes[domain].sort()
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("dataset_probabilities must sum to 1")

        manifest = [(d, e, self.episodes[e]) for d in self.domains
                    for e in self.domain_episodes[d]]
        self.fingerprint = hashlib.sha256(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
        # A separate deterministic generator per rank; no global RNG mutation
        # and no sharding of the episode pool. Cross-rank duplicate draws are valid.
        self._rng = random.Random(f"episode-sampler-v1:{seed}:{rank}")
        self._state = {
            "consumed": 0,
            "draws": 0,
            "completed": 0,
            "active": None,
            "replayed_items": 0,
            "by_domain": {d: {"items": 0, "started": 0, "completed": 0} for d in self.domains},
        }

    def __len__(self):
        return self.num_samples - self._state["consumed"]

    def set_epoch(self, epoch):
        # A fixed update budget is one continuous stream, never an epoch loop.
        if epoch != 0:
            raise ValueError("probabilistic_episode has a fixed budget; set_epoch only accepts 0")

    def _advance(self, state, rng):
        if state["consumed"] >= self.num_samples:
            raise RuntimeError("Cannot consume beyond the fixed sampler budget")
        active = state["active"]
        if active is None:
            domain = rng.choices(self.domains, weights=self.probabilities, k=1)[0]
            episode = rng.choice(self.domain_episodes[domain])
            occurrence = f"{episode}::rank{self.rank}::draw{state['draws']}"
            active = {
                "episode": episode, "domain": domain, "id": occurrence,
                "offset": 0, "replay_until": 0,
            }
            state["active"] = active
            state["draws"] += 1
            state["by_domain"][domain]["started"] += 1
        token = (self.episodes[active["episode"]][active["offset"]], active["id"])
        if active["offset"] < active["replay_until"]:
            state["replayed_items"] += 1
        active["offset"] += 1
        state["consumed"] += 1
        state["by_domain"][active["domain"]]["items"] += 1
        if active["offset"] == len(self.episodes[active["episode"]]):
            state["completed"] += 1
            state["by_domain"][active["domain"]]["completed"] += 1
            state["active"] = None
        return token

    def __iter__(self):
        # Snapshot when iter(sampler) is called, not on the generator's first
        # next(): subsequent trainer acknowledgements cannot affect prefetch.
        state = copy.deepcopy(self._state)
        rng = random.Random()
        rng.setstate(self._rng.getstate())

        def generate():
            while state["consumed"] < self.num_samples:
                yield self._advance(state, rng)

        return generate()

    def mark_batch_consumed(self, episode_ids):
        """Acknowledge exactly one rank-local batch after its training step."""
        if isinstance(episode_ids, str):
            episode_ids = [episode_ids]
        if episode_ids is None or len(episode_ids) != 1:
            raise ValueError("Probabilistic episode sampling requires batchsize=1")
        # Keep invalid acknowledgements transactional. Copying the small active
        # state is cheap; copying the RNG is only needed at episode boundaries.
        old_state = copy.deepcopy(self._state)
        old_rng = self._rng.getstate() if self._state["active"] is None else None
        expected = self._advance(self._state, self._rng)
        if episode_ids[0] != expected[1]:
            self._state = old_state
            if old_rng is not None:
                self._rng.setstate(old_rng)
            raise ValueError(
                f"Out-of-order episode acknowledgement on rank {self.rank}: "
                f"expected {expected[1]!r}, got {episode_ids[0]!r}"
            )

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "domains": list(self.domains),
            "probabilities": list(self.probabilities),
            "rng_state": self._rng.getstate(),
            "progress": copy.deepcopy(self._state),
        }

    def load_state_dict(self, saved):
        current = self.state_dict()
        for field in ("version", "rank", "num_replicas", "num_samples", "seed",
                      "fingerprint", "domains", "probabilities"):
            if saved.get(field) != current[field]:
                raise ValueError(f"Sampler checkpoint {field} does not match this training run")
        state = copy.deepcopy(saved["progress"])
        if not 0 <= state["consumed"] <= self.num_samples:
            raise ValueError("Invalid sampler checkpoint consumed count")
        if set(state["by_domain"]) != set(self.domains):
            raise ValueError("Invalid sampler checkpoint domain counts")
        if sum(v["items"] for v in state["by_domain"].values()) != state["consumed"]:
            raise ValueError("Sampler checkpoint item counts disagree with consumption")
        if sum(v["started"] for v in state["by_domain"].values()) != state["draws"]:
            raise ValueError("Sampler checkpoint episode counts disagree with draws")
        active = state["active"]
        if active is not None:
            episode = active["episode"]
            if (episode not in self.episodes
                    or episode not in self.domain_episodes.get(active["domain"], ())
                    or not 0 <= active["offset"] < len(self.episodes[episode])):
                raise ValueError("Invalid active episode in sampler checkpoint")
            # Model checkpoints do not save the online memory/KV caches.
            # Re-train the unfinished prefix to rebuild causal history rather
            # than silently resuming mid-episode with an empty memory bank.
            # Consumed updates are not rolled back; replay uses remaining budget.
            active["replay_until"] = max(active["replay_until"], active["offset"])
            active["offset"] = 0
        rng = random.Random()
        rng.setstate(saved["rng_state"])
        self._rng = rng
        self._state = state

    def get_progress_metrics(self):
        metrics = {
            "consumed_items": self._state["consumed"],
            "started_episodes": self._state["draws"],
            "completed_episodes": self._state["completed"],
            "replayed_items": self._state["replayed_items"],
            "unfinished_episode": int(self._state["active"] is not None),
        }
        for domain, values in self._state["by_domain"].items():
            for key, value in values.items():
                metrics[f"{domain}/{key}"] = value
        return metrics
