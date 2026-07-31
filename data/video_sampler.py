import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class VideoDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False, num_streams=1,
                 log_dir="./training_logs"):
        if not dist.is_available() or not dist.is_initialized():
            num_replicas = 1
            rank = 0
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.num_streams = int(num_streams)
        self.dataset = dataset
        self.log_dir = log_dir

        # Every rank writes a distinct progress sidecar. Creating the shared
        # directory on every rank avoids relying on rank 0 winning a filesystem
        # race before another rank reaches its first checkpoint.
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.video_to_indices = {}
        video_to_clip_ids = {}
        stream_episode_to_indices = getattr(self.dataset, "stream_episode_to_indices", None)

        if stream_episode_to_indices is not None:
            stream_episode_to_clip_ids = getattr(self.dataset, "stream_episode_to_clip_ids", None) or {}
            for episode_id, global_indices in stream_episode_to_indices.items():
                global_indices = list(global_indices)
                if not global_indices:
                    continue
                clip_ids = list(stream_episode_to_clip_ids.get(episode_id, range(len(global_indices))))
                if len(clip_ids) != len(global_indices):
                    clip_ids = list(range(len(global_indices)))
                self.video_to_indices[episode_id] = global_indices
                video_to_clip_ids[episode_id] = clip_ids
        elif hasattr(self.dataset, "datasets"):
            base = 0
            for ds in self.dataset.datasets:
                n = len(ds.meta_data)
                for local_idx, meta_data in enumerate(ds.meta_data):
                    global_idx = base + local_idx
                    episode_id = meta_data['episode_id']
                    clip_id = meta_data['clip_id']
                    self.video_to_indices.setdefault(episode_id, []).append(global_idx)
                    video_to_clip_ids.setdefault(episode_id, []).append(clip_id)
                base += n
        else:
            for global_idx, meta_data in enumerate(self.dataset.meta_data):
                episode_id = meta_data['episode_id']
                clip_id = meta_data.get('clip_id', global_idx)
                self.video_to_indices.setdefault(episode_id, []).append(global_idx)
                video_to_clip_ids.setdefault(episode_id, []).append(clip_id)
        self.video_ids_list = list(self.video_to_indices.keys())

        # Sort the clip indices for each video to ensure consistent ordering
        for episode_id, global_indices in self.video_to_indices.items():
            clip_ids = video_to_clip_ids[episode_id]
            sorted_pairs = sorted(zip(clip_ids, global_indices), key=lambda item: item[0])
            self.video_to_indices[episode_id] = [global_idx for _, global_idx in sorted_pairs]

        # Progress is confirmed by the trainer only after a DataLoader batch has
        # completed. ``__iter__`` may run ahead because DataLoader workers
        # prefetch indices, so dispatching an episode's last index must never
        # mutate these consumption counters.
        self.pending_episodes = []
        self._consumed_clip_counts = {}
        self._completed_episode_ids = set()
        self._active_video_ids = set()

        # Cache the length to avoid re-calculating or communicating in __len__
        self._len_cache = 0
        self._precompute_length()
    
    def _get_log_path(self, epoch):
        return os.path.join(self.log_dir, f"finished_eps_epoch_{epoch}_rank_{self.rank}.txt")

    def _get_skipped_episodes(self):
        log_path = self._get_log_path(self.epoch)
        if not os.path.exists(log_path):
            return set()
        
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        skipped = set(line.strip() for line in lines if line.strip())
        return skipped
    
    @staticmethod
    def _normalize_episode_ids(episode_ids):
        if isinstance(episode_ids, torch.Tensor):
            return episode_ids.detach().cpu().reshape(-1).tolist()
        if isinstance(episode_ids, (str, bytes)):
            return [episode_ids]
        try:
            return list(episode_ids)
        except TypeError:
            return [episode_ids]

    def mark_batch_consumed(self, episode_ids):
        """Confirm the rank-local episodes represented by one completed batch.

        An episode becomes eligible for the progress sidecar only after all of
        its clips have been returned to the trainer and their batches completed.
        This method deliberately lives outside ``__iter__`` so DataLoader
        prefetch cannot advance checkpoint progress.
        """
        if episode_ids is None:
            raise ValueError(
                "Cannot confirm sampler progress without batch episode_id values."
            )

        normalized_episode_ids = self._normalize_episode_ids(episode_ids)
        if not normalized_episode_ids:
            raise ValueError(
                "Cannot confirm sampler progress for an empty batch."
            )

        for episode_id in normalized_episode_ids:
            if episode_id not in self.video_to_indices:
                raise KeyError(
                    f"Unknown episode_id {episode_id!r} for sampler rank {self.rank}."
                )
            if episode_id not in self._active_video_ids:
                raise ValueError(
                    f"Episode {episode_id!r} is not active on sampler rank "
                    f"{self.rank} in epoch {self.epoch}."
                )
            if episode_id in self._completed_episode_ids:
                raise RuntimeError(
                    f"Episode {episode_id!r} was consumed more than once on "
                    f"sampler rank {self.rank} in epoch {self.epoch}."
                )

            consumed_count = self._consumed_clip_counts.get(episode_id, 0) + 1
            expected_count = len(self.video_to_indices[episode_id])
            if consumed_count > expected_count:
                raise RuntimeError(
                    f"Consumed {consumed_count} clips for episode "
                    f"{episode_id!r}, but the sampler only contains "
                    f"{expected_count}."
                )

            self._consumed_clip_counts[episode_id] = consumed_count
            if consumed_count == expected_count:
                self._completed_episode_ids.add(episode_id)
                self.pending_episodes.append(episode_id)
    
    def sync_progress(self):
        """
        Flushes the pending episodes to the log file.
        Call this immediately after saving model checkpoint.
        """
        if not self.pending_episodes:
            return
        
        log_path = self._get_log_path(self.epoch)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(log_path, 'a') as f:
            for eps in self.pending_episodes:
                f.write(f"{eps}\n")
            f.flush()
            os.fsync(f.fileno())
        
        # Clear the buffer now that they are safely on disk
        self.pending_episodes.clear()
        if self.rank == 0:
            print(f"Synced sampler progress for epoch {self.epoch} to {log_path}")

    def _all_gather_ints(self, local_int: int):
        """
        Helper: gather an integer from all ranks and return list of ints.

        NCCL collectives require a CUDA tensor on the rank's local device;
        ``dist.get_rank()`` is a global rank and is not a valid CUDA index on
        nodes after the first one.
        """
        if not dist.is_available() or not dist.is_initialized():
            return [local_int]

        backend = str(dist.get_backend()).lower()
        if "nccl" in backend:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "NCCL sampler synchronization requires an available CUDA device."
                )
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        world_size = dist.get_world_size()

        t = torch.tensor([local_int], dtype=torch.long, device=device)
        gather_list = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(gather_list, t)
        return [int(x.item()) for x in gather_list]
    
    def _get_my_videos(self):
        """
        Shared logic to determine which videos belong to this rank.
        Ensures consistency between __iter__ and __len__.
        """
        if self.shuffle:
            g = torch.Generator()
            # Use self.epoch + self.seed to ensure deterministic shuffle per epoch
            g.manual_seed(self.epoch + self.seed if hasattr(self, "epoch") else self.seed)
            perm = torch.randperm(len(self.video_ids_list), generator=g).tolist()
            video_ids = [self.video_ids_list[i] for i in perm]
        else:
            video_ids = self.video_ids_list.copy()
        
        total_videos = len(video_ids)
        videos_per_rank = (total_videos + self.num_replicas - 1) // self.num_replicas
        start = self.rank * videos_per_rank
        end = min(start + videos_per_rank, total_videos)

        # This list matches exactly what __iter__ would use
        return video_ids[start:end]

    def _get_active_videos(self):
        """
        Get videos for this rank AND filter out the skipped ones.
        Used by BOTH _precompute_length and __iter__ to ensure consistency.
        """
        my_videos = self._get_my_videos()
        skipped_ids = self._get_skipped_episodes()

        if skipped_ids:
            # Filter logic
            original_len = len(my_videos)
            active_videos = [vid for vid in my_videos if str(vid) not in skipped_ids]

            return active_videos
        
        return my_videos
    
    def _precompute_length(self):
        """
        Calculates the GLOBAL MINIMUM length for both the FULL epoch and the ACTIVE part considering the SKIP logic.
        This prevents DDP hangs by ensuring all ranks yield the same number of samples.
        """
        # 1. Calculate Full Epoch Length
        my_videos_full = self._get_my_videos()
        total_clips_full = sum(len(self.video_to_indices[vid]) for vid in my_videos_full)
        full_batches_local = total_clips_full // self.num_streams

        # 2. Calculate Active Length (Skipping finished eps)
        active_videos = self._get_active_videos()
        self._active_video_ids = set(active_videos)
        total_clips_active = sum(len(self.video_to_indices[vid]) for vid in active_videos)
        active_batches_local = total_clips_active // self.num_streams

        # 3. Sync BOTH across ranks
        all_full = self._all_gather_ints(full_batches_local)
        min_full = min(all_full) if all_full else 0

        all_active = self._all_gather_ints(active_batches_local)
        min_active = min(all_active) if all_active else 0

        # 4. Set Cache and Skipped Count
        self._len_cache = min_active * self.num_streams
        self.skipped_batches = (min_full - min_active) * self.num_streams

    def set_epoch(self, epoch):
        if self.pending_episodes:
            raise RuntimeError(
                "Sampler has completed episodes that were not synchronized. "
                "Call sync_progress() after the checkpoint succeeds before "
                "starting another epoch."
            )
        super().set_epoch(epoch)
        self._consumed_clip_counts.clear()
        self._completed_episode_ids.clear()
        # Re-calculate length because shuffling changes the distribution of videos
        self._precompute_length()
    
    def __iter__(self):
        # Use the same logic as precompute_length
        active_videos = self._get_active_videos()

        # Just for logging
        if self.rank == 0:
            raw_count = len(self._get_my_videos())
            curr_count = len(active_videos)
            if curr_count < raw_count:
                print(f"[Epoch {self.epoch}] Resuming: Skipping {raw_count - curr_count} videos on rank {self.rank} due to prior completion.")
        
        if not active_videos:
            return iter([])
        
        # Stream whole episodes while interleaving up to num_streams active episodes.
        remaining = iter(active_videos)
        streams = []
        for _ in range(self.num_streams):
            try:
                vid = next(remaining)
                streams.append(vid)
            except StopIteration:
                streams.append(None)
        
        ptrs = [0] * self.num_streams
        per_video_lists = {vid: self.video_to_indices[vid] for vid in active_videos}

        # Counter for DDP trimming
        yielded_count = 0
        limit = self._len_cache

        while True:
            active_any = False
            batch_indices = []

            if yielded_count >= limit:
                break

            for s in range(self.num_streams):
                vid = streams[s]
                if vid is None: continue
                lst = per_video_lists[vid]
                p = ptrs[s]

                if p < len(lst):
                    batch_indices.append(lst[p])
                    ptrs[s] = p + 1
                    active_any = True
                else:
                    try:
                        next_vid = next(remaining)
                        streams[s] = next_vid
                        ptrs[s] = 0

                        lst2 = per_video_lists[next_vid]
                        batch_indices.append(lst2[0])
                        ptrs[s] = 1
                        active_any = True
                    except StopIteration:
                        streams[s] = None
                        ptrs[s] = 0
                        continue
            if not active_any:
                break
        
            # Yield items one by one
            for idx in batch_indices:
                if yielded_count < limit:
                    yield idx
                    yielded_count += 1
                else:
                    return # Stop immediately if we hit limit

    def __len__(self):
        # Just return the cached interger. Fast and safe.
        return self._len_cache
