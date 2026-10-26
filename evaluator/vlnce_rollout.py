"""Shared fast-path utilities for VLN-CE rollout evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch


def scene_result_id(scene_id: str) -> str:
    """Return the scene identifier used by the evaluator result files."""
    parts = str(scene_id).rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def episode_result_key(episode: Any) -> tuple[str, str]:
    return scene_result_id(episode.scene_id), str(episode.episode_id)


def select_video_episode_keys(
    scene_episode_dict: Mapping[str, Sequence[Any]],
    shard_index: int,
    shard_count: int,
    limit: int,
    seed: Any = 0,
    excluded_keys: Optional[set[tuple[str, str]]] = None,
) -> set[tuple[str, str]]:
    """Select at most ``limit`` local episodes in deterministic scene round-robin order."""
    limit = max(0, int(limit))
    if limit == 0:
        return set()
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )

    local_episodes = []
    excluded_keys = excluded_keys or set()
    for scene in sorted(scene_episode_dict):
        assigned = [
            episode
            for episode in scene_episode_dict[scene][shard_index::shard_count]
            if episode_result_key(episode) not in excluded_keys
        ]

        def stable_episode_order(episode: Any) -> tuple[bytes, str]:
            scene_id, episode_id = episode_result_key(episode)
            digest = hashlib.sha256(
                f"{seed}\0{scene_id}\0{episode_id}".encode("utf-8")
            ).digest()
            return digest, episode_id

        local_episodes.append(sorted(assigned, key=stable_episode_order))
    selected: set[tuple[str, str]] = set()
    round_index = 0
    while len(selected) < limit:
        added = False
        for episodes in local_episodes:
            if round_index >= len(episodes):
                continue
            selected.add(episode_result_key(episodes[round_index]))
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        round_index += 1
    return selected


@dataclass
class VideoReplayTrace:
    scene_id: str
    episode_id: str
    instruction: str
    actions: list[int]
    navigation_text: list[str]
    generation_steps: list[int] = field(default_factory=list)
    generated_text: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.episode_id

    def validate(self) -> None:
        if len(self.actions) != len(self.navigation_text):
            raise ValueError(
                "Video replay trace action/text lengths differ for "
                f"{self.scene_id}_{self.episode_id}: "
                f"{len(self.actions)} actions and {len(self.navigation_text)} captions"
            )
        if len(self.generation_steps) != len(self.generated_text):
            raise ValueError(
                "Video replay trace generation step/text lengths differ for "
                f"{self.scene_id}_{self.episode_id}"
            )
        if any(step < 0 or step >= len(self.actions) for step in self.generation_steps):
            raise ValueError(
                "Video replay trace contains a generation step outside its "
                f"physical action trace for {self.scene_id}_{self.episode_id}"
            )


def save_video_traces(
    traces: Iterable[VideoReplayTrace],
    path: Path,
) -> Optional[Path]:
    """Atomically persist one shard's replay inputs before rendering videos."""
    traces = list(traces)
    if not traces:
        return None
    for trace in traces:
        trace.validate()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "traces": [asdict(trace) for trace in traces],
    }
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def configure_fast_habitat(
    config: Any,
    *,
    split: str,
    gpu_device_id: int,
    fast_habitat: bool = True,
) -> Any:
    """Configure the rollout environment without debug-only map measurements."""
    import habitat

    with habitat.config.read_write(config):
        simulator = config.habitat.simulator
        if hasattr(simulator, "habitat_sim_v0"):
            simulator.habitat_sim_v0.gpu_device_id = int(gpu_device_id)
        if "gpu_device_id" in simulator:
            simulator.gpu_device_id = int(gpu_device_id)
        config.habitat.dataset.split = split
        measurements = config.habitat.task.measurements
        if "ndtw" in measurements and "split" in measurements["ndtw"]:
            # Also update configs whose split interpolation was already resolved.
            measurements["ndtw"].split = split
        for name in ("top_down_map", "collisions"):
            if name in measurements:
                measurements.pop(name)
        if fast_habitat:
            sim_sensors = simulator.agents.main_agent.sim_sensors
            if "depth_sensor" in sim_sensors:
                sim_sensors.pop("depth_sensor")
            lab_sensors = getattr(config.habitat.task, "lab_sensors", None)
            if lab_sensors is not None:
                for name in ("gps_sensor", "compass_sensor"):
                    if name in lab_sensors:
                        lab_sensors.pop(name)
    return config


def build_debug_habitat_config(config: Any) -> Any:
    """Clone a fast config and enable measurements required only for video replay."""
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )

    debug_config = copy.deepcopy(config)
    with habitat.config.read_write(debug_config):
        debug_config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return debug_config


def finalize_lazy_measures(env: Any) -> set[str]:
    """Finalize measures, such as nDTW, that intentionally defer heavy work."""
    task = getattr(env, "task", None)
    if task is None:
        task = getattr(env, "_task", None)
    measurements = getattr(task, "measurements", None)
    measure_map = getattr(measurements, "measures", {})
    finalized = set()
    for name, measure in measure_map.items():
        finalize = getattr(measure, "finalize_metric", None)
        if callable(finalize):
            finalize()
            finalized.add(str(name))
    return finalized


def reset_episode_model_state(
    model: Any,
    episode_id: str,
    *,
    empty_cache_on_episode_end: bool = False,
) -> None:
    """Release model state scoped to a completed episode."""
    if model is None:
        return
    target = model.module if hasattr(model, "module") else model
    reset = getattr(target, "reset_episode_state", None)
    if callable(reset):
        reset(episode_id)
    if empty_cache_on_episode_end and torch.cuda.is_available():
        torch.cuda.empty_cache()


def replay_episode_trace(
    env: Any,
    episode: Any,
    trace: VideoReplayTrace,
    *,
    frame_builder: Callable[[Any, Any], Any],
    add_text: Callable[[Any, str, str], Any],
) -> list[Any]:
    """Replay a physical action trace and return the pre-action debug frames."""
    trace.validate()
    env.current_episode = episode
    observations = env.reset()
    frames = []
    for action, navigation_text in zip(trace.actions, trace.navigation_text):
        info = env.get_metrics()
        frame = frame_builder(observations, info)
        frames.append(add_text(frame, trace.instruction, navigation_text))
        observations = env.step(action)
    return frames


def replay_video_traces(
    config: Any,
    traces: Iterable[VideoReplayTrace],
    *,
    save_dir: Path,
    add_text: Callable[[Any, str, str], Any],
    fps: int = 6,
    quality: int = 9,
) -> None:
    """Render selected traces after the fast model rollout has completed."""
    traces = list(traces)
    if not traces:
        return

    from habitat import Env
    from habitat.utils.visualizations.utils import (
        images_to_video,
        observations_to_image,
    )

    video_dir = Path(save_dir) / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    env = Env(config=build_debug_habitat_config(config))
    try:
        episode_lookup = {
            episode_result_key(episode): episode for episode in env.episodes
        }
        for trace in traces:
            episode = episode_lookup.get(trace.key)
            if episode is None:
                raise KeyError(
                    "Video replay episode is absent from the debug environment: "
                    f"{trace.scene_id}_{trace.episode_id}"
                )
            frames = replay_episode_trace(
                env,
                episode,
                trace,
                frame_builder=lambda observations, info: observations_to_image(
                    {"rgb": observations["rgb"]}, info
                ),
                add_text=add_text,
            )
            images_to_video(
                frames,
                os.fspath(video_dir),
                f"{trace.scene_id}_{trace.episode_id}",
                fps=fps,
                quality=quality,
            )
    finally:
        env.close()
