"""Cache RGB observations and oracle actions for VLN reference trajectories.

The output directory contains one ``<trajectory_id>/rgb`` directory per
trajectory and an ``annotations.json`` file.  Episodes that share a trajectory
reuse the same action sequence while retaining their individual instructions.

Habitat is imported lazily so that ``--help`` and the validation helpers remain
usable in environments where the simulator has not been installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import closing
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOP_ACTION = 0
VALID_NAVIGATION_ACTIONS = frozenset({0, 1, 2, 3})


def trajectory_directory_name(trajectory_id: Any) -> str:
    """Return a safe directory name for a trajectory identifier."""

    name = str(trajectory_id)
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid trajectory_id: {trajectory_id!r}")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(
            "trajectory_id must not contain path separators: "
            f"{trajectory_id!r}"
        )
    return name


def validate_reference_path(reference_path: Any, trajectory_id: str) -> Sequence[Any]:
    """Validate the minimal sequence interface required by the generator."""

    if isinstance(reference_path, (str, bytes)) or not (
        hasattr(reference_path, "__len__")
        and hasattr(reference_path, "__getitem__")
    ):
        raise ValueError(
            f"reference_path for trajectory {trajectory_id!r} must be a sequence"
        )
    if len(reference_path) == 0:
        raise ValueError(
            f"reference_path for trajectory {trajectory_id!r} must not be empty"
        )
    return reference_path


def normalize_action(action: Any, trajectory_id: str) -> int:
    """Convert a Habitat action to an integer and reject invalid values."""

    if action is None:
        raise RuntimeError(
            f"ShortestPathFollower could not find an action for trajectory "
            f"{trajectory_id!r}"
        )
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError(
            f"Expected an integer action for trajectory {trajectory_id!r}, "
            f"got {action!r}"
        )

    normalized = int(action)
    if normalized not in VALID_NAVIGATION_ACTIONS:
        raise ValueError(
            f"Action for trajectory {trajectory_id!r} must be one of "
            f"{sorted(VALID_NAVIGATION_ACTIONS)}, got {normalized}"
        )
    return normalized


def scene_name(scene_id: Any) -> str:
    """Extract the scan name from a Habitat scene path."""

    path = Path(str(scene_id))
    if path.parent.name:
        return path.parent.name
    return path.stem


class CacheTrajectory:
    """Generate cached RGB frames and oracle actions from a Habitat config."""

    def __init__(
        self,
        config_path: Union[str, Path],
        output_path: Union[str, Path],
        *,
        goal_radius: float = 0.5,
        final_goal_radius: float = 0.25,
        overwrite: bool = False,
    ) -> None:
        self.config_path = Path(config_path).expanduser()
        self.output_path = Path(output_path).expanduser()
        self.goal_radius = goal_radius
        self.final_goal_radius = final_goal_radius
        self.overwrite = overwrite

        if self.goal_radius <= 0 or self.final_goal_radius <= 0:
            raise ValueError("Goal radii must be positive")

    def config_env(self) -> Any:
        """Create a Habitat environment from the configured YAML file."""

        if not self.config_path.is_file():
            raise FileNotFoundError(f"Habitat config not found: {self.config_path}")

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        import habitat
        from habitat_baselines.config.default import (
            get_config as get_habitat_config,
        )

        # Import for the registry side effects used by the project configs.
        from habitat_extensions import measures as _measures  # noqa: F401

        config = get_habitat_config(str(self.config_path))
        return habitat.Env(config=config)

    @staticmethod
    def _create_follower(env: Any, goal_radius: float) -> Any:
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        return ShortestPathFollower(
            sim=env.sim,
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def _prepare_rgb_directory(self, trajectory_id: str) -> Path:
        rgb_dir = self.output_path / trajectory_id / "rgb"
        if rgb_dir.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"RGB output already exists for trajectory {trajectory_id!r}: "
                    f"{rgb_dir}. Pass --overwrite to replace it."
                )
            shutil.rmtree(rgb_dir)
        rgb_dir.mkdir(parents=True, exist_ok=False)
        return rgb_dir

    def _next_action(
        self,
        env: Any,
        reference_path: Sequence[Any],
        next_waypoint_id: int,
        trajectory_id: str,
    ) -> tuple[int, int]:
        """Return the next non-stop action, or STOP after the final waypoint."""

        while next_waypoint_id < len(reference_path):
            is_final_waypoint = next_waypoint_id == len(reference_path) - 1
            radius = (
                self.final_goal_radius if is_final_waypoint else self.goal_radius
            )
            follower = self._create_follower(env, radius)
            action = normalize_action(
                follower.get_next_action(reference_path[next_waypoint_id]),
                trajectory_id,
            )
            if action != STOP_ACTION:
                return action, next_waypoint_id
            next_waypoint_id += 1

        return STOP_ACTION, next_waypoint_id

    def _generate_episode(
        self,
        env: Any,
        episode: Any,
        trajectory_id: str,
    ) -> list[int]:
        from PIL import Image

        reference_path = validate_reference_path(
            episode.reference_path, trajectory_id
        )
        rgb_dir = self._prepare_rgb_directory(trajectory_id)

        env.current_episode = episode
        observation = env.reset()
        actions: list[int] = []
        next_waypoint_id = 1

        while not env.episode_over:
            if "rgb" not in observation:
                raise KeyError(
                    f"Observation for trajectory {trajectory_id!r} has no 'rgb' "
                    "sensor output"
                )

            frame_id = len(actions)
            Image.fromarray(observation["rgb"]).convert("RGB").save(
                rgb_dir / f"{frame_id}.jpg"
            )

            # A one-point path has no navigation target: cache the initial frame
            # and pair it with STOP, preserving the frame/action length contract.
            if len(reference_path) == 1:
                action = STOP_ACTION
            else:
                action, next_waypoint_id = self._next_action(
                    env,
                    reference_path,
                    next_waypoint_id,
                    trajectory_id,
                )

            actions.append(action)
            if action == STOP_ACTION:
                break
            observation = env.step(action)

        if not actions:
            raise RuntimeError(
                f"Habitat ended trajectory {trajectory_id!r} before producing "
                "an observation"
            )
        return actions

    @staticmethod
    def _annotation(episode: Any, trajectory_id: str, actions: list[int]) -> dict:
        episode_id = episode.episode_id
        if isinstance(episode_id, Integral) and not isinstance(episode_id, bool):
            episode_id = int(episode_id)
        elif isinstance(episode_id, str) and episode_id.isdecimal():
            episode_id = int(episode_id)

        return {
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            "scene_id": scene_name(episode.scene_id),
            "instruction": episode.instruction.instruction_text,
            "actions": actions,
        }

    def _write_annotations(
        self,
        annotations: list[dict],
        annotations_path: Path,
    ) -> None:
        """Atomically write annotations without obscuring write failures."""

        with tempfile.TemporaryDirectory(
            prefix=".annotations-",
            dir=self.output_path,
        ) as temporary_dir:
            temporary_path = Path(temporary_dir) / annotations_path.name
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(annotations, output, indent=2, ensure_ascii=False)
                output.write("\n")
            os.replace(temporary_path, annotations_path)

    def generate(self) -> list[dict]:
        """Generate the cache, returning the annotations written to disk."""

        from tqdm import tqdm

        self.output_path.mkdir(parents=True, exist_ok=True)
        annotations_path = self.output_path / "annotations.json"
        if annotations_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Annotation output already exists: {annotations_path}. "
                "Pass --overwrite to replace it."
            )

        with closing(self.config_env()) as env:
            episodes_by_scene: dict[str, list[Any]] = {}
            for episode in env.episodes:
                episodes_by_scene.setdefault(str(episode.scene_id), []).append(episode)

            finished_trajectories: dict[str, list[int]] = {}
            annotations: list[dict] = []
            scene_progress = tqdm(
                episodes_by_scene.items(),
                desc="Scenes",
                unit="scene",
            )
            for _, episodes in scene_progress:
                episode_progress = tqdm(
                    episodes,
                    desc=scene_name(episodes[0].scene_id),
                    unit="episode",
                    leave=False,
                )
                for episode in episode_progress:
                    trajectory_id = trajectory_directory_name(
                        episode.trajectory_id
                    )
                    actions = finished_trajectories.get(trajectory_id)
                    if actions is None:
                        actions = self._generate_episode(
                            env, episode, trajectory_id
                        )
                        finished_trajectories[trajectory_id] = actions

                    annotations.append(
                        self._annotation(episode, trajectory_id, actions)
                    )

            self._write_annotations(annotations, annotations_path)

            return annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache RGB observations and oracle actions for the episodes in a "
            "Habitat VLN config."
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Habitat YAML config containing the dataset and simulator settings.",
    )
    parser.add_argument(
        "--output-path",
        "--output-dir",
        dest="output_path",
        type=Path,
        required=True,
        help="Directory in which to write trajectory frames and annotations.json.",
    )
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.5,
        help="Follower radius for intermediate waypoints (default: 0.5).",
    )
    parser.add_argument(
        "--final-goal-radius",
        type=float,
        default=0.25,
        help="Follower radius for the final waypoint (default: 0.25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing annotation and per-trajectory RGB outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generator = CacheTrajectory(
        config_path=args.config_path,
        output_path=args.output_path,
        goal_radius=args.goal_radius,
        final_goal_radius=args.final_goal_radius,
        overwrite=args.overwrite,
    )
    annotations = generator.generate()
    print(
        f"Wrote {len(annotations)} episode annotations to "
        f"{generator.output_path / 'annotations.json'}"
    )


if __name__ == "__main__":
    main()
