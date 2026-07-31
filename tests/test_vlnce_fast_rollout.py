import contextlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from evaluator.vlnce_rollout import (
    VideoReplayTrace,
    configure_fast_habitat,
    finalize_lazy_measures,
    replay_episode_trace,
    replay_video_traces,
    reset_episode_model_state,
    save_video_traces,
    select_video_episode_keys,
)


def episode(scene, episode_id):
    return SimpleNamespace(scene_id=f"/dataset/{scene}/{scene}.glb", episode_id=episode_id)


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def habitat_config():
    return AttrDict(
        habitat=AttrDict(
            simulator=AttrDict(
                gpu_device_id=9,
                habitat_sim_v0=AttrDict(gpu_device_id=9),
                agents=AttrDict(
                    main_agent=AttrDict(
                        sim_sensors=AttrDict(
                            rgb_sensor=object(),
                            depth_sensor=object(),
                        )
                    )
                ),
            ),
            dataset=AttrDict(split="old"),
            task=AttrDict(
                measurements=AttrDict(
                    success=object(),
                    top_down_map=object(),
                    collisions=object(),
                ),
                lab_sensors=AttrDict(
                    gps_sensor=object(),
                    compass_sensor=object(),
                    other_sensor=object(),
                ),
            ),
        )
    )


class FastRolloutUtilityTests(unittest.TestCase):
    def test_video_selection_is_local_deterministic_scene_round_robin(self):
        scenes = {
            "/dataset/b/b.glb": [episode("b", index) for index in range(4)],
            "/dataset/a/a.glb": [episode("a", index) for index in range(5)],
        }

        selected = select_video_episode_keys(
            scenes,
            shard_index=1,
            shard_count=2,
            limit=3,
        )

        self.assertEqual(selected, {("a", "1"), ("b", "1"), ("a", "3")})
        self.assertEqual(
            selected,
            select_video_episode_keys(scenes, 1, 2, 3, seed=0),
        )
        self.assertNotEqual(
            selected,
            select_video_episode_keys(scenes, 1, 2, 3, seed=1),
        )

    def test_zero_video_limit_selects_nothing(self):
        scenes = {"/dataset/a/a.glb": [episode("a", 0)]}

        self.assertEqual(select_video_episode_keys(scenes, 0, 1, 0), set())

    def test_video_selection_excludes_completed_episodes_before_hash_ordering(self):
        scenes = {
            "/dataset/a/a.glb": [episode("a", index) for index in range(6)],
            "/dataset/b/b.glb": [episode("b", index) for index in range(4)],
        }
        completed = {("a", "0"), ("b", "0")}

        selected = select_video_episode_keys(
            scenes,
            shard_index=0,
            shard_count=2,
            limit=3,
            seed=11,
            excluded_keys=completed,
        )

        self.assertEqual(len(selected), 3)
        self.assertTrue(selected.isdisjoint(completed))

    def test_empty_trace_list_does_not_build_a_debug_environment(self):
        with mock.patch(
            "evaluator.vlnce_rollout.build_debug_habitat_config"
        ) as build_debug:
            replay_video_traces(
                object(),
                [],
                save_dir=Path("/unused"),
                add_text=mock.Mock(),
            )

        build_debug.assert_not_called()

    def test_video_trace_is_persisted_as_a_shard_local_json_before_replay(self):
        trace = VideoReplayTrace(
            scene_id="scene",
            episode_id="episode",
            instruction="向前走",
            actions=[1, 0],
            navigation_text=["move forward 25 cm", "stop"],
            generation_steps=[0, 1],
            generated_text=["move forward 25 cm", "stop"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_traces_rank_2.json"

            persisted = save_video_traces([trace], path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(persisted, path)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["traces"][0]["actions"], [1, 0])
        self.assertEqual(payload["traces"][0]["instruction"], "向前走")
        self.assertEqual(payload["traces"][0]["generation_steps"], [0, 1])
        self.assertEqual(
            payload["traces"][0]["generated_text"],
            ["move forward 25 cm", "stop"],
        )

    def test_empty_video_trace_list_does_not_create_a_trace_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_traces_rank_0.json"

            persisted = save_video_traces([], path)

            self.assertIsNone(persisted)
            self.assertFalse(path.exists())

    def test_fast_habitat_removes_only_unconsumed_sensors_and_debug_measures(self):
        @contextlib.contextmanager
        def read_write(config):
            yield config

        fake_habitat = SimpleNamespace(
            config=SimpleNamespace(read_write=read_write)
        )
        config = habitat_config()
        with mock.patch.dict(sys.modules, {"habitat": fake_habitat}):
            configure_fast_habitat(
                config,
                split="val_unseen",
                gpu_device_id=2,
                fast_habitat=True,
            )

        self.assertEqual(config.habitat.dataset.split, "val_unseen")
        self.assertEqual(config.habitat.simulator.gpu_device_id, 2)
        self.assertEqual(config.habitat.simulator.habitat_sim_v0.gpu_device_id, 2)
        self.assertEqual(
            set(config.habitat.simulator.agents.main_agent.sim_sensors),
            {"rgb_sensor"},
        )
        self.assertEqual(
            set(config.habitat.task.lab_sensors),
            {"other_sensor"},
        )
        self.assertEqual(
            set(config.habitat.task.measurements),
            {"success"},
        )

    def test_non_fast_habitat_retains_sensors_but_still_disables_inline_video(self):
        @contextlib.contextmanager
        def read_write(config):
            yield config

        fake_habitat = SimpleNamespace(
            config=SimpleNamespace(read_write=read_write)
        )
        config = habitat_config()
        with mock.patch.dict(sys.modules, {"habitat": fake_habitat}):
            configure_fast_habitat(
                config,
                split="val_seen",
                gpu_device_id=0,
                fast_habitat=False,
            )

        self.assertIn(
            "depth_sensor",
            config.habitat.simulator.agents.main_agent.sim_sensors,
        )
        self.assertIn("gps_sensor", config.habitat.task.lab_sensors)
        self.assertNotIn("top_down_map", config.habitat.task.measurements)

    def test_replay_preserves_every_physical_action_and_caption(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []
                self.observation_id = 0

            def reset(self):
                self.observation_id = 0
                return {"rgb": np.array([self.observation_id])}

            def get_metrics(self):
                return {"top_down_map": self.observation_id}

            def step(self, action):
                self.actions.append(action)
                self.observation_id += 1
                return {"rgb": np.array([self.observation_id])}

        trace = VideoReplayTrace(
            scene_id="scene",
            episode_id="episode",
            instruction="go",
            actions=[1, 1, 0],
            navigation_text=[
                "move forward 50 cm",
                "Pending action: 1",
                "stop",
            ],
        )
        env = FakeEnv()

        frames = replay_episode_trace(
            env,
            episode("scene", "episode"),
            trace,
            frame_builder=lambda observations, info: (
                int(observations["rgb"][0]),
                info["top_down_map"],
            ),
            add_text=lambda frame, instruction, caption: (
                frame,
                instruction,
                caption,
            ),
        )

        self.assertEqual(env.actions, [1, 1, 0])
        self.assertEqual(
            frames,
            [
                ((0, 0), "go", "move forward 50 cm"),
                ((1, 1), "go", "Pending action: 1"),
                ((2, 2), "go", "stop"),
            ],
        )

    def test_trace_length_mismatch_fails_fast(self):
        trace = VideoReplayTrace("scene", "episode", "go", [1], [])

        with self.assertRaisesRegex(ValueError, "lengths differ"):
            trace.validate()

    def test_lazy_measures_are_finalized_once_on_request(self):
        measure = SimpleNamespace(finalize_metric=mock.Mock())
        env = SimpleNamespace(
            task=SimpleNamespace(
                measurements=SimpleNamespace(measures={"ndtw": measure})
            )
        )

        finalized = finalize_lazy_measures(env)

        measure.finalize_metric.assert_called_once_with()
        self.assertEqual(finalized, {"ndtw"})

    def test_episode_reset_precedes_optional_allocator_cleanup(self):
        events = []

        class FakeModel:
            def reset_episode_state(self, episode_id):
                events.append(("reset", episode_id))

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torch.cuda.empty_cache",
                side_effect=lambda: events.append(("empty_cache", None)),
            ),
        ):
            reset_episode_model_state(
                FakeModel(),
                "R2R-episode",
                empty_cache_on_episode_end=True,
            )

        self.assertEqual(
            events,
            [("reset", "R2R-episode"), ("empty_cache", None)],
        )


class LazyNDTWTests(unittest.TestCase):
    def test_ndtw_defers_work_and_finalizes_idempotently(self):
        class Measure:
            pass

        class Registry:
            @staticmethod
            def register_measure(cls):
                return cls

        embodied_task = types.ModuleType("habitat.core.embodied_task")
        embodied_task.EmbodiedTask = object
        embodied_task.Measure = Measure
        registry_module = types.ModuleType("habitat.core.registry")
        registry_module.registry = Registry()
        simulator_module = types.ModuleType("habitat.core.simulator")
        simulator_module.Simulator = object
        nav_module = types.ModuleType("habitat.tasks.nav.nav")
        nav_module.DistanceToGoal = SimpleNamespace(cls_uuid="distance_to_goal")
        nav_module.Success = SimpleNamespace(cls_uuid="success")
        def dtw_impl(locations, gt_locations, dist):
            del dist
            return (
                float(len(locations) + len(gt_locations) + locations[-1][0]),
                [],
            )

        fastdtw_module = types.ModuleType("fastdtw")
        fastdtw_module.fastdtw = mock.Mock(side_effect=dtw_impl)
        stub_modules = {
            "fastdtw": fastdtw_module,
            "habitat": types.ModuleType("habitat"),
            "habitat.core": types.ModuleType("habitat.core"),
            "habitat.core.embodied_task": embodied_task,
            "habitat.core.registry": registry_module,
            "habitat.core.simulator": simulator_module,
            "habitat.tasks": types.ModuleType("habitat.tasks"),
            "habitat.tasks.nav": types.ModuleType("habitat.tasks.nav"),
            "habitat.tasks.nav.nav": nav_module,
        }
        module_path = (
            Path(__file__).resolve().parents[1]
            / "habitat_extensions"
            / "measures.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_test_lazy_vlnce_measures", module_path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stub_modules):
            spec.loader.exec_module(module)

        position = [0.0, 0.0, 0.0]
        sim = SimpleNamespace(
            get_agent_state=lambda: SimpleNamespace(
                position=np.asarray(position, dtype=np.float32)
            )
        )
        measure = module.NDTW.__new__(module.NDTW)
        measure._sim = sim
        measure.dtw_func = fastdtw_module.fastdtw
        trajectories = {
            "episode-a": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "episode-b": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            "episode-c": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        }
        gt_locations = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        measure.gt_json = {
            episode_id: {"locations": gt_locations}
            for episode_id in trajectories
        }

        for episode_index, (episode_id, locations) in enumerate(
            trajectories.items(), start=1
        ):
            position[:] = locations[0]
            measure.reset_metric(episode=SimpleNamespace(episode_id=episode_id))
            for location in locations[1:]:
                position[:] = location
                measure.update_metric()

            self.assertEqual(measure.dtw_func.call_count, episode_index - 1)
            final_only_metric = measure.finalize_metric()
            self.assertEqual(measure.dtw_func.call_count, episode_index)
            self.assertEqual(final_only_metric, measure.finalize_metric())
            self.assertEqual(measure.dtw_func.call_count, episode_index)

            old_locations = []
            old_final_metric = None
            for location in locations:
                if old_locations and location == old_locations[-1]:
                    continue
                old_locations.append(location)
                old_distance = dtw_impl(
                    old_locations,
                    gt_locations,
                    module.euclidean_distance,
                )[0]
                old_final_metric = np.exp(
                    -old_distance / (len(gt_locations) * 3.0)
                )

            self.assertEqual(measure.locations, old_locations)
            self.assertAlmostEqual(final_only_metric, old_final_metric)


if __name__ == "__main__":
    unittest.main()
