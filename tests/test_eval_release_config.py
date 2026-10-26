import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class ReleaseEvaluationConfigTests(unittest.TestCase):
    def test_paper_recipe_matches_server_evaluation_settings(self):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="eval_release")

        self.assertEqual(cfg.rng_seed, 42)
        self.assertEqual(cfg.num_gpu, 8)
        self.assertEqual(cfg.llm.torch_dtype, "bf16")
        self.assertEqual(cfg.llm.attn_implementation, "flash_attention_2")
        self.assertEqual(cfg.vision3d.name, "StreamVGGT")
        self.assertEqual(cfg.llm.mem_module.param_dtype, "bf16")
        self.assertEqual(cfg.llm.mem_module.memory_consolidate_type, "frme")
        self.assertEqual(cfg.llm.mem_module.episode_max_frame_num, 100)
        self.assertEqual(cfg.llm.mem_module.frame_max_tokens_num, 100)
        self.assertEqual(cfg.llm.mem_module.memory_token_num, 300)
        self.assertFalse(cfg.llm.mem_module.memory_merge_vggt_feat)
        self.assertEqual(cfg.data.vln.max_num_frames, 16)
        self.assertEqual(cfg.data.vln.obs_hist_sample, "nearest")
        self.assertTrue(cfg.data.vln.memory_prompt)
        self.assertTrue(cfg.data.vln.merge_action)
        self.assertFalse(cfg.data.vln.skip_merge_step_obs)
        self.assertTrue(cfg.data.vln.vggt_cache_feat)
        self.assertEqual(
            OmegaConf.to_container(cfg.inference, resolve=True),
            {
                "enabled": True,
                "cache_grounded_memory": True,
                "skip_memory_attention_weights": True,
                "fast_habitat": False,
                "empty_cache_on_episode_end": False,
                "video_selection_seed": 42,
            },
        )
        self.assertEqual(set(cfg.task.vlnce), {"R2R", "RxR"})
        for dataset in cfg.task.vlnce.values():
            self.assertEqual(dataset.args.split, "val_unseen")
            self.assertEqual(dataset.args.save_video, 0)

    def test_scene_episode_and_gt_roots_can_be_set_independently(self):
        environment = {
            "MEMVLN_SCENES_DIR": "/datasets/scene files",
            "MEMVLN_R2R_DATA_ROOT": "/datasets/r2r",
            "MEMVLN_RXR_DATA_ROOT": "/datasets/rxr_en",
            "MEMVLN_RXR_GT_ROOT": "/datasets/rxr_gt",
        }
        with patch.dict(os.environ, environment):
            r2r = OmegaConf.load(ROOT / "configs/habitat/r2r.yaml").habitat
            rxr = OmegaConf.load(ROOT / "configs/habitat/rxr.yaml").habitat
            self.assertEqual(r2r.dataset.scenes_dir, environment["MEMVLN_SCENES_DIR"])
            self.assertEqual(rxr.dataset.scenes_dir, environment["MEMVLN_SCENES_DIR"])
            self.assertEqual(
                r2r.dataset.data_path.format(split="val_unseen"),
                "/datasets/r2r/val_unseen/val_unseen.json.gz",
            )
            self.assertEqual(
                rxr.dataset.data_path.format(split="val_unseen"),
                "/datasets/rxr_en/val_unseen/val_unseen_guide_en.json.gz",
            )
            rxr.dataset.split = "val_seen"
            ndtw = rxr.task.measurements.ndtw
            self.assertEqual(ndtw.split, "val_seen")
            self.assertEqual(
                ndtw.gt_path.format(split=ndtw.split),
                "/datasets/rxr_gt/val_seen/val_seen_guide_gt.json.gz",
            )


if __name__ == "__main__":
    unittest.main()
