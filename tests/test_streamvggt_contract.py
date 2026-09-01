import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.qwen.modeling_qwen3_vl import Qwen3VLModel_VGGT


ROOT = Path(__file__).resolve().parents[1]


class FakeAggregator:
    cache_feat = True

    def __init__(self, output):
        self.output = output
        self.call = None

    def __call__(self, media_input, **kwargs):
        self.call = (media_input, kwargs)
        return [self.output], 5, {"cached": True}


class FakeMerger:
    def __init__(self):
        self.call = None

    def __call__(self, *args, **kwargs):
        self.call = (args, kwargs)
        return ("history", "current")


class StreamVGGTOnlineContractTests(unittest.TestCase):
    def test_clip_ids_cover_every_dense_input_frame(self):
        media_input = [torch.zeros(3, 3, 4, 5)]

        clip_ids = Qwen3VLModel_VGGT._build_stream_clip_ids(
            media_input,
            torch.tensor([3]),
        )

        self.assertEqual(clip_ids, [[1, 2, 3]])

    def test_invalid_clip_mapping_fails_before_aggregator(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Qwen3VLModel_VGGT._build_stream_clip_ids(
                [torch.zeros(3, 3, 4, 5)],
                torch.tensor([1]),
            )

    def test_online_aggregator_updates_split_merger(self):
        output = torch.arange(1 * 3 * 2 * 4, dtype=torch.float32).reshape(
            1, 3, 2, 4
        )
        aggregator = FakeAggregator(output)
        merger = FakeMerger()
        model = SimpleNamespace(
            vggt=SimpleNamespace(dtype=torch.float32, aggregator=aggregator),
            vggt_embedding_merger=merger,
        )
        media_input = [torch.zeros(3, 3, 4, 5)]

        merged = Qwen3VLModel_VGGT.get_vggt_embeds(
            model,
            media_input=media_input,
            media_type="image",
            episode_id=["episode-a"],
            clip_ids=[[1, 2, 3]],
        )

        self.assertEqual(merged, ("history", "current"))
        self.assertEqual(aggregator.call[1]["episode_id"], ["episode-a"])
        self.assertEqual(aggregator.call[1]["clip_ids"], [[1, 2, 3]])
        self.assertTrue(aggregator.call[1]["use_cache"])
        self.assertEqual(merger.call[0][1:3], (5, "image"))
        self.assertTrue(merger.call[1]["split"])

    def test_no_memory_model_routes_split_merger_outputs_by_media_type(self):
        history = torch.tensor([1.0])
        current = torch.tensor([2.0])
        split_output = (history, current)

        self.assertIs(
            Qwen3VLModel_VGGT._select_stream_vggt_embeds(
                split_output, "video"
            ),
            history,
        )
        self.assertIs(
            Qwen3VLModel_VGGT._select_stream_vggt_embeds(
                split_output, "image"
            ),
            current,
        )
        self.assertIs(
            Qwen3VLModel_VGGT._select_stream_vggt_embeds(current, "image"),
            current,
        )
        with self.assertRaisesRegex(ValueError, "media type"):
            Qwen3VLModel_VGGT._select_stream_vggt_embeds(
                split_output, "audio"
            )

class OfflineFeatureRemovalContractTests(unittest.TestCase):
    def test_obsolete_files_and_directory_are_removed(self):
        for path in (
            ROOT / "inference.py",
            ROOT / "trainer" / "feat_extractor.py",
            ROOT / "configs" / "vision3d" / "vggt.yaml",
            ROOT / "habitat_extensions" / "summarize_results.py",
            ROOT / "model" / "vggt",
            ROOT / "model" / "qwen" / "vggt_processing.py",
        ):
            with self.subTest(path=path):
                self.assertFalse(path.exists())

        self.assertTrue(
            (ROOT / "model" / "streamvggt" / "processing.py").is_file()
        )

    def test_offline_and_auxiliary_symbols_are_absent(self):
        checked_files = [
            ROOT / "configs" / "data" / "default.yaml",
            ROOT / "configs" / "llm" / "qwen3-4b.yaml",
            ROOT / "configs" / "llm" / "qwen3-4b-mem.yaml",
            ROOT / "configs" / "vision3d" / "streamvggt.yaml",
            ROOT / "data" / "build.py",
            ROOT / "data" / "datasets.py",
            ROOT / "model" / "memvln_agent.py",
            ROOT / "model" / "qwen" / "modeling_qwen3_vl.py",
            ROOT / "model" / "qwen" / "modeling_qwen3_vl_mem_vggt.py",
            ROOT / "trainer" / "__init__.py",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in checked_files)
        forbidden_patterns = (
            r"\buse_feat\b",
            r"\bsceneverse_base\b",
            r"\bsceneverse_feat_base\b",
            r"\bqwen_feat\b",
            r"\bvggt_feat\b",
            r"\bmedia_feat\b",
            r"\bFeatExtractor\b",
            r"\buse_camera\b",
            r"\bcamera_gt\b",
            r"\bmi_loss_weight\b",
            r"\bvisual_mode\b",
            r"\badd_t_embed\b",
            r"\bvlnloc\b",
            r"\bimages_input\b",
            r"\bvideos_input\b",
            r"from model\.vggt",
            r"import model\.vggt",
        )
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, source))

    def test_online_stream_cache_and_memory_surface_is_preserved(self):
        aggregator_source = (
            ROOT / "model" / "streamvggt" / "models" / "aggregator.py"
        ).read_text(encoding="utf-8")
        qwen_source = (
            ROOT / "model" / "qwen" / "modeling_qwen3_vl.py"
        ).read_text(encoding="utf-8")
        mem_source = (
            ROOT / "model" / "qwen" / "modeling_qwen3_vl_mem_vggt.py"
        ).read_text(encoding="utf-8")

        for symbol in ("past_key_values", "past_clip_ids", "past_feats"):
            self.assertIn(symbol, aggregator_source)
        self.assertIn("VGGTEmbeddingMergerCamera", qwen_source)
        self.assertIn("split=self.vggt.aggregator.cache_feat", qwen_source)
        self.assertIn("memory_merge_vggt_feat", mem_source)


if __name__ == "__main__":
    unittest.main()
