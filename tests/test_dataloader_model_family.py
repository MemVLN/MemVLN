import unittest
from types import SimpleNamespace
from unittest import mock

from data.build import build_dataloader_leo
from data.datasets import LeoMix
from model.qwen.factory import QWEN2_5_VL, QWEN3_VL


class LeoMixModelFamilyTests(unittest.TestCase):
    @staticmethod
    def _mix(*model_families):
        mix = LeoMix.__new__(LeoMix)
        mix.datasets = [
            SimpleNamespace(model_family=model_family)
            for model_family in model_families
        ]
        return mix

    def test_forwards_the_shared_child_model_family(self):
        mix = self._mix(QWEN2_5_VL, QWEN2_5_VL)

        self.assertEqual(mix.model_family, QWEN2_5_VL)

    def test_rejects_mixed_child_model_families(self):
        mix = self._mix(QWEN2_5_VL, QWEN3_VL)

        with self.assertRaisesRegex(ValueError, "one model family"):
            _ = mix.model_family

    def test_rejects_an_empty_mix(self):
        mix = self._mix()

        with self.assertRaisesRegex(ValueError, "at least one child dataset"):
            _ = mix.model_family


class DataloaderModelFamilyTests(unittest.TestCase):
    def test_routes_collate_from_run_config_not_dataset_wrapper(self):
        processor = object()
        dataset = SimpleNamespace(
            processor=processor,
            use_custom_sampler=False,
        )
        cfg = SimpleNamespace(
            llm=SimpleNamespace(model_family=QWEN2_5_VL),
            vision3d=SimpleNamespace(name="none"),
            data=SimpleNamespace(vln=SimpleNamespace()),
        )
        dataloader_args = SimpleNamespace(batchsize=2, num_workers=0)
        collate_fn = object()
        dataloader = object()

        with (
            mock.patch("data.build.get_dataset_leo", return_value=dataset),
            mock.patch(
                "data.build.make_collate_fn_qwen", return_value=collate_fn
            ) as make_collate,
            mock.patch("data.build.DataLoader", return_value=dataloader) as loader,
        ):
            result = build_dataloader_leo(
                cfg,
                split="train",
                dataset_name="LeoMix",
                dataloader_args=dataloader_args,
            )

        self.assertIs(result, dataloader)
        make_collate.assert_called_once_with(
            processor,
            model_family=QWEN2_5_VL,
            use_vggt=False,
            dense_vggt_input=False,
        )
        self.assertIs(loader.call_args.kwargs["collate_fn"], collate_fn)


if __name__ == "__main__":
    unittest.main()
