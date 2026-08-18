import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from data.build import build_dataloader_leo, make_collate_fn_qwen
from data.datasets import LeoMix
from model.qwen.factory import QWEN2_5_VL, QWEN3_VL


class _CollateTokenizer:
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        if text != "<|im_start|>assistant" or add_special_tokens:
            raise AssertionError("Unexpected tokenizer.encode call")
        return [7, 8]

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join("\n" if token_id == 9 else "token" for token_id in token_ids)


def _sample(index, last_role="assistant", output_gt="turn left"):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Go"}]},
    ]
    if last_role == "assistant":
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_gt}],
        })
    return {
        "source": "dataset-source",
        "scene_id": f"scene-{index}",
        "episode_id": f"episode-{index}",
        "clip_id": index,
        "messages": messages,
        "output_gt": output_gt,
    }


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


class QwenCollateContractTests(unittest.TestCase):
    @staticmethod
    def _processor(input_ids):
        return SimpleNamespace(
            tokenizer=_CollateTokenizer(),
            apply_chat_template=mock.Mock(return_value={
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.ones_like(torch.tensor(input_ids)),
            }),
        )

    def test_supervised_and_generation_batches_use_matching_template_mode(self):
        supervised_processor = self._processor([[1, 7, 8, 9, 41, 42]])
        supervised_collate = make_collate_fn_qwen(
            supervised_processor,
            model_family=QWEN3_VL,
        )
        supervised_sample = _sample(1)

        supervised = supervised_collate([supervised_sample])

        self.assertFalse(
            supervised_processor.apply_chat_template.call_args.kwargs[
                "add_generation_prompt"
            ]
        )
        self.assertIn("messages", supervised_sample)
        self.assertNotIn("messages", supervised)
        self.assertTrue(torch.all(supervised["labels"][0, :4] == -100))
        torch.testing.assert_close(
            supervised["labels"][0, 4:],
            supervised["input_ids"][0, 4:],
        )

        generation_processor = self._processor([[1, 7, 8, 9]])
        generation_collate = make_collate_fn_qwen(
            generation_processor,
            model_family=QWEN3_VL,
        )
        generation_sample = _sample(2, last_role="user")

        generation = generation_collate([generation_sample])

        self.assertTrue(
            generation_processor.apply_chat_template.call_args.kwargs[
                "add_generation_prompt"
            ]
        )
        self.assertTrue(torch.all(generation["labels"] == -100))

    def test_mixed_batch_fails_with_sample_identity(self):
        processor = self._processor([[1, 7, 8, 9], [1, 7, 8, 9]])
        collate = make_collate_fn_qwen(processor, model_family=QWEN3_VL)
        batch = [_sample(1), _sample(2, last_role="user")]

        with self.assertRaisesRegex(
            ValueError,
            "Cannot mix supervised and generation-only conversations",
        ) as caught:
            collate(batch)

        error = str(caught.exception)
        self.assertIn("episode_id='episode-1'", error)
        self.assertIn("episode_id='episode-2'", error)
        processor.apply_chat_template.assert_not_called()

    def test_masking_failure_preserves_batch_and_reports_provenance(self):
        processor = self._processor([[1, 2, 3, 4]])
        collate = make_collate_fn_qwen(processor, model_family=QWEN3_VL)
        sample = _sample(7, output_gt="\nproblematic instruction")

        with self.assertRaisesRegex(
            ValueError,
            "Qwen assistant label masking failed",
        ) as caught:
            collate([sample])

        self.assertIsInstance(caught.exception.__cause__, ValueError)
        error = str(caught.exception)
        for expected in (
            "source='dataset-source'",
            "scene_id='scene-7'",
            "episode_id='episode-7'",
            "clip_id=7",
            "roles=['user', 'assistant']",
            "output_gt='\\nproblematic instruction'",
            "active_token_length=4",
        ):
            self.assertIn(expected, error)
        self.assertIn("messages", sample)


if __name__ == "__main__":
    unittest.main()
