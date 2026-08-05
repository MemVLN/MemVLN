import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from transformers import (
    AutoConfig,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from model.leo_agent import LeoAgent
from model.qwen.factory import (
    MEMORY_END_TOKEN,
    MEMORY_START_TOKEN,
    MEMORY_TOKEN,
    QWEN2_5_VL,
    QWEN3_VL,
    apply_chat_template,
    build_processor,
    configure_processor,
    count_multimodal_tokens,
    expand_memory_markers,
    get_model_class,
    initialize_memory_tokens,
    mask_labels_for_last_assistant,
)
from model.qwen.modeling_qwen2_5_vl_mem_vggt import (
    Qwen2_5_VLForConditionalGeneration_Mem_VGGT,
    Qwen2_5_VLModel_Mem_VGGT,
)
from common.optim_utils import build_lora_optimizer_params
from model.streamvggt.processing import build_online_vggt_input


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
LOCAL_QWEN25 = Path(
    os.environ.get(
        "QWEN25_VL_7B_PATH",
        "/Users/jetpack/Documents/pretrained_weights/Qwen2.5-VL-7B-Instruct",
    )
)


def tiny_qwen25_config():
    return Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_scaling": {
                "type": "default",
                "mrope_section": [2, 1, 1],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "fullatt_block_indexes": [0],
            "window_size": 8,
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.additional_special_tokens = ["<existing_special>"]
        self.ids = {"<existing_special>": 6}
        self.next_id = 10
        self.replace_additional_special_tokens = None

    def add_special_tokens(
        self,
        payload,
        replace_additional_special_tokens=True,
    ):
        self.replace_additional_special_tokens = replace_additional_special_tokens
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.ids:
                self.ids[token] = self.next_id
                self.next_id += 1
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token)

    def encode(self, text, add_special_tokens=False):
        if text != "<|im_start|>assistant\n" or add_special_tokens:
            raise AssertionError("Unexpected tokenizer.encode call")
        return [7, 8, 9]


class FactoryAndConfigTests(unittest.TestCase):
    def test_control_configs_lock_the_two_qwen25_endpoints(self):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_DIR),
        ):
            baseline = compose(config_name="qwen2_5_baseline")
            full = compose(config_name="qwen2_5_full")
            qwen3_default = compose(config_name="default")

        self.assertEqual(baseline.llm.model_family, QWEN2_5_VL)
        self.assertEqual(baseline.llm.hidden_dim, 3584)
        self.assertEqual(baseline.llm.lora.rank, 16)
        self.assertEqual(baseline.llm.lora.alpha, 16)
        self.assertTrue(baseline.eval.run_after_train)
        self.assertNotIn("mem_module", baseline.llm)
        self.assertEqual(baseline.vision3d.name, "Dummy")
        self.assertFalse(baseline.data.vln.memory_prompt)
        self.assertEqual(
            list(baseline.task.leomix.LeoMix.mix.vln),
            ["mp3d_r2r", "mp3d_rxr_v2"],
        )

        self.assertEqual(full.llm.model_family, QWEN2_5_VL)
        self.assertEqual(full.llm.hidden_dim, 3584)
        self.assertEqual(full.llm.lora.rank, 16)
        self.assertEqual(full.llm.lora.alpha, 16)
        self.assertTrue(full.eval.run_after_train)
        self.assertEqual(
            list(full.llm.lora.target_modules),
            list(baseline.llm.lora.target_modules),
        )
        self.assertEqual(full.llm.mem_module.feature_dim, 3584)
        self.assertEqual(full.llm.mem_module.memory_token_num, 300)
        self.assertEqual(full.llm.mem_module.frame_max_tokens_num, 100)
        self.assertEqual(full.vision3d.merger_type, "camera")
        self.assertEqual(full.vision3d.fusion_op, "add")
        self.assertFalse(full.vision3d.fusion_learnable)
        self.assertEqual(list(full.vision3d.input_grid_hw), [30, 40])

        self.assertEqual(qwen3_default.llm.model_family, QWEN3_VL)
        self.assertEqual(qwen3_default.llm.lora.rank, 128)
        self.assertEqual(qwen3_default.llm.lora.alpha, 256)
        self.assertEqual(qwen3_default.llm.mem_module.memory_token_num, 300)

    def test_factory_uses_official_plain_and_thin_full_classes(self):
        plain_cls = get_model_class(QWEN2_5_VL)
        full_cls = get_model_class(
            QWEN2_5_VL,
            use_mem=True,
            use_vggt=True,
        )

        self.assertIs(plain_cls, Qwen2_5_VLForConditionalGeneration)
        self.assertIs(full_cls, Qwen2_5_VLForConditionalGeneration_Mem_VGGT)
        self.assertTrue(
            issubclass(
                Qwen2_5_VLModel_Mem_VGGT,
                __import__(
                    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
                    fromlist=["Qwen2_5_VLModel"],
                ).Qwen2_5_VLModel,
            )
        )
        for use_mem, use_vggt in ((True, False), (False, True)):
            with self.subTest(use_mem=use_mem, use_vggt=use_vggt):
                with self.assertRaisesRegex(ValueError, "only the plain baseline"):
                    get_model_class(
                        QWEN2_5_VL,
                        use_mem=use_mem,
                        use_vggt=use_vggt,
                    )

    def test_qwen25_processor_factory_forces_official_slow_processor(self):
        cfg = SimpleNamespace(
            llm=SimpleNamespace(
                model_family=QWEN2_5_VL,
                cfg_path="checkpoint",
                padding_side="left",
            )
        )
        sentinel = object()
        with mock.patch.object(
            Qwen2_5_VLProcessor,
            "from_pretrained",
            return_value=sentinel,
        ) as from_pretrained:
            self.assertIs(build_processor(cfg), sentinel)

        from_pretrained.assert_called_once_with(
            "checkpoint",
            use_fast=False,
            padding_side="left",
        )
        with self.assertRaisesRegex(ValueError, "use_fast=False"):
            build_processor(cfg, use_fast=True)

    def test_qwen25_vggt_glue_does_not_replace_official_preprocess(self):
        class VideoProcessor:
            def preprocess(self, videos=None, **kwargs):
                return videos, kwargs

        processor = SimpleNamespace(
            image_processor=SimpleNamespace(),
            video_processor=VideoProcessor(),
        )
        native_preprocess = processor.video_processor.preprocess.__func__
        cfg = SimpleNamespace(
            llm=SimpleNamespace(model_family=QWEN2_5_VL),
            data=SimpleNamespace(
                visual_size=SimpleNamespace(
                    image={"shortest_edge": 1, "longest_edge": 2},
                    video={"shortest_edge": 3, "longest_edge": 4},
                )
            ),
        )

        configure_processor(processor, cfg, use_vggt=True)

        self.assertIs(processor.video_processor.preprocess.__func__, native_preprocess)
        self.assertTrue(callable(processor.video_processor.vggt_preprocess))
        self.assertFalse(processor.video_processor.do_sample_frames)


class MemoryAndLabelGlueTests(unittest.TestCase):
    def test_memory_tokens_preserve_existing_specials_and_expand_to_300(self):
        tokenizer = FakeTokenizer()
        processor = SimpleNamespace(tokenizer=tokenizer)

        token_ids = initialize_memory_tokens(processor, 300, QWEN2_5_VL)

        self.assertEqual(token_ids, (10, 11, 12))
        self.assertFalse(tokenizer.replace_additional_special_tokens)
        self.assertIn("<existing_special>", tokenizer.additional_special_tokens)
        self.assertEqual(processor.memory_token_num, 300)
        self.assertEqual(processor.memory_token_id, 11)

        original = [{"role": "user", "content": [{"type": "text", "text": (
            MEMORY_START_TOKEN + MEMORY_TOKEN + MEMORY_END_TOKEN
        )}]}]
        expanded = expand_memory_markers(original, 300)
        text = expanded[0]["content"][0]["text"]
        self.assertEqual(text.count(MEMORY_TOKEN), 300)
        self.assertEqual(original[0]["content"][0]["text"].count(MEMORY_TOKEN), 1)
        self.assertEqual(expand_memory_markers(expanded, 300), expanded)

        malformed = MEMORY_TOKEN * 2
        with self.assertRaisesRegex(ValueError, "either one placeholder"):
            expand_memory_markers(malformed, 300)

    def test_dynamic_label_mask_uses_last_active_assistant_header(self):
        tokenizer = FakeTokenizer()
        input_ids = torch.tensor(
            [
                [0, 0, 1, 7, 8, 9, 41, 2, 7, 8, 9, 51, 52, 53],
                [0, 0, 0, 0, 1, 2, 7, 8, 9, 61, 62, 63, 64, 65],
            ]
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        labels = mask_labels_for_last_assistant(
            input_ids,
            tokenizer,
            attention_mask=attention_mask,
        )

        self.assertTrue(torch.all(labels[0, :11] == -100))
        torch.testing.assert_close(labels[0, 11:], input_ids[0, 11:])
        self.assertTrue(torch.all(labels[1, :9] == -100))
        torch.testing.assert_close(labels[1, 9:], input_ids[1, 9:])
        with self.assertRaisesRegex(ValueError, "assistant header"):
            mask_labels_for_last_assistant(
                torch.tensor([[1, 2, 3, 4]]),
                tokenizer,
            )

    def test_chat_template_adds_only_qwen25_timing_and_memory_glue(self):
        qwen25_processor = SimpleNamespace(
            memory_token_num=3,
            apply_chat_template=mock.Mock(return_value={"ok": True}),
        )
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": MEMORY_TOKEN}],
        }]

        result = apply_chat_template(
            qwen25_processor,
            messages,
            QWEN2_5_VL,
            tokenize=True,
        )

        self.assertEqual(result, {"ok": True})
        call = qwen25_processor.apply_chat_template.call_args
        self.assertEqual(
            call.args[0][0]["content"][0]["text"].count(MEMORY_TOKEN),
            3,
        )
        self.assertEqual(messages[0]["content"][0]["text"], MEMORY_TOKEN)
        self.assertEqual(call.kwargs["fps"], 2.0)
        self.assertFalse(call.kwargs["do_sample_frames"])

        qwen3_processor = SimpleNamespace(
            apply_chat_template=mock.Mock(return_value="qwen3")
        )
        self.assertEqual(
            apply_chat_template(
                qwen3_processor,
                messages,
                QWEN3_VL,
                tokenize=True,
            ),
            "qwen3",
        )
        qwen3_processor.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=True,
        )

    def test_memory_embedding_hook_keeps_only_resolved_token_ids(self):
        weight = nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(12, 4))
        memory_ids = (2, 7, 10)
        weight.register_hook(
            LeoAgent.make_memory_embedding_grad_hook(None, memory_ids)
        )

        weight.sum().backward()

        for row in range(weight.shape[0]):
            expected = torch.ones(4) if row in memory_ids else torch.zeros(4)
            torch.testing.assert_close(weight.grad[row], expected)

    def test_adamw_updates_only_memory_embedding_rows(self):
        weight = nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(12, 4))
        memory_ids = (2, 7, 10)
        weight.register_hook(
            LeoAgent.make_memory_embedding_grad_hook(None, memory_ids)
        )
        lora_parameter = nn.Parameter(torch.ones(2, 2))
        groups = build_lora_optimizer_params(
            {
                "model.embed_tokens.weight": weight,
                "model.q_proj.lora_A.weight": lora_parameter,
            },
            model_family=QWEN2_5_VL,
            use_mem=True,
        )
        optimizer = torch.optim.AdamW(groups, lr=0.1, weight_decay=0.05)
        before = weight.detach().clone()

        (weight.sum() + lora_parameter.sum()).backward()
        optimizer.step()

        for row in range(weight.shape[0]):
            if row in memory_ids:
                self.assertFalse(torch.equal(weight[row], before[row]))
            else:
                torch.testing.assert_close(weight[row], before[row])
        self.assertEqual(optimizer.param_groups[-1]["weight_decay"], 0.0)

    def test_full_agent_freezes_everything_except_declared_modules(self):
        tiny_model = Qwen2_5_VLForConditionalGeneration_Mem_VGGT(
            tiny_qwen25_config()
        )
        cfg = OmegaConf.create(
            {
                "llm": {
                    "model_family": QWEN2_5_VL,
                    "cfg_path": "unused",
                    "max_out_len": 8,
                    "hidden_dim": 32,
                    "tune_proj": False,
                    "lora": {
                        "flag": True,
                        "rank": 2,
                        "alpha": 4,
                        "target_modules": ["q_proj"],
                        "dropout": 0.0,
                    },
                    "mem_module": {},
                },
                "vision3d": {"name": "StreamVGGT"},
            }
        )

        def attach_memory(agent, _cfg):
            agent.memory_token_ids = (2, 7, 10)
            agent.model.model.memory = nn.Linear(32, 32, bias=False)
            agent.model.model.memory_token_id = 7

        def attach_vggt(agent, _cfg):
            agent.model.model.vggt = nn.Linear(3, 3, bias=False)
            agent.model.model.vggt_embedding_merger = nn.Linear(
                32, 32, bias=False
            )

        with (
            mock.patch("model.leo_agent.build_processor", return_value=SimpleNamespace()),
            mock.patch("model.leo_agent.build_model", return_value=tiny_model),
            mock.patch("model.leo_agent.logger.info"),
            mock.patch.object(LeoAgent, "build_mem_modules", attach_memory),
            mock.patch.object(LeoAgent, "build_vggt_modules", attach_vggt),
        ):
            agent = LeoAgent(cfg)

        trainable = {
            name for name, parameter in agent.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any("lora_" in name for name in trainable))
        self.assertTrue(any(".memory." in name for name in trainable))
        self.assertTrue(any("vggt_embedding_merger" in name for name in trainable))
        self.assertTrue(any("embed_tokens.weight" in name for name in trainable))
        self.assertFalse(any(".vggt.weight" in name for name in trainable))
        self.assertFalse(any(".visual." in name for name in trainable))


class TinyOfficialModelContractTests(unittest.TestCase):
    def test_full_adapter_loads_an_official_checkpoint_without_base_key_drift(self):
        torch.manual_seed(7)
        official = Qwen2_5_VLForConditionalGeneration(
            tiny_qwen25_config()
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            official.save_pretrained(directory)
            full, loading_info = (
                Qwen2_5_VLForConditionalGeneration_Mem_VGGT.from_pretrained(
                    directory,
                    output_loading_info=True,
                )
            )
        full.eval()

        self.assertEqual(loading_info["missing_keys"], [])
        self.assertEqual(loading_info["unexpected_keys"], [])
        self.assertEqual(loading_info["mismatched_keys"], [])
        self.assertEqual(
            tuple(official.state_dict()),
            tuple(full.state_dict()),
        )

        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            official_logits = official(input_ids, use_cache=False).logits
            full_logits = full(input_ids, use_cache=False).logits
        torch.testing.assert_close(full_logits, official_logits, rtol=0, atol=0)

    def test_visual_prefill_calls_memory_once(self):
        class FakeMemory(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, visual_features, *args):
                self.calls += 1
                return visual_features.new_zeros(
                    visual_features.shape[0],
                    2,
                    visual_features.shape[-1],
                )

        model = Qwen2_5_VLForConditionalGeneration_Mem_VGGT(
            tiny_qwen25_config()
        ).eval()
        model.model.memory = FakeMemory()
        model.model.memory_token_id = 124
        input_ids = torch.tensor([[122, 120, 123, 124, 124, 1]])

        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=torch.randn(4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            episode_id=["episode-a"],
            clip_id=torch.tensor([0]),
            current_odom=torch.zeros(1, 3),
            use_cache=False,
        )

        self.assertEqual(tuple(output.logits.shape), (1, 6, 128))
        self.assertEqual(model.model.memory.calls, 1)

    def test_generation_keeps_visual_state_only_during_prefill(self):
        model = Qwen2_5_VLForConditionalGeneration_Mem_VGGT(
            tiny_qwen25_config()
        ).eval()
        input_ids = torch.tensor([[122, 121, 123, 1]])
        vggt_input = [torch.zeros(1, 3, 4, 4)]
        common = {
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values_videos": torch.randn(4, 24),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "second_per_grid_ts": torch.tensor([1.0]),
            "vggt_input": vggt_input,
            "episode_id": ["episode-a"],
            "clip_id": torch.tensor([0]),
            "current_odom": torch.zeros(1, 3),
        }

        prefill = model.prepare_inputs_for_generation(
            input_ids,
            cache_position=torch.arange(input_ids.shape[1]),
            **common,
        )
        decode = model.prepare_inputs_for_generation(
            input_ids[:, -1:],
            cache_position=torch.tensor([input_ids.shape[1]]),
            position_ids=torch.tensor([[input_ids.shape[1]]]),
            **common,
        )

        self.assertIs(prefill["vggt_input"], vggt_input)
        self.assertEqual(prefill["episode_id"], ["episode-a"])
        torch.testing.assert_close(
            prefill["second_per_grid_ts"], torch.tensor([1.0])
        )
        self.assertIsNone(decode["pixel_values_videos"])
        self.assertIsNone(decode["vggt_input"])
        self.assertIsNone(decode["episode_id"])
        self.assertIsNone(decode["clip_id"])
        self.assertIsNone(decode["current_odom"])

    def test_qwen25_full_generation_rejects_non_greedy_modes(self):
        agent = SimpleNamespace(
            model_family=QWEN2_5_VL,
            use_mem=True,
            use_vggt=True,
        )
        for kwargs in (
            {"do_sample": True},
            {"num_beams": 2},
            {"num_return_sequences": 2},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "greedy decoding"):
                    LeoAgent.generate(agent, {"input_ids": torch.tensor([[1]])}, **kwargs)


@unittest.skipUnless(
    LOCAL_QWEN25.is_dir(),
    f"local Qwen2.5 processor files not found at {LOCAL_QWEN25}",
)
class LocalOfficialProcessorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_DIR),
        ):
            cls.cfg = compose(config_name="qwen2_5_full")
        cls.processor = build_processor(cls.cfg)
        cls.native_video_preprocess = (
            cls.processor.video_processor.preprocess.__func__,
        )
        cls.original_special_tokens = set(
            cls.processor.tokenizer.additional_special_tokens
        )
        configure_processor(cls.processor, cls.cfg, use_vggt=True)
        cls.memory_token_ids = initialize_memory_tokens(
            cls.processor,
            300,
            cls.cfg,
        )

    def test_official_processor_contract_with_real_local_files(self):
        self.assertIsInstance(self.processor, Qwen2_5_VLProcessor)
        self.assertEqual(type(self.processor.image_processor).__name__, "Qwen2VLImageProcessor")
        self.assertIs(
            self.processor.video_processor.preprocess.__func__,
            self.native_video_preprocess[0],
        )
        self.assertTrue(
            self.original_special_tokens.issubset(
                set(self.processor.tokenizer.additional_special_tokens)
            )
        )
        self.assertEqual(self.memory_token_ids, (151665, 151666, 151667))
        self.assertLess(max(self.memory_token_ids), 152064)

        image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": MEMORY_START_TOKEN + MEMORY_TOKEN + MEMORY_END_TOKEN,
                },
                {"type": "video", "video": [image] * 16},
                {"type": "image", "image": image},
                {"type": "text", "text": "What is the next action?"},
            ],
        }]
        output = apply_chat_template(
            self.processor,
            messages,
            self.cfg,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        self.assertEqual(output["image_grid_thw"].tolist(), [[1, 34, 46]])
        self.assertEqual(
            int((output["input_ids"] == self.processor.image_token_id).sum()),
            391,
        )
        self.assertEqual(output["video_grid_thw"].tolist(), [[8, 34, 46]])
        self.assertEqual(
            int((output["input_ids"] == self.processor.video_token_id).sum()),
            3128,
        )
        torch.testing.assert_close(
            output["second_per_grid_ts"],
            torch.tensor([1.0]),
        )
        self.assertEqual(
            int((output["input_ids"] == self.processor.memory_token_id).sum()),
            300,
        )
        counts = count_multimodal_tokens(output["input_ids"], self.processor)
        self.assertEqual(counts["visual_token_count"].tolist(), [3519])
        self.assertEqual(counts["memory_token_count"].tolist(), [300])
        vggt_input = build_online_vggt_input(
            self.processor,
            messages,
            output["image_grid_thw"],
            dense_history=True,
        )
        self.assertEqual(tuple(vggt_input[0].shape), (3, 3, 420, 560))

    def test_real_checkpoint_index_matches_full_class_after_official_key_conversion(self):
        config = AutoConfig.from_pretrained(LOCAL_QWEN25)
        with init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration_Mem_VGGT(config)
        model_keys = set(model.state_dict())

        index = json.loads(
            (LOCAL_QWEN25 / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint_keys = set(index["weight_map"])
        converted_keys = set()
        for key in checkpoint_keys:
            for pattern, replacement in model._checkpoint_conversion_mapping.items():
                key = re.sub(pattern, replacement, key)
            converted_keys.add(key)

        self.assertEqual(converted_keys, model_keys)


if __name__ == "__main__":
    unittest.main()
