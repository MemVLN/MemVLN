import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from model.memory.memory_module import ToMeMemoryBank


def make_memory_config(consolidation="fifo", max_frames=3):
    return SimpleNamespace(
        feature_dim=8,
        frame_max_tokens_num=2,
        episode_max_frame_num=max_frames,
        memory_consolidate_type=consolidation,
        use_pos_embed="fourier",
        pos_embed_fourier_feat_num=4,
        pos_embed_learnable=True,
        pos_embed_mlp_hidden=8,
        pos_embed_mapping_type="rff",
        use_time_embed="sinus",
        time_embed_max_length_num=32,
        retrieval_mha_heads_num=2,
        retrieval_layers_num=1,
        tome_mha_heads_num=2,
        tome_layers_num=1,
        memory_token_num=2,
    )


def make_inference_config(enabled):
    return SimpleNamespace(
        enabled=enabled,
        cache_grounded_memory=True,
        skip_memory_attention_weights=True,
    )


def step(module, feature, clip_id, episode="episode-a"):
    return module(
        feature.unsqueeze(0),
        [episode],
        torch.tensor([clip_id]),
        current_odom=torch.tensor([[clip_id * 0.25, 0.0, clip_id * 15.0]]),
    )


class MemoryInferenceCacheTests(unittest.TestCase):
    def make_pair(self, consolidation="fifo", max_frames=3):
        torch.manual_seed(7)
        reference = ToMeMemoryBank(
            make_memory_config(consolidation, max_frames),
            make_inference_config(False),
        )
        cached = ToMeMemoryBank(
            make_memory_config(consolidation, max_frames),
            make_inference_config(True),
        )
        cached.load_state_dict(reference.state_dict())
        reference.eval()
        cached.eval()
        return reference, cached

    def test_cached_and_reference_eval_match_and_duplicate_is_not_rebuilt(self):
        reference, cached = self.make_pair()
        features = [torch.randn(4, 8) for _ in range(3)]

        with mock.patch.object(
            cached,
            "_compress_batch_clips_tensor",
            wraps=cached._compress_batch_clips_tensor,
        ) as cached_compress, mock.patch.object(
            reference,
            "_compress_batch_clips_tensor",
            wraps=reference._compress_batch_clips_tensor,
        ) as reference_compress:
            for clip_id, feature in enumerate(features):
                torch.testing.assert_close(
                    step(cached, feature, clip_id),
                    step(reference, feature, clip_id),
                    rtol=1e-5,
                    atol=1e-6,
                )
            self.assertEqual(cached_compress.call_count, 3)
            self.assertEqual(reference_compress.call_count, 3)
            self.assertEqual(
                sum(call.args[0].shape[0] for call in cached_compress.call_args_list),
                3,
            )
            self.assertEqual(
                sum(
                    call.args[0].shape[0]
                    for call in reference_compress.call_args_list
                ),
                6,
            )

            before = cached.grounded_history_memory["episode-a"][2].clone()
            step(cached, features[-1], 2)
            self.assertEqual(cached_compress.call_count, 3)
            torch.testing.assert_close(
                cached.grounded_history_memory["episode-a"][2], before
            )

    def test_fifo_removes_raw_and_grounded_entries_together(self):
        _, cached = self.make_pair(consolidation="fifo", max_frames=2)
        for clip_id in range(3):
            step(cached, torch.randn(4, 8), clip_id)

        self.assertEqual(list(cached.history_memory["episode-a"]), [1, 2])
        self.assertEqual(
            list(cached.grounded_history_memory["episode-a"]), [1, 2]
        )

    def test_frme_recomputes_fused_entry_from_raw_feature(self):
        reference, cached = self.make_pair(consolidation="frme", max_frames=2)
        base = torch.randn(4, 8)
        features = [base, base + 0.02, -base]

        with mock.patch.object(
            cached,
            "_compress_batch_clips_tensor",
            wraps=cached._compress_batch_clips_tensor,
        ) as compress:
            for clip_id, feature in enumerate(features):
                cached_out = step(cached, feature, clip_id)
                reference_out = step(reference, feature, clip_id)

            torch.testing.assert_close(
                cached_out, reference_out, rtol=1e-5, atol=1e-6
            )
            self.assertEqual(compress.call_count, 3)

        raw_bank = cached.history_memory["episode-a"]
        grounded = cached.grounded_history_memory["episode-a"]
        self.assertEqual(list(raw_bank), list(grounded))
        self.assertEqual(len(raw_bank), 2)
        self.assertNotIn(1, raw_bank)
        torch.testing.assert_close(raw_bank[0][2], 0.5 * (features[0] + features[1]))

    def test_training_uses_reference_path_and_does_not_populate_eval_cache(self):
        module = ToMeMemoryBank(
            make_memory_config(),
            make_inference_config(True),
        )
        module.train()

        with mock.patch.object(
            module.retrieval_blocks[0].attn,
            "forward",
            wraps=module.retrieval_blocks[0].attn.forward,
        ) as attention:
            step(module, torch.randn(4, 8), 0)

        self.assertEqual(module.grounded_history_memory, {})
        self.assertTrue(attention.call_args.kwargs["need_weights"])

    def test_reference_eval_keeps_attention_weights_enabled(self):
        module = ToMeMemoryBank(
            make_memory_config(),
            make_inference_config(False),
        )
        module.eval()

        with mock.patch.object(
            module.retrieval_blocks[0].attn,
            "forward",
            wraps=module.retrieval_blocks[0].attn.forward,
        ) as attention:
            step(module, torch.randn(4, 8), 0)

        self.assertTrue(attention.call_args.kwargs["need_weights"])

    def test_mode_and_dtype_changes_invalidate_derived_cache(self):
        module = ToMeMemoryBank(
            make_memory_config(),
            make_inference_config(True),
        )
        module.eval()
        step(module, torch.randn(4, 8), 0)
        self.assertTrue(module.grounded_history_memory)

        module.train()
        self.assertEqual(module.grounded_history_memory, {})
        self.assertEqual(module._grounded_cache_meta, {})

        module.eval()
        step(module, torch.randn(4, 8), 1)
        self.assertTrue(module.grounded_history_memory)
        module.double()
        self.assertEqual(module.grounded_history_memory, {})
        self.assertEqual(module._grounded_cache_meta, {})

        output = step(module, torch.randn(4, 8, dtype=torch.float64), 2)
        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(
            all(
                value.dtype == torch.float64
                for value in module.grounded_history_memory["episode-a"].values()
            )
        )

    def test_frme_only_rebuilds_fused_and_new_entries(self):
        _, cached = self.make_pair(consolidation="frme", max_frames=3)
        base = torch.randn(4, 8)
        features = [base, base.clone(), -base, torch.randn(4, 8)]
        for clip_id, feature in enumerate(features[:3]):
            step(cached, feature, clip_id)

        untouched_ptr = cached.grounded_history_memory["episode-a"][2].data_ptr()
        with mock.patch.object(
            cached,
            "_compress_batch_clips_tensor",
            wraps=cached._compress_batch_clips_tensor,
        ) as compress:
            step(cached, features[3], 3)

        self.assertEqual(list(cached.history_memory["episode-a"]), [0, 2, 3])
        self.assertEqual(
            list(cached.grounded_history_memory["episode-a"]), [0, 2, 3]
        )
        self.assertEqual(
            cached.grounded_history_memory["episode-a"][2].data_ptr(),
            untouched_ptr,
        )
        self.assertEqual(compress.call_count, 1)
        self.assertEqual(compress.call_args.args[0].shape[0], 2)

    def test_optimized_eval_disables_attention_weights_and_cleanup_is_symmetric(self):
        module = ToMeMemoryBank(
            make_memory_config(),
            make_inference_config(True),
        )
        module.eval()

        with mock.patch.object(
            module.retrieval_blocks[0].attn,
            "forward",
            wraps=module.retrieval_blocks[0].attn.forward,
        ) as attention:
            step(module, torch.randn(4, 8), 0)

        self.assertFalse(attention.call_args.kwargs["need_weights"])
        step(module, torch.randn(4, 8), 0, episode="episode-b")
        self.assertNotIn("episode-a", module.history_memory)
        self.assertNotIn("episode-a", module.grounded_history_memory)

        module.clear_inference_cache("episode-b")
        self.assertNotIn("episode-b", module.grounded_history_memory)


if __name__ == "__main__":
    unittest.main()
