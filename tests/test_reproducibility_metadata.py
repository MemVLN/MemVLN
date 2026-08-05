import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch.nn as nn
import transformers

from common.reproducibility import (
    build_reproducibility_manifest,
    count_trainable_parameters,
    describe_hf_checkpoint,
    hash_artifact_file,
    hash_local_hf_checkpoint,
    write_json_atomic,
)


def _write_index(root, weight_map):
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


class _TinyMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.projection = nn.Linear(4, 4)


class LocalCheckpointFingerprintTests(unittest.TestCase):
    def test_single_weight_artifact_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "StreamVGGT-1B.pth"
            artifact.write_bytes(b"stream-vggt")

            descriptor = hash_artifact_file(artifact)

            self.assertEqual(descriptor["size_bytes"], len(b"stream-vggt"))
            self.assertEqual(
                descriptor["sha256"],
                hashlib.sha256(b"stream-vggt").hexdigest(),
            )

    def test_unique_shards_are_sorted_hashed_and_aggregated_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.safetensors").write_bytes(b"z-shard")
            (root / "a.safetensors").write_bytes(b"a-shard")
            _write_index(
                root,
                {
                    "model.layer.1": "z.safetensors",
                    "model.layer.0": "a.safetensors",
                    "model.shared": "z.safetensors",
                },
            )

            first = hash_local_hf_checkpoint(root)
            self.assertEqual(first["shard_count"], 2)
            self.assertEqual(
                [record["filename"] for record in first["shards"]],
                ["a.safetensors", "z.safetensors"],
            )
            self.assertEqual(
                first["shards"][0]["sha256"],
                hashlib.sha256(b"a-shard").hexdigest(),
            )

            # Reordering the weight map must not change the aggregate digest.
            _write_index(
                root,
                {
                    "model.shared": "z.safetensors",
                    "model.layer.0": "a.safetensors",
                    "model.layer.1": "z.safetensors",
                },
            )
            second = hash_local_hf_checkpoint(root)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(first["shards"], second["shards"])

    def test_missing_index_or_referenced_shard_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "index"):
                hash_local_hf_checkpoint(root)

            _write_index(root, {"model.weight": "missing.safetensors"})
            with self.assertRaisesRegex(FileNotFoundError, "missing.safetensors"):
                hash_local_hf_checkpoint(root)

    def test_shard_path_cannot_escape_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            (root.parent / "outside.safetensors").write_bytes(b"outside")
            _write_index(root, {"model.weight": "../outside.safetensors"})

            with self.assertRaisesRegex(ValueError, "escapes"):
                hash_local_hf_checkpoint(root)


class RemoteCheckpointDescriptionTests(unittest.TestCase):
    def test_remote_model_id_does_not_claim_a_checkpoint_hash(self):
        descriptor = describe_hf_checkpoint(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            revision="fixed-commit",
        )

        self.assertEqual(descriptor["source"], "huggingface_hub")
        self.assertEqual(descriptor["revision"], "fixed-commit")
        self.assertEqual(
            descriptor["hash_status"],
            "not_computed_for_remote_model_id",
        )
        self.assertNotIn("manifest_sha256", descriptor)
        self.assertNotIn("shards", descriptor)

    def test_missing_absolute_local_path_is_not_treated_as_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-checkpoint"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                describe_hf_checkpoint(missing)

    def test_empty_model_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            describe_hf_checkpoint("")


class TrainableParameterCountTests(unittest.TestCase):
    def test_memory_embedding_uses_three_rows_for_effective_count(self):
        model = _TinyMemoryModel()

        counts = count_trainable_parameters(
            model,
            memory_token_ids=(7, 8, 9),
            hidden_size=4,
        )

        self.assertEqual(counts["total_parameters"], 60)
        self.assertEqual(counts["raw_trainable_parameters"], 60)
        self.assertEqual(counts["effective_trainable_parameters"], 32)
        self.assertEqual(
            counts["memory_embedding"],
            {
                "parameter_name": "embed_tokens.weight",
                "token_ids": [7, 8, 9],
                "hidden_size": 4,
                "raw_parameter_count": 40,
                "effective_parameter_count": 12,
            },
        )

    def test_plain_model_reports_raw_and_effective_counts_as_equal(self):
        model = _TinyMemoryModel()
        model.embed_tokens.weight.requires_grad = False

        counts = count_trainable_parameters(model)

        self.assertEqual(counts["total_parameters"], 60)
        self.assertEqual(counts["raw_trainable_parameters"], 20)
        self.assertEqual(counts["effective_trainable_parameters"], 20)
        self.assertIsNone(counts["memory_embedding"])

    def test_invalid_memory_embedding_contract_fails_fast(self):
        model = _TinyMemoryModel()
        with self.assertRaisesRegex(ValueError, "exactly three"):
            count_trainable_parameters(model, memory_token_ids=(8, 9))
        with self.assertRaisesRegex(ValueError, "vocab size"):
            count_trainable_parameters(model, memory_token_ids=(8, 9, 10))
        with self.assertRaisesRegex(ValueError, "embedding width"):
            count_trainable_parameters(
                model,
                memory_token_ids=(7, 8, 9),
                hidden_size=8,
            )


class ManifestAndAtomicWriteTests(unittest.TestCase):
    def test_manifest_collects_versions_checkpoint_and_parameter_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            (root / "model-1.safetensors").write_bytes(b"weights")
            _write_index(root, {"model.weight": "model-1.safetensors"})

            manifest = build_reproducibility_manifest(
                _TinyMemoryModel(),
                model_family="qwen2_5_vl",
                model_revision_or_path=root,
                revision="local-snapshot",
                memory_token_ids=(7, 8, 9),
                hidden_size=4,
            )

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["transformers_version"], transformers.__version__)
            self.assertEqual(manifest["model_family"], "qwen2_5_vl")
            self.assertEqual(manifest["model_revision_or_path"], str(root))
            self.assertEqual(manifest["revision"], "local-snapshot")
            self.assertEqual(manifest["checkpoint"]["source"], "local")
            self.assertIn("manifest_sha256", manifest["checkpoint"])
            self.assertEqual(
                manifest["parameter_counts"]["effective_trainable_parameters"],
                32,
            )

    def test_json_write_uses_atomic_replace_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "manifest.json"
            payload = {"z": 1, "a": {"unicode": "内存"}}

            with mock.patch(
                "common.reproducibility.os.replace",
                wraps=os.replace,
            ) as replace:
                returned = write_json_atomic(destination, payload)

            self.assertEqual(returned, destination)
            replace.assert_called_once()
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                payload,
            )
            self.assertTrue(destination.read_text(encoding="utf-8").endswith("\n"))
            leftovers = list(destination.parent.glob(".manifest.json.*.tmp"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
