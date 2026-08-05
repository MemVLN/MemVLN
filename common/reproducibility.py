"""Deterministic reproducibility metadata for model training runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch.nn as nn
import transformers


_HF_SAFETENSORS_INDEX = "model.safetensors.index.json"
_HASH_CHUNK_SIZE = 1024 * 1024
_SUPPORTED_MODEL_FAMILIES = frozenset(("qwen3_vl", "qwen2_5_vl"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_artifact_file(
    artifact_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a fail-fast SHA256 descriptor for one local weight artifact."""
    path = Path(artifact_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Weight artifact does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "hash_algorithm": "sha256",
        "sha256": _sha256_file(path),
    }


def _load_unique_shard_names(index_path: Path) -> list[str]:
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid Hugging Face safetensors index JSON: {index_path}"
        ) from exc

    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(
            f"{index_path} must contain a non-empty object named weight_map."
        )
    invalid = [
        value
        for value in weight_map.values()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError(
            f"{index_path} contains invalid shard filenames in weight_map."
        )
    return sorted(set(weight_map.values()))


def _resolve_shard_path(checkpoint_root: Path, shard_name: str) -> Path:
    relative_path = Path(shard_name)
    if relative_path.is_absolute():
        raise ValueError(
            f"Shard paths must be relative to the checkpoint: {shard_name!r}."
        )
    root = checkpoint_root.resolve()
    shard_path = (root / relative_path).resolve()
    try:
        shard_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Shard path escapes the checkpoint directory: {shard_name!r}."
        ) from exc
    if not shard_path.is_file():
        raise FileNotFoundError(
            f"Referenced checkpoint shard does not exist: {shard_path}"
        )
    return shard_path


def hash_local_hf_checkpoint(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash every unique shard referenced by a local HF safetensors index.

    The aggregate digest is computed over a canonical JSON representation of
    the sorted shard name/hash/size records.  It is therefore independent of
    ``weight_map`` ordering and of the checkpoint directory's absolute path.
    """
    checkpoint_root = Path(checkpoint_path).expanduser()
    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Local Hugging Face checkpoint does not exist: {checkpoint_root}"
        )
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(
            f"Hugging Face checkpoint must be a directory: {checkpoint_root}"
        )

    index_path = checkpoint_root / _HF_SAFETENSORS_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Local Hugging Face checkpoint is missing {_HF_SAFETENSORS_INDEX}: "
            f"{checkpoint_root}"
        )

    shard_records = []
    for shard_name in _load_unique_shard_names(index_path):
        shard_path = _resolve_shard_path(checkpoint_root, shard_name)
        shard_records.append(
            {
                "filename": shard_name,
                "size_bytes": shard_path.stat().st_size,
                "sha256": _sha256_file(shard_path),
            }
        )

    fingerprint_payload = {
        "algorithm": "sha256",
        "shards": shard_records,
    }
    return {
        "source": "local",
        "path": str(checkpoint_root.resolve()),
        "index_file": _HF_SAFETENSORS_INDEX,
        "hash_algorithm": "sha256",
        "shard_count": len(shard_records),
        "shards": shard_records,
        "manifest_sha256": _canonical_sha256(fingerprint_payload),
    }


def describe_hf_checkpoint(
    model_revision_or_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Describe a local checkpoint or an unresolved Hugging Face Hub ID."""
    source_text = os.fspath(model_revision_or_path)
    if not source_text.strip():
        raise ValueError("model_revision_or_path must not be empty.")
    candidate = Path(source_text).expanduser()
    if candidate.exists():
        descriptor = hash_local_hf_checkpoint(candidate)
        descriptor["revision"] = revision
        return descriptor

    # An absent absolute or explicitly relative filesystem path is almost
    # certainly a configuration error, not a Hub model ID.
    if candidate.is_absolute() or source_text.startswith(("./", "../", "~")):
        raise FileNotFoundError(
            f"Configured local Hugging Face checkpoint does not exist: {candidate}"
        )
    return {
        "source": "huggingface_hub",
        "model_id": source_text,
        "revision": revision,
        "hash_status": "not_computed_for_remote_model_id",
    }


def _validate_memory_token_ids(memory_token_ids: Sequence[int]) -> tuple[int, int, int]:
    token_ids = tuple(memory_token_ids)
    if len(token_ids) != 3:
        raise ValueError(
            "Memory effective-parameter accounting requires exactly three "
            f"token IDs, got {len(token_ids)}."
        )
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        raise TypeError("Memory token IDs must be integers.")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"Memory token IDs must be non-negative, got {token_ids}.")
    if len(set(token_ids)) != 3:
        raise ValueError(f"Memory token IDs must be distinct, got {token_ids}.")
    return token_ids


def count_trainable_parameters(
    model: nn.Module,
    *,
    memory_token_ids: Sequence[int] = (),
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Count raw and effective trainable parameters.

    Gradient masking keeps a full embedding table in the optimizer even though
    only the three Memory rows can update.  When Memory token IDs are supplied,
    the effective count replaces that table's raw size with ``3 * hidden``.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}.")

    named_parameters = list(model.named_parameters())
    total_parameters = sum(parameter.numel() for _, parameter in named_parameters)
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    raw_trainable = sum(parameter.numel() for _, parameter in trainable)
    effective_trainable = raw_trainable
    memory_embedding = None

    if memory_token_ids:
        token_ids = _validate_memory_token_ids(memory_token_ids)
        embedding_candidates = [
            (name, parameter)
            for name, parameter in trainable
            if name.endswith("embed_tokens.weight") and parameter.ndim == 2
        ]
        if len(embedding_candidates) != 1:
            candidate_names = [name for name, _ in embedding_candidates]
            raise ValueError(
                "Expected exactly one trainable embed_tokens.weight for Memory "
                f"accounting, found {candidate_names}."
            )
        embedding_name, embedding_parameter = embedding_candidates[0]
        vocab_size, embedding_hidden_size = embedding_parameter.shape
        if any(token_id >= vocab_size for token_id in token_ids):
            raise ValueError(
                f"Memory token IDs {token_ids} exceed embedding vocab size "
                f"{vocab_size}."
            )
        if hidden_size is None:
            hidden_size = int(embedding_hidden_size)
        elif hidden_size != int(embedding_hidden_size):
            raise ValueError(
                f"Configured hidden_size={hidden_size} does not match Memory "
                f"embedding width {embedding_hidden_size}."
            )
        raw_embedding_count = embedding_parameter.numel()
        effective_embedding_count = len(token_ids) * hidden_size
        effective_trainable = (
            raw_trainable - raw_embedding_count + effective_embedding_count
        )
        memory_embedding = {
            "parameter_name": embedding_name,
            "token_ids": list(token_ids),
            "hidden_size": hidden_size,
            "raw_parameter_count": raw_embedding_count,
            "effective_parameter_count": effective_embedding_count,
        }

    return {
        "total_parameters": total_parameters,
        "raw_trainable_parameters": raw_trainable,
        "effective_trainable_parameters": effective_trainable,
        "memory_embedding": memory_embedding,
    }


def build_reproducibility_manifest(
    model: nn.Module,
    *,
    model_family: str,
    model_revision_or_path: str | os.PathLike[str],
    revision: str | None = None,
    memory_token_ids: Sequence[int] = (),
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable model and checkpoint reproducibility metadata."""
    normalized_family = str(model_family).strip().lower()
    if normalized_family not in _SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported model_family={model_family!r}; expected one of "
            f"{tuple(sorted(_SUPPORTED_MODEL_FAMILIES))}."
        )
    source_text = os.fspath(model_revision_or_path)
    return {
        "schema_version": 1,
        "transformers_version": transformers.__version__,
        "model_family": normalized_family,
        "model_revision_or_path": source_text,
        "revision": revision,
        "checkpoint": describe_hf_checkpoint(source_text, revision=revision),
        "parameter_counts": count_trainable_parameters(
            model,
            memory_token_ids=memory_token_ids,
            hidden_size=hidden_size,
        ),
    }


def write_json_atomic(
    output_path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> Path:
    """Atomically replace ``output_path`` with deterministic UTF-8 JSON."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}.")
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "hash_artifact_file",
    "hash_local_hf_checkpoint",
    "describe_hf_checkpoint",
    "count_trainable_parameters",
    "build_reproducibility_manifest",
    "write_json_atomic",
]
