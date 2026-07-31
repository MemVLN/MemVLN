"""Checkpoint-path contracts for standalone split evaluation."""

from pathlib import Path


def require_checkpoint_file(path):
    """Return an exact checkpoint file path or fail before model evaluation."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Split evaluation requires an exact checkpoint file, but none "
            f"exists at: {checkpoint_path}"
        )
    return checkpoint_path
