"""Filter RxR episodes by language and emit R2R-compatible gzip JSON.

By default the CLI selects English (including tags such as ``en-US``), replaces
each RxR instruction object with the two fields expected by Habitat's R2R
loader, and writes an empty vocabulary.  Other top-level dataset metadata is
preserved.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional


EMPTY_INSTRUCTION_VOCAB = {
    "word_list": [],
    "word2idx_dict": {},
    "stoi": {},
    "itos": [],
    "num_vocab": 0,
    "UNK_INDEX": 1,
    "PAD_INDEX": 0,
}


def normalize_language_tag(tag: str) -> str:
    """Normalize a BCP-47-style language tag for matching."""

    normalized = tag.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("Language tags must not be empty")
    return normalized


def instruction_languages(value: Any, episode_index: int) -> list[str]:
    """Return normalized language tags from an RxR instruction."""

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(
            f"Episode {episode_index} instruction.language must be a string "
            "or a sequence of strings"
        )

    if not all(isinstance(item, str) for item in values):
        raise TypeError(
            f"Episode {episode_index} instruction.language contains a "
            "non-string value"
        )
    return [normalize_language_tag(item) for item in values]


def language_matches(candidate: str, requested: set[str]) -> bool:
    """Match an exact tag or a regional variant of a requested base tag."""

    return any(
        candidate == language or candidate.startswith(f"{language}-")
        for language in requested
    )


def convert_dataset(
    dataset: Mapping[str, Any],
    languages: Sequence[str] = ("en",),
) -> tuple[dict[str, Any], int]:
    """Filter and convert an in-memory RxR dataset.

    Returns the converted dataset and the number of episodes removed.
    """

    raw_episodes = dataset.get("episodes")
    if not isinstance(raw_episodes, list):
        raise TypeError("Input JSON must contain an 'episodes' list")

    requested_languages = {normalize_language_tag(tag) for tag in languages}
    if not requested_languages:
        raise ValueError("At least one language must be requested")

    converted_episodes: list[dict[str, Any]] = []
    for index, episode in enumerate(raw_episodes):
        if not isinstance(episode, Mapping):
            raise TypeError(f"Episode {index} must be a JSON object")

        instruction = episode.get("instruction")
        if not isinstance(instruction, Mapping):
            raise TypeError(f"Episode {index} must contain an instruction object")
        if "language" not in instruction:
            raise KeyError(f"Episode {index} instruction has no 'language' field")

        candidate_languages = instruction_languages(
            instruction["language"], index
        )
        if not any(
            language_matches(candidate, requested_languages)
            for candidate in candidate_languages
        ):
            continue

        instruction_text = instruction.get("instruction_text")
        if not isinstance(instruction_text, str):
            raise TypeError(
                f"Episode {index} instruction_text must be a string"
            )

        converted_episode = dict(episode)
        converted_episode["instruction"] = {
            "instruction_text": instruction_text,
            "instruction_tokens": [],
        }
        converted_episodes.append(converted_episode)

    converted_dataset = dict(dataset)
    converted_dataset["episodes"] = converted_episodes
    converted_dataset["instruction_vocab"] = dict(EMPTY_INSTRUCTION_VOCAB)
    return converted_dataset, len(raw_episodes) - len(converted_episodes)


def load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        dataset = json.load(source)
    if not isinstance(dataset, dict):
        raise TypeError("Input JSON root must be an object")
    return dataset


def write_gzip_json(
    dataset: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool,
    indent: Optional[int],
) -> None:
    """Atomically write gzip JSON without clobbering output by default."""

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as output:
            json.dump(
                dataset,
                output,
                indent=indent,
                ensure_ascii=False,
            )
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an RxR gzip JSON dataset by language and convert its "
            "instructions to the R2R-compatible schema."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input RxR .json.gz file (for example, train_guide.json.gz).",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination .json.gz file.",
    )
    parser.add_argument(
        "--language",
        dest="languages",
        action="append",
        metavar="TAG",
        help=(
            "Language tag to retain; repeat for multiple languages. A base tag "
            "also matches regional variants. Defaults to en."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Optional JSON indentation inside the gzip file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_path if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input_path.expanduser()
    output_path = args.output_path.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if args.indent is not None and args.indent < 0:
        raise ValueError("--indent must be non-negative")

    languages = args.languages or ["en"]
    source_dataset = load_gzip_json(input_path)
    converted_dataset, removed_count = convert_dataset(
        source_dataset, languages
    )
    write_gzip_json(
        converted_dataset,
        output_path,
        overwrite=args.overwrite,
        indent=args.indent,
    )

    kept_count = len(converted_dataset["episodes"])
    print(
        f"Wrote {kept_count} episodes to {output_path} "
        f"({removed_count} filtered out)"
    )


if __name__ == "__main__":
    main()
