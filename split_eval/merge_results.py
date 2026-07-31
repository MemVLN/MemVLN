"""Validate and merge per-split VLN-CE JSONL result files."""

import argparse
import json
import math
import re
from pathlib import Path


CORE_METRICS = (
    "success",
    "oracle_success",
    "spl",
    "distance_to_goal",
)
OPTIONAL_METRICS = ("ndtw",)
REQUIRED_FIELDS = ("scene_id", "episode_id", "steps", *CORE_METRICS)
_SPLIT_FILE_PATTERN = re.compile(r"results_split_(\d+)\.json")


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def discover_split_files(target_dir, expected_splits=None):
    target_dir = Path(target_dir)
    split_paths = sorted(target_dir.glob("results_split_*.json"))
    if not split_paths:
        raise FileNotFoundError(
            f"No results_split_*.json files found in {target_dir}"
        )

    indexed_paths = {}
    for path in split_paths:
        match = _SPLIT_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid split result filename: {path}")
        split_index = int(match.group(1))
        if split_index in indexed_paths:
            raise ValueError(
                f"Duplicate split index {split_index}: "
                f"{indexed_paths[split_index]} and {path}"
            )
        indexed_paths[split_index] = path

    if expected_splits is not None:
        expected_indices = set(range(expected_splits))
        actual_indices = set(indexed_paths)
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)
            unexpected = sorted(actual_indices - expected_indices)
            raise ValueError(
                "Split set mismatch: "
                f"expected indices 0..{expected_splits - 1}, "
                f"missing={missing}, unexpected={unexpected}"
            )

    return [indexed_paths[index] for index in sorted(indexed_paths)]


def _record_location(path, line_number):
    return f"{path}:{line_number}"


def _validate_identifier(value, field, location):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"{location}: field {field!r} must be a string or integer"
        )
    if not str(value).strip():
        raise ValueError(f"{location}: field {field!r} must not be empty")


def _validate_finite_number(value, field, location):
    if isinstance(value, bool):
        if field not in {"success", "oracle_success"}:
            raise ValueError(f"{location}: field {field!r} must be numeric")
        value = int(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{location}: field {field!r} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{location}: field {field!r} must be finite")
    return value


def load_split_records(split_paths):
    records = []
    seen_episodes = {}

    for path in split_paths:
        split_records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                location = _record_location(path, line_number)
                if not line.strip():
                    raise ValueError(f"{location}: blank result line")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{location}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{location}: result must be a JSON object")

                missing = [
                    field for field in REQUIRED_FIELDS if field not in record
                ]
                if missing:
                    raise ValueError(
                        f"{location}: missing required fields: {missing}"
                    )

                _validate_identifier(record["scene_id"], "scene_id", location)
                _validate_identifier(
                    record["episode_id"], "episode_id", location
                )
                episode_key = (
                    str(record["scene_id"]),
                    str(record["episode_id"]),
                )
                if episode_key in seen_episodes:
                    raise ValueError(
                        f"{location}: duplicate episode {episode_key}; "
                        f"first seen at {seen_episodes[episode_key]}"
                    )
                seen_episodes[episode_key] = location

                for field in ("steps", *CORE_METRICS, *OPTIONAL_METRICS):
                    if field in record:
                        _validate_finite_number(record[field], field, location)

                steps = record["steps"]
                if (
                    isinstance(steps, bool)
                    or not isinstance(steps, int)
                    or steps < 0
                ):
                    raise ValueError(
                        f"{location}: field 'steps' must be a non-negative integer"
                    )

                split_records.append(record)
                records.append(record)

        if not split_records:
            raise ValueError(f"{path}: split result file is empty")

    for field in OPTIONAL_METRICS:
        present_count = sum(field in record for record in records)
        if present_count not in (0, len(records)):
            raise ValueError(
                f"Optional metric {field!r} is present for {present_count}/"
                f"{len(records)} episodes"
            )

    return records


def aggregate_records(records):
    if not records:
        raise ValueError("Cannot aggregate an empty result set")

    metric_fields = [*CORE_METRICS, "steps"]
    metric_fields.extend(
        field for field in OPTIONAL_METRICS if field in records[0]
    )
    averages = {
        field: sum(record[field] for record in records) / len(records)
        for field in metric_fields
    }
    averages["count"] = len(records)
    return averages


def validate_expected_episode_count(records, expected_episodes):
    if expected_episodes is not None and len(records) != expected_episodes:
        raise ValueError(
            f"Episode count mismatch: expected {expected_episodes}, "
            f"found {len(records)}"
        )


def merge_results(target_dir, expected_splits=None, expected_episodes=None):
    target_dir = Path(target_dir)
    split_paths = discover_split_files(target_dir, expected_splits)
    records = load_split_records(split_paths)
    validate_expected_episode_count(records, expected_episodes)

    averages = aggregate_records(records)
    output_path = target_dir / "final_merged.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(averages, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument(
        "--expected-splits",
        type=positive_int,
        help="Require the exact split index set 0..N-1.",
    )
    parser.add_argument(
        "--expected-episodes",
        type=positive_int,
        help="Require exactly this many unique episodes.",
    )
    args = parser.parse_args()

    averages = merge_results(
        args.dir,
        expected_splits=args.expected_splits,
        expected_episodes=args.expected_episodes,
    )
    print(f"Final: {averages}")


if __name__ == "__main__":
    main()
