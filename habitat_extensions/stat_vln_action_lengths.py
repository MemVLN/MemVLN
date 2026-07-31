#!/usr/bin/env python3
"""Summarize action and trajectory lengths for MP3D R2R and RxR-v2 data.

This utility intentionally uses only the Python standard library so it can be
run without importing the training stack.
"""

import argparse
import csv
import gzip
import json
import math
import os
import statistics
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data" / "default.yaml"
DEFAULT_THRESHOLDS = (50, 100, 200)
VLN_ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
}


class AnnotationFormatError(ValueError):
    """Raised when an annotation file does not have the expected structure."""


def merge_actions(
    actions, merge_move=True, merge_turn=True, move_unit=25, turn_unit=15
):
    """Merge consecutive VLN actions using the project's data-processing rule."""
    if not actions:
        return [], []

    previous = actions[0]
    cumulative_start_indices = []
    total_count = 0
    count = 1
    merged_actions = []

    def flush_action(action, action_count):
        if action == "stop":
            merged_actions.append(action)
        elif action == "move forward":
            merged_actions.append(f"{action} {action_count * move_unit} cm")
        else:
            merged_actions.append(f"{action} {action_count * turn_unit} degree")

    for action in actions[1:]:
        should_merge_move = (
            action == "move forward"
            and merge_move
            and action == previous
            and count < 3
        )
        should_merge_turn = (
            "turn" in action and merge_turn and action == previous and count < 3
        )
        if should_merge_move or should_merge_turn:
            count += 1
            continue

        cumulative_start_indices.append(total_count)
        total_count += count
        flush_action(previous, count)
        count = 1
        previous = action

    cumulative_start_indices.append(total_count)
    flush_action(previous, count)

    assert len(cumulative_start_indices) == len(merged_actions)
    return merged_actions, cumulative_start_indices


def _read_default_anno_dir(config_path):
    """Read the top-level annotation_base setting and return its vln directory."""
    config_path = Path(config_path).expanduser()
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OSError(
            f"Cannot read data config {config_path}. Pass --anno-dir to use an "
            "annotation directory directly."
        ) from exc

    annotation_base = None
    for line in lines:
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        key, separator, raw_value = line.partition(":")
        if separator and key.strip() == "annotation_base":
            annotation_base = raw_value.split(" #", 1)[0].strip().strip("'\"")
            break

    if not annotation_base:
        raise AnnotationFormatError(
            f"Could not find a non-empty top-level annotation_base in {config_path}"
        )

    expanded = Path(os.path.expandvars(annotation_base)).expanduser()
    if "$" in str(expanded):
        raise AnnotationFormatError(
            f"annotation_base in {config_path} contains an unresolved variable: "
            f"{annotation_base!r}"
        )
    if not expanded.is_absolute():
        expanded = REPO_ROOT / expanded
    return expanded.resolve() / "vln"


def _load_json(path, compressed=False):
    path = Path(path)
    try:
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as file:
                return json.load(file)
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Annotation file not found: {path}") from exc
    except (gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise AnnotationFormatError(
            f"Cannot decode annotation file {path}: {exc}"
        ) from exc


def _require_list(value, description):
    if not isinstance(value, list):
        raise AnnotationFormatError(
            f"{description} must be a JSON list, got {type(value).__name__}"
        )
    return value


def _required_field(item, field, description):
    if not isinstance(item, dict):
        raise AnnotationFormatError(
            f"{description} must be a JSON object, got {type(item).__name__}"
        )
    if field not in item:
        raise AnnotationFormatError(
            f"{description} is missing required field {field!r}"
        )
    return item[field]


def _action_names(action_ids, description):
    action_ids = _require_list(action_ids, f"{description} actions")
    actions = []
    for index, raw_action_id in enumerate(action_ids):
        if isinstance(raw_action_id, bool):
            raise AnnotationFormatError(
                f"{description} action at index {index} is boolean, not an action id"
            )
        try:
            action_id = int(raw_action_id)
        except (TypeError, ValueError) as exc:
            raise AnnotationFormatError(
                f"{description} action at index {index} is not an integer: "
                f"{raw_action_id!r}"
            ) from exc
        if isinstance(raw_action_id, float) and not raw_action_id.is_integer():
            raise AnnotationFormatError(
                f"{description} action at index {index} is not an integer: "
                f"{raw_action_id!r}"
            )
        if action_id not in VLN_ACTION_ID_TO_NAME:
            valid_ids = ", ".join(str(value) for value in VLN_ACTION_ID_TO_NAME)
            raise AnnotationFormatError(
                f"{description} has unknown action id {raw_action_id!r} at index "
                f"{index}; expected one of {valid_ids}"
            )
        actions.append(VLN_ACTION_ID_TO_NAME[action_id])
    return actions


def _percentile(values, percentile):
    """Return a linearly interpolated percentile, matching NumPy's default."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def _summarize(values, thresholds):
    if not values:
        return {}
    summary = OrderedDict()
    summary["n"] = len(values)
    summary["mean"] = float(statistics.fmean(values))
    summary["median"] = float(statistics.median(values))
    summary["min"] = min(values)
    summary["p75"] = _percentile(values, 75)
    summary["p90"] = _percentile(values, 90)
    summary["p95"] = _percentile(values, 95)
    summary["max"] = max(values)
    for threshold in thresholds:
        summary[f">{threshold}"] = (
            sum(value > threshold for value in values) / len(values) * 100.0
        )
    return summary


def _format_summary(name, values, thresholds):
    summary = _summarize(values, thresholds)
    if not summary:
        return f"{name}: no data"
    over_bits = "  ".join(
        f"{key}={value:.1f}%" for key, value in summary.items() if key.startswith(">")
    )
    return (
        f"{name}: n={summary['n']}  mean={summary['mean']:.2f}  "
        f"median={summary['median']:.1f}  min={summary['min']}  "
        f"p75={summary['p75']:.1f}  p90={summary['p90']:.1f}  "
        f"p95={summary['p95']:.1f}  max={summary['max']}  {over_bits}"
    )


def _format_capacity_pressure(name, values, capacities):
    if not values:
        return f"{name}: no data"

    parts = []
    for capacity in capacities:
        excess = [max(value - capacity, 0) for value in values]
        affected_excess = [value for value in excess if value > 0]
        affected_pct = len(affected_excess) / len(excess) * 100.0
        avg_excess_all = statistics.fmean(excess)
        avg_excess_affected = (
            statistics.fmean(affected_excess) if affected_excess else 0.0
        )
        parts.append(
            f"cap={capacity}: affected={affected_pct:.1f}%  "
            f"avg_excess/all={avg_excess_all:.2f}  "
            f"avg_excess/affected={avg_excess_affected:.2f}  "
            f"total_excess={sum(excess)}"
        )

    return f"{name}:\n  " + "\n  ".join(parts)


def _load_r2r(anno_dir, split):
    split_dir = Path(anno_dir) / "r2r" / split
    subset_path = split_dir / f"{split}_sub.json"
    if subset_path.exists():
        anno_path = subset_path
        anno_data = _load_json(anno_path)
    else:
        anno_path = split_dir / f"{split}.json.gz"
        payload = _load_json(anno_path, compressed=True)
        if not isinstance(payload, dict) or "episodes" not in payload:
            raise AnnotationFormatError(
                f"{anno_path} must be a JSON object containing an 'episodes' list"
            )
        anno_data = payload["episodes"]
    anno_data = _require_list(anno_data, f"{anno_path} episodes")

    gt_path = split_dir / f"{split}_gt.json.gz"
    gt_data = _load_json(gt_path, compressed=True)
    if not isinstance(gt_data, dict):
        raise AnnotationFormatError(
            f"{gt_path} must be a JSON object keyed by episode id"
        )

    records = []
    for index, item in enumerate(anno_data):
        description = f"{anno_path} episode at index {index}"
        episode_id = str(_required_field(item, "episode_id", description))
        trajectory_id = str(_required_field(item, "trajectory_id", description))
        if episode_id not in gt_data:
            raise AnnotationFormatError(
                f"{description} has episode_id {episode_id!r}, which is absent "
                f"from {gt_path}"
            )
        gt_description = f"{gt_path} episode {episode_id!r}"
        action_ids = _required_field(gt_data[episode_id], "actions", gt_description)
        actions = _action_names(action_ids, gt_description)
        merged_actions, _ = merge_actions(actions)
        records.append(
            {
                "dataset": "mp3d_r2r",
                "episode_id": f"r2r-{episode_id}",
                "trajectory_id": trajectory_id,
                "raw_len": len(actions),
                "merged_len": len(merged_actions),
            }
        )
    return records


def _load_rxr_v2(anno_dir, split):
    anno_path = Path(anno_dir) / "rxr" / f"{split}_v2" / "annotations.json"
    anno_data = _require_list(_load_json(anno_path), f"{anno_path} episodes")

    records = []
    for index, item in enumerate(anno_data):
        description = f"{anno_path} episode at index {index}"
        episode_id = str(_required_field(item, "episode_id", description))
        trajectory_id = str(_required_field(item, "trajectory_id", description))
        actions = _action_names(
            _required_field(item, "actions", description), description
        )
        merged_actions, _ = merge_actions(actions)
        records.append(
            {
                "dataset": "mp3d_rxr_v2",
                "episode_id": f"rxr-{episode_id}",
                "trajectory_id": trajectory_id,
                "raw_len": len(actions),
                "merged_len": len(merged_actions),
            }
        )
    return records


def _dedup_by_trajectory(records):
    seen = set()
    deduped = []
    for record in records:
        key = (record["dataset"], record["trajectory_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _print_dataset_stats(records, thresholds, title):
    print(f"\n== {title} ==")
    dataset_names = list(
        dict.fromkeys(record["dataset"] for record in records)
    )
    for dataset_name in dataset_names:
        subset = [record for record in records if record["dataset"] == dataset_name]
        print(f"\n[{dataset_name}]")
        print(
            _format_summary(
                "raw_actions", [record["raw_len"] for record in subset], thresholds
            )
        )
        merged_lengths = [record["merged_len"] for record in subset]
        print(_format_summary("merged_actions", merged_lengths, thresholds))
        print(
            _format_capacity_pressure(
                "merged_action_capacity_pressure", merged_lengths, thresholds
            )
        )


def _write_csv(records, output_path, overwrite=False):
    output_path = Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"CSV output already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("dataset", "episode_id", "trajectory_id", "raw_len", "merged_len")
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def _validate_thresholds(thresholds):
    if any(threshold <= 0 for threshold in thresholds):
        raise ValueError("--thresholds values must all be positive integers")
    return list(dict.fromkeys(thresholds))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compute action-length statistics for MP3D R2R and RxR-v2."
    )
    parser.add_argument(
        "--anno-dir",
        help=(
            "VLN annotation directory containing r2r/ and rxr/. If omitted, "
            "annotation_base/vln is read from --config."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_DATA_CONFIG),
        help=(
            "Data YAML used to find annotation_base when --anno-dir is omitted "
            f"(default: {DEFAULT_DATA_CONFIG})."
        ),
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)."
    )
    parser.add_argument(
        "--datasets",
        choices=("r2r", "rxr"),
        nargs="+",
        default=("r2r", "rxr"),
        help="Datasets to analyze (default: r2r rxr).",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Positive length thresholds for percentage summaries.",
    )
    parser.add_argument(
        "--csv",
        help=(
            "Optional path for per-episode CSV output; parent directories are "
            "created."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing --csv output.",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        thresholds = _validate_thresholds(args.thresholds)
        if not args.split.strip():
            raise ValueError("--split must not be empty")

        if args.anno_dir:
            anno_dir = Path(os.path.expandvars(args.anno_dir)).expanduser().resolve()
        else:
            anno_dir = _read_default_anno_dir(args.config)
        if not anno_dir.is_dir():
            raise FileNotFoundError(
                f"Annotation directory does not exist or is not a directory: {anno_dir}"
            )

        print(f"Annotation dir: {anno_dir}")
        print(f"Split: {args.split}")
        print(f"Datasets: {', '.join(args.datasets)}")
        print(f"Thresholds: {thresholds}")

        records = []
        if "r2r" in args.datasets:
            records.extend(_load_r2r(anno_dir, args.split))
        if "rxr" in args.datasets:
            records.extend(_load_rxr_v2(anno_dir, args.split))
        if not records:
            raise AnnotationFormatError(
                "The selected annotation files contain no episodes"
            )

        _print_dataset_stats(records, thresholds, "Episode-level records")
        _print_dataset_stats(
            _dedup_by_trajectory(records), thresholds, "Unique trajectory records"
        )

        if args.csv:
            output_path = _write_csv(
                records,
                args.csv,
                overwrite=args.overwrite,
            )
            print(f"\nSaved per-episode CSV to {output_path}")
    except (AnnotationFormatError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
