#!/usr/bin/env python3
"""Periodically print partial metrics from split-evaluation JSONL files."""

import argparse
import fcntl
import json
import time
from datetime import datetime
from pathlib import Path


METRICS = (
    "success",
    "oracle_success",
    "spl",
    "distance_to_goal",
    "ndtw",
    "steps",
)


def read_results(result_dir):
    files = sorted(result_dir.glob("results_split_*.json"))
    sums = {metric: 0.0 for metric in METRICS}
    counts = {metric: 0 for metric in METRICS}
    shard_counts = {}

    for path in files:
        shard_count = 0
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSON: {error.msg}"
                        ) from error

                    shard_count += 1
                    for metric in METRICS:
                        if metric in record:
                            sums[metric] += float(record[metric])
                            counts[metric] += 1
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        shard_counts[path.stem.removeprefix("results_split_")] = shard_count

    return files, sums, counts, shard_counts


def print_results(result_dir, expected_episodes=None):
    files, sums, counts, shard_counts = read_results(result_dir)
    episode_count = sum(shard_counts.values())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    progress = str(episode_count)
    if expected_episodes is not None:
        percentage = 100.0 * episode_count / expected_episodes
        progress = f"{episode_count}/{expected_episodes} ({percentage:.2f}%)"

    print(f"[{timestamp}] dir={result_dir}")
    print(f"files={len(files)} episodes={progress} shards={shard_counts}")
    if episode_count == 0:
        print("No completed episodes yet.", flush=True)
        return

    values = []
    for metric in METRICS:
        if counts[metric]:
            average = sums[metric] / counts[metric]
            values.append(f"{metric}={average:.4f} (n={counts[metric]})")
    print(" | ".join(values), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Watch partial R2R/RxR split-evaluation metrics."
    )
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.expected_episodes is not None and args.expected_episodes <= 0:
        parser.error("--expected-episodes must be positive")

    try:
        while True:
            print_results(args.dir, args.expected_episodes)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
