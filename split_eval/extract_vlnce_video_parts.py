#!/usr/bin/env python3
"""Split videos produced by evaluator/vlnce_eval.py into their rendered parts.

The evaluator builds each video frame as:

1. ``observations_to_image({"rgb": ...}, info)``: RGB observation on the left,
   top-down map on the right.
2. ``addtext(...)``: a 150-pixel white text band appended below the image area.

This script reverses that rendering layout and saves four semantic crops:
observation, top-down map, instruction, and action. By default it also saves
PNG frames for each crop. The combined text band can be exported explicitly
with ``--include-text``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_TEXT_HEIGHT = 150
DEFAULT_OBSERVATION_WIDTH = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse-split vlnce_eval MP4s into observation, topdown map, "
            "instruction, and action crops."
        ),
    )
    parser.add_argument("videos", nargs="*", type=Path, help="Input MP4 path(s).")
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Text file with one MP4 path per line. Relative paths are resolved "
            "from the manifest directory; blank lines and # comments are ignored."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional shared output root. When omitted, each video is written under "
            "<epoch_dir>/video_parts/<video_stem>/."
        ),
    )
    parser.add_argument(
        "--obs-width",
        type=int,
        default=DEFAULT_OBSERVATION_WIDTH,
        help="RGB observation crop width. R2R/RxR Habitat configs in this repo use 640.",
    )
    parser.add_argument(
        "--text-height",
        type=int,
        default=DEFAULT_TEXT_HEIGHT,
        help="Bottom text band height added by VLNCEEvaluator.addtext().",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Only save MP4 crops and metadata, not per-frame PNG crops.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Also export the full combined instruction/action text band.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-video outputs and split_summary.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect inputs and print exact planned outputs without writing them.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[Path]:
    videos: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            video = Path(line).expanduser()
            if not video.is_absolute():
                video = path.parent / video
            videos.append(video)
    return videos


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+ " + shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffprobe failed for {path}{suffix}") from exc
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": stream.get("nb_frames"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": stream.get("duration"),
    }


def output_dir_for(video: Path, output_root: Optional[Path]) -> Path:
    if output_root is None:
        return video.parent.parent / "video_parts" / video.stem

    dataset, epoch = infer_dataset_epoch(video)
    if dataset and epoch:
        return output_root / dataset / epoch / video.stem
    return output_root / video.stem


def infer_dataset_epoch(video: Path) -> tuple[Optional[str], Optional[str]]:
    # Expected path: .../eval_results/<dataset>/<epoch>/video/<stem>.mp4
    parts = video.parts
    try:
        idx = parts.index("eval_results")
    except ValueError:
        return None, None
    if len(parts) <= idx + 3:
        return None, None
    return parts[idx + 1], parts[idx + 2]


def parse_scene_episode(stem: str) -> tuple[Optional[str], Optional[str]]:
    match = re.match(r"(.+)_(\d+)$", stem)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def load_instruction(video: Path) -> Optional[str]:
    scene_id, episode_id = parse_scene_episode(video.stem)
    if scene_id is None or episode_id is None:
        return None

    results_path = video.parent.parent / "results.json"
    if not results_path.exists():
        return None

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("scene_id")) == scene_id and str(row.get("episode_id")) == episode_id:
                instruction = row.get("episode_instruction")
                return str(instruction) if instruction is not None else None
    return None


def ffmpeg_crop(
    video: Path,
    out_file: Path,
    crop: dict[str, int],
    *,
    dry_run: bool,
    overwrite: bool,
) -> None:
    vf = f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"
    run(
        [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ],
        dry_run=dry_run,
    )


def ffmpeg_frames(
    video: Path,
    frames_dir: Path,
    crop: dict[str, int],
    *,
    dry_run: bool,
    overwrite: bool,
) -> None:
    vf = f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"
    if not dry_run:
        frames_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            vf,
            str(frames_dir / "frame_%05d.png"),
        ],
        dry_run=dry_run,
    )


def extract_first_crop(
    video: Path,
    out_file: Path,
    crop: dict[str, int],
) -> None:
    vf = f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(out_file),
        ],
    )


def detect_action_top(text_image: Path, text_height: int) -> int:
    """Find the top of the last rendered text line in the bottom text band."""
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Action/instruction detection requires the Python packages numpy and Pillow."
        ) from exc

    with Image.open(text_image) as image:
        arr = np.asarray(image.convert("L"))

    # The band is white with black cv2 text. A loose threshold is robust to MP4
    # compression around glyph edges.
    dark = arr < 235
    min_dark_pixels = max(5, arr.shape[1] // 500)
    ink_rows = np.flatnonzero(dark.sum(axis=1) >= min_dark_pixels)
    if ink_rows.size == 0:
        return text_height // 2

    groups: list[tuple[int, int]] = []
    start = prev = int(ink_rows[0])
    for row in map(int, ink_rows[1:]):
        if row - prev <= 3:
            prev = row
            continue
        groups.append((start, prev))
        start = prev = row
    groups.append((start, prev))

    action_start, _ = groups[-1]
    action_top = max(0, action_start - 5)

    # H.264/yuv420 outputs need even crop dimensions, so keep the split even.
    action_top = max(2, (action_top // 2) * 2)
    if action_top >= text_height:
        return max(2, text_height - 2)
    return action_top


def validate_crops(crops: dict[str, dict[str, int]], frame_width: int, frame_height: int) -> None:
    for name, crop in crops.items():
        if crop["w"] <= 0 or crop["h"] <= 0:
            raise ValueError(f"{name} crop has non-positive size: {crop}")
        if crop["x"] < 0 or crop["y"] < 0:
            raise ValueError(f"{name} crop has negative origin: {crop}")
        if crop["x"] + crop["w"] > frame_width or crop["y"] + crop["h"] > frame_height:
            raise ValueError(
                f"{name} crop exceeds frame bounds {frame_width}x{frame_height}: {crop}"
            )


def write_json(
    path: Path,
    data: dict[str, Any],
    *,
    dry_run: bool,
    overwrite: bool = False,
) -> None:
    print(f"+ write {path}")
    if dry_run:
        return
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str, *, dry_run: bool) -> None:
    print(f"+ write {path}")
    if dry_run:
        return
    path.write_text(text, encoding="utf-8")


def prepare_output_dir(output_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        print(f"+ remove {output_dir}")
        if not dry_run:
            shutil.rmtree(output_dir)

    print(f"+ create {output_dir}")
    if not dry_run:
        output_dir.mkdir(parents=True)


def split_video(video: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not video.is_file():
        raise FileNotFoundError(f"Input video does not exist or is not a file: {video}")

    info = probe_video(video)
    frame_w = int(info["width"])
    frame_h = int(info["height"])
    upper_h = frame_h - args.text_height
    if upper_h <= 0:
        raise ValueError(f"text height {args.text_height} leaves no image area for {video}")
    if args.obs_width >= frame_w:
        raise ValueError(
            f"observation width {args.obs_width} is not smaller than frame width {frame_w}"
        )

    output_dir = output_dir_for(video, args.output_root)

    text_crop = {"x": 0, "y": upper_h, "w": frame_w, "h": args.text_height}
    observation_crop = {"x": 0, "y": 0, "w": args.obs_width, "h": upper_h}
    topdown_crop = {"x": args.obs_width, "y": 0, "w": frame_w - args.obs_width, "h": upper_h}

    # Dry runs still inspect one frame in a temporary directory so the printed
    # instruction/action split is the same as a real run. No output-tree files
    # are created.
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_text = Path(tmpdir) / "text_first_frame.png"
        extract_first_crop(video, sample_text, text_crop)
        action_top = detect_action_top(sample_text, args.text_height)

    instruction_crop = {"x": 0, "y": upper_h, "w": frame_w, "h": action_top}
    action_crop = {
        "x": 0,
        "y": upper_h + action_top,
        "w": frame_w,
        "h": args.text_height - action_top,
    }
    crops = {
        "observation": observation_crop,
        "topdown_map": topdown_crop,
        "instruction": instruction_crop,
        "action": action_crop,
    }
    if args.include_text:
        crops["text"] = text_crop
    validate_crops({**crops, "full_text_band": text_crop}, frame_w, frame_h)

    prepare_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    outputs: dict[str, Any] = {}
    for name, crop in crops.items():
        out_video = output_dir / f"{name}.mp4"
        ffmpeg_crop(
            video,
            out_video,
            crop,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        outputs[f"{name}_video"] = str(out_video)

        if not args.no_frames:
            frames_dir = output_dir / f"{name}_frames"
            ffmpeg_frames(
                video,
                frames_dir,
                crop,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
            outputs[f"{name}_frames"] = str(frames_dir)

    instruction = load_instruction(video)
    if instruction is not None:
        instruction_txt = output_dir / "instruction.txt"
        write_text(instruction_txt, instruction + "\n", dry_run=args.dry_run)
        outputs["instruction_txt"] = str(instruction_txt)

    metadata = {
        "source_video": str(video),
        "frame": info,
        "layout": {
            "source": (
                "evaluator/vlnce_eval.py and evaluator/vlnce_eval_split.py "
                "addtext()/observations_to_image()"
            ),
            "observation_width": args.obs_width,
            "text_height": args.text_height,
            "image_height": upper_h,
            "action_top_in_text_band": action_top,
            "full_text_band": text_crop,
            "full_text_band_exported": args.include_text,
        },
        "crops": crops,
        "outputs": outputs,
    }
    write_json(output_dir / "metadata.json", metadata, dry_run=args.dry_run)
    return metadata


def dedupe(items: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for item in items:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def check_dependencies() -> None:
    missing_commands = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing_commands:
        joined = ", ".join(missing_commands)
        raise RuntimeError(f"Required command(s) not found on PATH: {joined}")

    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Required Python packages are missing. Install numpy and Pillow."
        ) from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.obs_width <= 0:
        raise ValueError("--obs-width must be greater than zero")
    if args.text_height <= 0:
        raise ValueError("--text-height must be greater than zero")


def summary_paths_for(
    videos: Iterable[Path],
    output_root: Optional[Path],
) -> set[Path]:
    if output_root is not None:
        return {output_root / "split_summary.json"}
    return {
        output_dir_for(video, output_root=None).parent / "split_summary.json"
        for video in videos
    }


def preflight_outputs(videos: Iterable[Path], args: argparse.Namespace) -> None:
    videos = list(videos)
    output_owners: dict[Path, Path] = {}
    for video in videos:
        if not video.is_file():
            raise FileNotFoundError(f"Input video does not exist or is not a file: {video}")

        output_dir = output_dir_for(video, args.output_root)
        resolved_output = output_dir.resolve()
        previous = output_owners.get(resolved_output)
        if previous is not None:
            raise ValueError(
                f"Inputs {previous} and {video} map to the same output directory: "
                f"{output_dir}"
            )
        output_owners[resolved_output] = video

        if output_dir.is_symlink():
            raise OSError(f"Refusing to replace an output-directory symlink: {output_dir}")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Pass --overwrite to replace it."
                )

    for summary_path in summary_paths_for(videos, args.output_root):
        if summary_path.is_symlink():
            raise OSError(f"Refusing to write summary through a symlink: {summary_path}")
        if summary_path.is_dir():
            raise IsADirectoryError(f"Summary path is a directory: {summary_path}")
        if summary_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Summary file already exists: {summary_path}. "
                "Pass --overwrite to replace it."
            )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        videos = list(args.videos)
        if args.manifest is not None:
            videos.extend(load_manifest(args.manifest))
        videos = dedupe(videos)

        if not videos:
            print("No videos provided. Use paths or --manifest.", file=sys.stderr)
            return 2

        check_dependencies()
        preflight_outputs(videos, args)

        all_metadata: list[dict[str, Any]] = []
        for idx, video in enumerate(videos, start=1):
            print(f"\n[{idx}/{len(videos)}] {video}")
            all_metadata.append(split_video(video, args))

        if args.output_root is not None:
            summary_path = args.output_root / "split_summary.json"
            if not args.dry_run:
                summary_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                summary_path,
                {"videos": all_metadata},
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        else:
            summary_groups: dict[Path, list[dict[str, Any]]] = {}
            for metadata in all_metadata:
                observation_video = Path(metadata["outputs"]["observation_video"])
                summary_dir = observation_video.parent.parent
                summary_groups.setdefault(summary_dir, []).append(metadata)
            for summary_dir, metadata_list in summary_groups.items():
                summary_path = summary_dir / "split_summary.json"
                if not args.dry_run:
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    summary_path,
                    {"videos": metadata_list},
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
