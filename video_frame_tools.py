#!/usr/bin/env python3
"""Filter sudden brightest-point jumps or interpolate duplicate video frames."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTENSIONS = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".webm",
}

DEFAULT_FOURCC = {
    ".avi": "MJPG",
    ".mkv": "XVID",
    ".mov": "mp4v",
    ".mp4": "mp4v",
    ".m4v": "mp4v",
    ".mpg": "MPEG",
    ".mpeg": "MPEG",
    ".wmv": "WMV2",
    ".webm": "VP80",
}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        type=Path,
        help="Input video file or directory containing videos",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output video file or directory. Defaults to a separate output path.",
    )
    parser.add_argument(
        "--fourcc",
        help="Optional four-character output codec, e.g. MJPG or mp4v",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process videos by filtering sudden brightest-point movement or "
            "interpolating duplicate frames."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    filter_parser = subparsers.add_parser(
        "filter",
        help="Replace frames whose brightest point jumps too far",
        description=(
            "Replace an unstable frame with a neighboring kept frame when the "
            "brightest pixel moves farther than the configured distance."
        ),
    )
    add_common_arguments(filter_parser)
    filter_parser.add_argument(
        "-t",
        "--threshold",
        "--max-distance",
        dest="max_distance",
        type=float,
        default=5.0,
        help="Maximum brightest-point movement in pixels (default: 5)",
    )

    interpolate_parser = subparsers.add_parser(
        "interpolate",
        help="Interpolate frames identical to their preceding frame",
        description=(
            "Replace a frame identical to its preceding frame with the 50/50 "
            "pixel-wise blend of that frame and the following frame."
        ),
    )
    add_common_arguments(interpolate_parser)

    both_parser = subparsers.add_parser(
        "both",
        help="Filter brightest-point jumps, then interpolate duplicate frames",
        description=(
            "Apply brightest-point jump filtering first, then interpolate "
            "frames duplicated by the filtering step or present in the input."
        ),
    )
    add_common_arguments(both_parser)
    both_parser.add_argument(
        "-t",
        "--threshold",
        "--max-distance",
        dest="max_distance",
        type=float,
        default=5.0,
        help="Maximum brightest-point movement in pixels (default: 5)",
    )

    args = parser.parse_args()
    if args.mode in {"filter", "both"} and args.max_distance < 0:
        parser.error("--max-distance must be non-negative")
    if args.fourcc is not None and len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters")
    return args


def default_output(input_path: Path, mode: str) -> Path:
    suffix = {
        "filter": "filtered",
        "interpolate": "interpolated",
        "both": "processed",
    }[mode]
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_{suffix}"
    return input_path.with_name(f"{input_path.stem}_{suffix}{input_path.suffix}")


def discover_videos(input_path: Path, output_path: Path) -> list:
    output_resolved = output_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path]

    videos = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        resolved = path.resolve()
        if resolved == output_resolved or output_resolved in resolved.parents:
            continue
        videos.append(path)
    return videos


def output_for_video(input_path: Path, video_path: Path, output_path: Path) -> Path:
    if input_path.is_file():
        return output_path
    return output_path / video_path.relative_to(input_path)


def brightest_point(frame: np.ndarray) -> Tuple[int, int]:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row, column = np.unravel_index(np.argmax(grayscale), grayscale.shape)
    return int(column), int(row)


def read_frames(
    capture: cv2.VideoCapture,
    width: int,
    height: int,
) -> List[np.ndarray]:
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame.shape[:2] != (height, width):
            raise RuntimeError("Unexpected frame size while reading video")
        frames.append(frame)

    if not frames:
        raise RuntimeError("Video contains no readable frames")
    return frames


def transition_is_unstable(
    left: Tuple[int, int],
    right: Tuple[int, int],
    max_distance: float,
) -> bool:
    return float(np.hypot(right[0] - left[0], right[1] - left[1])) > max_distance


def filter_brightest_point_jumps(
    frames: List[np.ndarray],
    max_distance: float,
) -> Tuple[List[np.ndarray], int, str]:
    points = [brightest_point(frame) for frame in frames]
    unstable = [False] + [
        transition_is_unstable(points[index - 1], points[index], max_distance)
        for index in range(1, len(points))
    ]

    leading_run = 0
    for is_unstable in unstable[1:]:
        if not is_unstable:
            break
        leading_run += 1

    trailing_run = 0
    for is_unstable in reversed(unstable[1:]):
        if not is_unstable:
            break
        trailing_run += 1

    filtered = list(frames)
    replaced_count = 0
    kept_points = list(points)

    if leading_run >= 3 and trailing_run >= 3:
        strategy = "center-out"
        center = len(frames) // 2
        for index in range(center + 1, len(frames)):
            if transition_is_unstable(kept_points[index - 1], points[index], max_distance):
                filtered[index] = filtered[index - 1]
                kept_points[index] = kept_points[index - 1]
                replaced_count += 1
        for index in range(center - 1, -1, -1):
            if transition_is_unstable(points[index], kept_points[index + 1], max_distance):
                filtered[index] = filtered[index + 1]
                kept_points[index] = kept_points[index + 1]
                replaced_count += 1
    elif leading_run >= 3:
        strategy = "backward"
        for index in range(len(frames) - 2, -1, -1):
            if transition_is_unstable(points[index], kept_points[index + 1], max_distance):
                filtered[index] = filtered[index + 1]
                kept_points[index] = kept_points[index + 1]
                replaced_count += 1
    else:
        strategy = "forward"
        for index in range(1, len(frames)):
            if transition_is_unstable(kept_points[index - 1], points[index], max_distance):
                filtered[index] = filtered[index - 1]
                kept_points[index] = kept_points[index - 1]
                replaced_count += 1

    return filtered, replaced_count, strategy


def interpolate_duplicate_frames(
    frames: List[np.ndarray],
) -> Tuple[List[np.ndarray], int, int]:
    if not frames:
        raise RuntimeError("Video contains no readable frames")

    interpolated = list(frames)
    interpolated_count = 0
    boundary_duplicate_count = 0
    for index in range(1, len(frames) - 1):
        if np.array_equal(frames[index - 1], frames[index]):
            interpolated[index] = cv2.addWeighted(
                frames[index - 1], 0.5, frames[index + 1], 0.5, 0
            )
            interpolated_count += 1

    if len(frames) > 1 and np.array_equal(frames[-2], frames[-1]):
        boundary_duplicate_count = 1
    return interpolated, interpolated_count, boundary_duplicate_count


def process_video(
    source: Path,
    destination: Path,
    mode: str,
    max_distance: float,
    fourcc_override: Optional[str],
    overwrite: bool,
) -> Tuple[int, int, int, int, str]:
    if source.resolve() == destination.resolve():
        raise ValueError(f"Input and output paths are the same: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination} (pass --overwrite to replace it)"
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {source}")

    codec = fourcc_override or DEFAULT_FOURCC.get(destination.suffix.lower())
    if codec is None:
        capture.release()
        raise ValueError(
            f"No default codec for {destination.suffix}; specify --fourcc"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.stem}.partial{destination.suffix}"
    )
    if temporary.exists():
        if not overwrite:
            capture.release()
            raise FileExistsError(
                f"Temporary output already exists: {temporary} "
                "(pass --overwrite to replace it)"
            )
        temporary.unlink()

    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"Could not create output video with codec {codec}: {temporary}"
        )

    try:
        frames = read_frames(capture, width, height)
        replaced_count = 0
        strategy = "not applied"
        if mode in {"filter", "both"}:
            frames, replaced_count, strategy = filter_brightest_point_jumps(
                frames, max_distance
            )

        interpolated_count = 0
        boundary_count = 0
        if mode in {"interpolate", "both"}:
            frames, interpolated_count, boundary_count = (
                interpolate_duplicate_frames(frames)
            )

        for frame in frames:
            writer.write(frame)
        result = (
            len(frames),
            replaced_count,
            interpolated_count,
            boundary_count,
            strategy,
        )
    except Exception:
        capture.release()
        writer.release()
        if temporary.exists():
            temporary.unlink()
        raise

    capture.release()
    writer.release()
    os.replace(temporary, destination)
    return result


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file() and not input_path.is_dir():
        print(f"Error: input is not a file or directory: {input_path}", file=sys.stderr)
        return 2

    output_path = (
        args.output.expanduser() if args.output else default_output(input_path, args.mode)
    )
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()

    try:
        videos = discover_videos(input_path, output_path)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if not videos:
        print(f"No supported videos found in: {input_path}", file=sys.stderr)
        return 1

    errors = 0
    total_frames = 0
    total_replaced = 0
    total_interpolated = 0
    total_boundary_duplicates = 0
    strategies = {}
    for video_path in videos:
        destination = output_for_video(input_path, video_path, output_path)
        try:
            (
                frame_count,
                replaced_count,
                interpolated_count,
                boundary_count,
                strategy,
            ) = process_video(
                video_path,
                destination,
                args.mode,
                getattr(args, "max_distance", 5.0),
                args.fourcc,
                args.overwrite,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors += 1
            print(f"ERROR {video_path}: {error}", file=sys.stderr)
            continue

        total_frames += frame_count
        total_replaced += replaced_count
        total_interpolated += interpolated_count
        total_boundary_duplicates += boundary_count
        if args.mode in {"filter", "both"}:
            strategies[strategy] = strategies.get(strategy, 0) + 1
        details = []
        if args.mode in {"filter", "both"}:
            details.append(f"{replaced_count}/{frame_count} frames replaced")
            details.append(f"{strategy} scan")
        if args.mode in {"interpolate", "both"}:
            details.append(f"{interpolated_count} duplicates interpolated")
        message = f"{video_path} -> {destination}: " + ", ".join(details)
        if boundary_count:
            message += (
                f", {boundary_count} duplicate frame(s) at the video boundary "
                "could not be interpolated"
            )
        print(message)

    summary = f"Finished: {len(videos) - errors}/{len(videos)} videos"
    if args.mode in {"filter", "both"}:
        summary += f", {total_replaced}/{total_frames} frames replaced"
    if args.mode in {"interpolate", "both"}:
        summary += f", {total_interpolated} duplicates interpolated"
    if strategies:
        summary += ", scan directions: " + ", ".join(
            f"{strategy}={count}" for strategy, count in sorted(strategies.items())
        )
    if args.mode in {"interpolate", "both"}:
        summary += (
            f", {total_boundary_duplicates} boundary duplicate(s) left unchanged."
        )
    else:
        summary += "."
    print(summary)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
