# -*- coding: utf-8 -*-
"""
Export images from TensorBoard event files without requiring the tensorboard package.

Uses tensorboardX proto definitions (already a project dependency) to parse
TFRecord event files and extract logged images by tag and/or step.
"""

import io
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

from PIL import Image


def _iter_raw_events(event_file: Path) -> Iterator[bytes]:
    """Yield raw proto bytes from a TFRecord event file.

    TFRecord wire format per record:
        uint64 length          (little-endian)
        uint32 masked_crc32c   (of the length bytes — skipped for reading)
        bytes  data[length]
        uint32 masked_crc32c   (of data — skipped for reading)
    """
    with open(event_file, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # skip crc of length
            data = f.read(length)
            if len(data) < length:
                break
            f.read(4)  # skip crc of data
            yield data


def _parse_image_events(
    event_file: Path,
) -> Iterator[Tuple[str, int, Image.Image]]:
    """Yield (tag, step, PIL.Image) for every image summary in an event file."""
    from tensorboardX.proto.event_pb2 import Event

    for raw in _iter_raw_events(event_file):
        event = Event()
        event.ParseFromString(raw)
        if not event.HasField("summary"):
            continue
        step = event.step
        for value in event.summary.value:
            if value.HasField("image"):
                img = Image.open(io.BytesIO(value.image.encoded_image_string))
                yield value.tag, step, img


def find_event_files(log_dir: Path) -> List[Path]:
    """Return all TFRecord event files found recursively under log_dir."""
    return sorted(log_dir.rglob("events.out.tfevents.*"))


def list_image_tags(log_dir: Path) -> Set[str]:
    """Return all image tags present in event files under log_dir."""
    tags: Set[str] = set()
    for event_file in find_event_files(log_dir):
        for tag, _step, _img in _parse_image_events(event_file):
            tags.add(tag)
    return tags


def parse_steps_filter(steps_str: Optional[str]) -> Optional[Set[int]]:
    """Parse --steps argument into a set of allowed step values.

    Accepts comma-separated integers and/or ranges:
        "5"        → {5}
        "0-20"     → {0, 1, …, 20}
        "0,10,20"  → {0, 10, 20}
        "0-5,10"   → {0, 1, 2, 3, 4, 5, 10}

    Returns None when steps_str is None, meaning all steps are allowed.
    """
    if steps_str is None:
        return None
    allowed: Set[int] = set()
    for part in steps_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            allowed.update(range(int(lo), int(hi) + 1))
        else:
            allowed.add(int(part))
    return allowed


def export_images(
    log_dir: Path,
    output_dir: Path,
    tags: Optional[List[str]] = None,
    steps: Optional[str] = None,
) -> int:
    """Export images from TensorBoard event files to PNG files.

    Args:
        log_dir: Root directory to search recursively for event files.
        output_dir: Destination directory for exported PNGs.
        tags: List of image tags to export. ``None`` exports all tags.
        steps: Step filter string (see :func:`parse_steps_filter`).
            ``None`` exports all steps.

    Returns:
        Number of images exported.
    """
    tag_filter: Optional[Set[str]] = set(tags) if tags else None
    step_filter = parse_steps_filter(steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for event_file in find_event_files(log_dir):
        for tag, step, img in _parse_image_events(event_file):
            if tag_filter is not None and tag not in tag_filter:
                continue
            if step_filter is not None and step not in step_filter:
                continue
            safe_tag = tag.replace("/", "_").replace(" ", "_")
            out_path = output_dir / f"{safe_tag}_step{step:06d}.png"
            img.save(out_path)
            count += 1

    return count
