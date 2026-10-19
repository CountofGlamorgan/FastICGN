#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Generate deterministic synthetic PGM inputs for FastICGN smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def write_pgm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)


def smooth(width: int, height: int, pixels: bytearray, passes: int) -> bytearray:
    current = pixels
    for _ in range(passes):
        out = bytearray(width * height)
        for y in range(height):
            for x in range(width):
                total = 0
                count = 0
                for yy, xx in (
                    (y, x),
                    (max(0, y - 1), x),
                    (min(height - 1, y + 1), x),
                    (y, max(0, x - 1)),
                    (y, min(width - 1, x + 1)),
                ):
                    total += current[yy * width + xx]
                    count += 1
                out[y * width + x] = total // count
        current = out
    return current


def shifted_image(width: int, height: int, src: bytearray, dx: int, dy: int, fill: int) -> bytearray:
    dst = bytearray([fill]) * (width * height)
    for y in range(height):
        sy = y - dy
        if sy < 0 or sy >= height:
            continue
        for x in range(width):
            sx = x - dx
            if 0 <= sx < width:
                dst[y * width + x] = src[sy * width + sx]
    return dst


def make_base(width: int, height: int, seed: int) -> bytearray:
    rng = random.Random(seed)
    pixels = bytearray(width * height)
    for idx in range(width * height):
        value = int(rng.gauss(127, 52))
        pixels[idx] = max(0, min(255, value))
    return smooth(width, height, pixels, passes=2)


def write_tar_list(path: Path, entries: list[tuple[Path, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for image_path, output_name in entries:
            handle.write(f"{image_path} {output_name}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    if args.width < 96 or args.height < 96:
        raise SystemExit("width and height must both be at least 96")
    if args.frames < 2:
        raise SystemExit("frames must be at least 2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = make_base(args.width, args.height, args.seed)
    write_pgm(out_dir / "ref.pgm", args.width, args.height, base)

    parity_shifts = [(i % 3, (i // 3) % 3) for i in range(args.frames)]
    parity_entries: list[tuple[Path, str]] = []
    for idx, (dx, dy) in enumerate(parity_shifts):
        image = shifted_image(args.width, args.height, base, dx, dy, fill=127)
        image_path = out_dir / f"frame_{idx:06d}.pgm"
        write_pgm(image_path, args.width, args.height, image)
        parity_entries.append((image_path, f"frame_{idx:06d}.csv"))

    tar0 = shifted_image(args.width, args.height, base, 0, 0, fill=127)
    tar_shift20 = shifted_image(args.width, args.height, base, 20, 0, fill=127)
    tar_oob_dx10 = shifted_image(args.width, args.height, base, 10, 0, fill=127)
    write_pgm(out_dir / "tar0.pgm", args.width, args.height, tar0)
    write_pgm(out_dir / "tar_shift20.pgm", args.width, args.height, tar_shift20)
    write_pgm(out_dir / "tar_oob_dx10.pgm", args.width, args.height, tar_oob_dx10)

    write_tar_list(out_dir / "tar_list_10.txt", parity_entries)
    write_tar_list(
        out_dir / "tar_list_fallback.txt",
        [
            (out_dir / "tar0.pgm", "frame_000000.csv"),
            (out_dir / "tar_shift20.pgm", "frame_000001.csv"),
        ],
    )

    metadata = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "seed": args.seed,
        "parity_shifts": parity_shifts,
        "fallback_shift": [20, 0],
        "oob_shift": [10, 0],
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(os.fspath(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
