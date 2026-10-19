#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Deterministic noisy speckle sequences with exact sub-pixel truth.

Standard-library Python only, like generate_synthetic_dic.py. That generator
shifts a noise-free image by whole pixels, so FFTCC already lands on the
answer and ICGN stops after one iteration; nothing downstream of the first
iteration is exercised (review 2026-10-03, section 2.4). This one renders

    I_k(x') = I_ref(X),   x' = X + u_k(X)

for an affine displacement field about the image centre (xc, yc),

    u(X, Y) = u0 + ux * (X - xc) + uy * (Y - yc)
    v(X, Y) = v0 + vx * (X - xc) + vy * (Y - yc)

where I_ref is a sum of Gaussian speckles, evaluated analytically at the
back-projected position of every target pixel. No image is interpolated, so
the displacement and its four gradients are exact at any sub-pixel value, and
a first-order (ICGN-1) subset shape function represents the field without
model error. Each image is then scaled to a chosen grey-level contrast,
given independent Gaussian sensor noise and quantised to 8 bit.

Everything is seeded (random.Random), so one interpreter reproduces the same
bytes on every run. Bytes are not promised across platforms: math.exp may
differ in the last place between C libraries and flip the rounding of a pixel.
Tests built on these images must therefore assert accuracy against the truth
in truth.json, not hashes; the SHA-256 values written there are provenance.

    python3 smoke/generate_subpixel_speckle.py --out-dir DIR \
        [--width 192 --height 144] [--motion u0,v0,ux,uy,vx,vy ...] \
        [--contrast 40 --noise 2] [--seed N]

writes DIR/ref.pgm, DIR/tar_000.pgm ..., DIR/tar_list.txt (multi-frame list
with frame_000000.csv ... output names) and DIR/truth.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

SPECKLE_SIGMA = 1.6      # px; about 3.8 px full width at half maximum
SPECKLE_DENSITY = 0.08   # speckles per px^2
REACH_SIGMAS = 4.0       # a speckle is rendered out to exp(-8) = 3.4e-4 of its peak
DEFAULT_SEED = 20261005

Motion = tuple  # (u0, v0, ux, uy, vx, vy)


def centre(width: int, height: int) -> tuple[float, float]:
    return (width - 1) / 2.0, (height - 1) / 2.0


def displacement(motion: Motion, width: int, height: int, x: float, y: float) -> tuple[float, float]:
    """True (u, v) in px of the material point at reference pixel (x, y)."""
    u0, v0, ux, uy, vx, vy = motion
    xc, yc = centre(width, height)
    return (u0 + ux * (x - xc) + uy * (y - yc), v0 + vx * (x - xc) + vy * (y - yc))


def make_speckles(seed: int, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
                  density: float = SPECKLE_DENSITY) -> list[tuple[float, float, float]]:
    rng = random.Random(seed)
    count = int(round(density * (x_hi - x_lo) * (y_hi - y_lo)))
    return [(rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), rng.uniform(0.5, 1.0))
            for _ in range(count)]


def render(width: int, height: int, speckles, motion: Motion,
           sigma: float = SPECKLE_SIGMA) -> list[float]:
    """The speckle field carried by `motion`, sampled at the target pixel centres."""
    u0, v0, ux, uy, vx, vy = motion
    xc, yc = centre(width, height)
    a, b, c, d = 1.0 + ux, uy, vx, 1.0 + vy
    det = a * d - b * c
    if det <= 0.5:
        raise ValueError("the displacement gradients are far outside the small-strain range")
    # Reference offset of target pixel (x', y'): [X - xc, Y - yc] = Ainv [x' - xc - u0, y' - yc - v0]
    i00, i01, i10, i11 = d / det, -b / det, -c / det, a / det
    reach = REACH_SIGMAS * sigma * (1.0 + abs(ux) + abs(uy) + abs(vx) + abs(vy)) + 0.5
    scale = -0.5 / (sigma * sigma)
    exp = math.exp
    image = [0.0] * (width * height)
    for sx, sy, amplitude in speckles:
        tx = sx + u0 + ux * (sx - xc) + uy * (sy - yc)
        ty = sy + v0 + vx * (sx - xc) + vy * (sy - yc)
        x0, x1 = max(0, math.ceil(tx - reach)), min(width - 1, math.floor(tx + reach))
        y0, y1 = max(0, math.ceil(ty - reach)), min(height - 1, math.floor(ty + reach))
        if x0 > x1 or y0 > y1:
            continue
        columns = range(x0, x1 + 1)
        px0 = -xc - u0
        for y in range(y0, y1 + 1):
            q = y - yc - v0
            ex = i00 * px0 + i01 * q + xc - sx   # X - sx = i00 * x' + ex
            ey = i10 * px0 + i11 * q + yc - sy   # Y - sy = i10 * x' + ey
            lo = y * width + x0
            hi = lo + (x1 - x0 + 1)
            image[lo:hi] = [value + amplitude * exp(scale * ((i00 * x + ex) ** 2 + (i10 * x + ey) ** 2))
                            for value, x in zip(image[lo:hi], columns)]
    return image


def quantize(image: list[float], mean: float, std: float, contrast: float,
             noise_sigma: float, rng: random.Random) -> bytearray:
    """8-bit grey levels: mean 128, standard deviation `contrast`, plus Gaussian noise."""
    gain = contrast / std
    gauss = rng.gauss
    out = bytearray(len(image))
    for i, value in enumerate(image):
        gray = 128.0 + gain * (value - mean) + gauss(0.0, noise_sigma)
        out[i] = 0 if gray <= 0.0 else 255 if gray >= 255.0 else int(gray + 0.5)
    return out


def write_pgm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)


def build_sequence(root: Path, width: int, height: int, motions: list[Motion], *,
                   contrast: float = 40.0, noise_sigma: float = 2.0,
                   seed: int = DEFAULT_SEED) -> dict:
    """Write ref.pgm, tar_NNN.pgm, tar_list.txt and truth.json under `root`; return the truth."""
    root.mkdir(parents=True, exist_ok=True)
    shift = max((abs(m[0]) + abs(m[1]) for m in motions), default=0.0)
    strain = max((sum(abs(g) for g in m[2:]) for m in motions), default=0.0)
    pad = REACH_SIGMAS * SPECKLE_SIGMA + 2.0 + shift + strain * max(width, height)
    speckles = make_speckles(seed, -pad, width + pad, -pad, height + pad)

    still = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reference = render(width, height, speckles, still)
    mean = sum(reference) / len(reference)
    std = math.sqrt(sum((value - mean) ** 2 for value in reference) / len(reference))
    pixels = quantize(reference, mean, std, contrast, noise_sigma, random.Random(seed + 100))
    write_pgm(root / "ref.pgm", width, height, pixels)
    truth = {
        "generator": "generate_subpixel_speckle.py",
        "width": width, "height": height, "centre": list(centre(width, height)),
        "seed": seed, "speckle_sigma_px": SPECKLE_SIGMA, "speckle_density": SPECKLE_DENSITY,
        "speckles": len(speckles), "contrast_grey_levels": contrast,
        "noise_sigma_grey_levels": noise_sigma,
        "motion_fields": ["u0", "v0", "ux", "uy", "vx", "vy"],
        "reference": {"file": "ref.pgm", "sha256": hashlib.sha256(bytes(pixels)).hexdigest()},
        "frames": [],
    }
    lines = []
    for k, motion in enumerate(motions):
        image = render(width, height, speckles, motion)
        pixels = quantize(image, mean, std, contrast, noise_sigma, random.Random(seed + 101 + k))
        name = f"tar_{k:03d}.pgm"
        write_pgm(root / name, width, height, pixels)
        output_name = f"frame_{k:06d}.csv"
        lines.append(f"{root / name} {output_name}\n")
        truth["frames"].append({"file": name, "output_name": output_name, "motion": list(motion),
                                "sha256": hashlib.sha256(bytes(pixels)).hexdigest()})
    (root / "tar_list.txt").write_text("".join(lines))
    (root / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    return truth


def parse_motion(text: str) -> Motion:
    values = [float(part) for part in text.split(",")]
    if len(values) == 2:
        values += [0.0, 0.0, 0.0, 0.0]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("a motion is u0,v0 or u0,v0,ux,uy,vx,vy")
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--motion", type=parse_motion, action="append",
                        help="u0,v0[,ux,uy,vx,vy] of one target frame (px, 1/px); repeatable")
    parser.add_argument("--contrast", type=float, default=40.0,
                        help="grey-level standard deviation of the speckle pattern")
    parser.add_argument("--noise", type=float, default=2.0,
                        help="grey-level standard deviation of the sensor noise")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.width < 32 or args.height < 32:
        raise SystemExit("width and height must both be at least 32")
    motions = args.motion or [(0.37, -0.21, 0.002, -0.001, 0.0015, -0.0025)]
    build_sequence(args.out_dir, args.width, args.height, motions,
                   contrast=args.contrast, noise_sigma=args.noise, seed=args.seed)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
