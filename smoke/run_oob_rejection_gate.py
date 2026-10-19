#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Assertion gate for the partially-out-of-bounds rejection rule.

A warped target subset that leaves the target image's interpolatable range
rejects its POI with ``zncc = -3``; a subset entirely inside it is untouched by
the rule. Both are asserted here, on a synthetic pair whose target is the
reference shifted by a known whole-pixel amount, so the band of POIs the rule
must reject is known in closed form before the CLI runs.

The rule is backend-independent, so the same assertions hold for the CUDA
kernels and for the CPU ICGN backend; the gate runs whichever
``opencorr_cli_cuda_mvp`` it is given. Pass ``--cpu-binary`` as well to assert
that the two backends reject the identical set of POIs.

Interpolatable range (oc_cubic_bspline.cpp, src/icgn2d1_cuda_mvp.cu,
src/oc_icgn.cpp): a sample at (gx, gy) is available iff
``1 <= gx < width - 2`` and ``1 <= gy < height - 2``.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic_dic import make_base, shifted_image, write_pgm  # noqa: E402

RECORD = struct.Struct("<iifffffff")
MINUS3 = -3.0


def read_raw(path: Path) -> dict[tuple[int, int], tuple[float, ...]]:
    blob = path.read_bytes()
    if len(blob) % RECORD.size:
        raise SystemExit(f"{path}: {len(blob)} bytes is not a whole number of records")
    rows = {}
    for offset in range(0, len(blob), RECORD.size):
        x, y, u, v, ux, uy, vx, vy, zncc = RECORD.unpack_from(blob, offset)
        rows[(x, y)] = (u, v, zncc)
    return rows


def predicted_out_of_range(x: int, y: int, radius: int, width: int, height: int,
                           dx: float, dy: float) -> bool:
    return (x + dx - radius < 1.0 or y + dy - radius < 1.0
            or x + dx + radius >= width - 2.0 or y + dy + radius >= height - 2.0)


def run_cli(binary: Path, ref: Path, tar: Path, out_dir: Path, log: Path,
            radius: int, extra: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_list = out_dir.parent / (out_dir.name + "-list.txt")
    tar_list.write_text(f"{tar} frame_000000.csv\n")
    command = [str(binary), "--multi-frame", "--ref", str(ref), "--tar-list", str(tar_list),
               "--output-dir", str(out_dir), str(radius), "1", "--threads", "4",
               "--conv-threshold", "0.001", "--no-skip-fftcc", "--output-format", "raw"] + extra
    with log.open("w") as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"CLI failed (rc={rc}); see {log}")
    return read_raw(out_dir / "frame_000000.bin")


def classify(rows: dict, radius: int, width: int, height: int, dx: float, dy: float):
    band = {k for k in rows if predicted_out_of_range(k[0], k[1], radius, width, height, dx, dy)}
    interior = set(rows) - band
    return {
        "poi": len(rows),
        "band": len(band),
        "interior": len(interior),
        "band_rejected": sum(rows[k][2] == MINUS3 for k in band),
        "band_valid": sum(rows[k][2] >= 0.0 for k in band),
        "interior_rejected": sum(rows[k][2] == MINUS3 for k in interior),
        "interior_valid": sum(rows[k][2] >= 0.0 for k in interior),
        "_band": band,
        "_minus3": {k for k in rows if rows[k][2] == MINUS3},
    }


def report(label: str, s: dict, errors: list[str]) -> None:
    print(f"{label}: poi={s['poi']} band={s['band']} band_rejected={s['band_rejected']} "
          f"band_valid={s['band_valid']} interior={s['interior']} "
          f"interior_rejected={s['interior_rejected']} interior_valid={s['interior_valid']}")
    if s["band"] == 0:
        errors.append(f"{label}: the synthetic case exercises no out-of-range subset")
        return
    # Every POI the shift pushes off the image must be rejected by the rule.
    if s["band_rejected"] != s["band"]:
        errors.append(f"{label}: {s['band'] - s['band_rejected']} of {s['band']} "
                      "out-of-range POIs were not rejected with zncc = -3")
    # ... and none of them may survive as a measurement.
    if s["band_valid"]:
        errors.append(f"{label}: {s['band_valid']} out-of-range POIs are still valid")
    # A subset that stays inside the image is untouched by the rule.
    if s["interior_rejected"]:
        errors.append(f"{label}: {s['interior_rejected']} in-range POIs were rejected "
                      "with zncc = -3; the rule is over-rejecting")
    if s["interior_valid"] == 0:
        errors.append(f"{label}: no in-range POI produced a measurement at all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path,
                        help="opencorr_cli_cuda_mvp from any preset (CUDA or CPU-only)")
    parser.add_argument("--cpu-binary", type=Path,
                        help="optional second CLI; both must reject the identical POI set")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--radius", type=int, default=15)
    parser.add_argument("--shift", type=int, default=10)
    args = parser.parse_args()

    root = args.work_dir or Path(tempfile.mkdtemp(prefix="opencorr-oob-gate-"))
    root.mkdir(parents=True, exist_ok=True)
    print("evidence:", root)

    base = make_base(args.width, args.height, seed=20260917)
    ref = root / "ref.pgm"
    tar = root / "tar_shift.pgm"
    write_pgm(ref, args.width, args.height, base)
    write_pgm(tar, args.width, args.height,
              shifted_image(args.width, args.height, base, args.shift, 0, fill=127))

    errors: list[str] = []
    rows = run_cli(args.binary, ref, tar, root / "out", root / "cli.log", args.radius,
                   ["--gpu-device", str(args.gpu_device)])
    primary = classify(rows, args.radius, args.width, args.height, args.shift, 0)
    report("binary", primary, errors)

    if args.cpu_binary:
        cpu_rows = run_cli(args.cpu_binary, ref, tar, root / "out-cpu", root / "cli-cpu.log",
                           args.radius, [])
        secondary = classify(cpu_rows, args.radius, args.width, args.height, args.shift, 0)
        report("cpu-binary", secondary, errors)
        only_a = primary["_minus3"] - secondary["_minus3"]
        only_b = secondary["_minus3"] - primary["_minus3"]
        print(f"cross-backend: minus3_both={len(primary['_minus3'] & secondary['_minus3'])} "
              f"only_binary={len(only_a)} only_cpu_binary={len(only_b)}")
        if only_a or only_b:
            errors.append(f"the two backends disagree on {len(only_a) + len(only_b)} POIs")

    for message in errors:
        print("FAIL:", message)
    if errors:
        return 1
    print("OOB rejection gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
