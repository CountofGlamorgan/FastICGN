#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Assertion gate for zero-texture (constant) reference subsets.

A reference subset whose sampled values are all equal has nothing to
correlate. The CUDA precompute kernel rejects such a POI before iterating with
``zncc = -3`` (2026-10-19; before that, the few of them whose Hessian was
merely ill-conditioned left the solver as a "converged" measurement with
``zncc`` near 0.5 or exactly 1.0). The CPU ICGN backend (vendored upstream
OpenCorr, unchanged) reaches the same subsets through a 0/0 cost and reports
``zncc = -5``. Both are asserted here on a synthetic speckle image carrying
three constant blocks -- black (0), mid-grey (128) and saturated (255) -- so
the set of POIs whose subset lies entirely inside a block is known in closed
form before the CLI runs.

Assertions (every run):

* every POI whose subset lies entirely inside a block is rejected: ``-3`` on
  the CUDA backend (default kernel and ``--kernel legacy``, and again with
  ``--input-normalize unit``, since the predicate is exact equality of the
  samples and not a norm threshold), ``-5`` on the CPU backend; their ``u``,
  ``v`` carry the incoming guess (0: FFTCC skips a subset without texture);
* no POI whose subset lies entirely outside the blocks is touched: for the
  pair ``ref -> ref`` every one of them is a valid measurement with
  ``zncc == 1`` and ``|u|, |v| <= 1e-4``.

With ``--reference-binary`` (a build without the rule, or any other build) the
gate also asserts, record by record, that every POI *not* entirely inside a
block -- the POIs partly inside a block included -- is byte-identical between
the two binaries: the rule changes only the POIs it names.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic_dic import make_base, shifted_image, write_pgm  # noqa: E402

RECORD = struct.Struct("<iifffffff")  # x y u v ux uy vx vy zncc
BLOCKS = {"black": 0, "grey": 128, "saturated": 255}
BLOCK_SIDE = 40
BLOCK_Y = 60
BLOCK_X = {"black": 40, "grey": 140, "saturated": 240}
FILL = 127


def paint_blocks(width: int, height: int, pixels: bytearray) -> bytearray:
    out = bytearray(pixels)
    for name, value in BLOCKS.items():
        x0 = BLOCK_X[name]
        for y in range(BLOCK_Y, BLOCK_Y + BLOCK_SIDE):
            row = y * width
            for x in range(x0, x0 + BLOCK_SIDE):
                out[row + x] = value
    return out


def block_relation(x: int, y: int, radius: int) -> str:
    """'inside' if the subset lies entirely inside one block, 'partial' if it
    overlaps a block without being inside it, else 'outside'."""
    for name in BLOCKS:
        x0, y0 = BLOCK_X[name], BLOCK_Y
        x1, y1 = x0 + BLOCK_SIDE - 1, y0 + BLOCK_SIDE - 1
        if x - radius >= x0 and x + radius <= x1 and y - radius >= y0 and y + radius <= y1:
            return "inside"
        if x + radius >= x0 and x - radius <= x1 and y + radius >= y0 and y - radius <= y1:
            return "partial"
    return "outside"


def read_raw(path: Path) -> dict[tuple[int, int], bytes]:
    blob = path.read_bytes()
    if len(blob) % RECORD.size:
        raise SystemExit(f"{path}: {len(blob)} bytes is not a whole number of records")
    rows = {}
    for offset in range(0, len(blob), RECORD.size):
        x, y = struct.unpack_from("<ii", blob, offset)
        rows[(x, y)] = blob[offset:offset + RECORD.size]
    return rows


def fields(record: bytes) -> tuple:
    return RECORD.unpack(record)


def backend_of(binary: Path) -> str:
    run = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
    for line in run.stdout.splitlines():
        if line.startswith("backend="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{binary}: --version did not report backend= (rc {run.returncode})")


def run_cli(binary: Path, ref: Path, tar: Path, out: Path, log: Path, radius: int,
            extra: list[str]) -> dict[tuple[int, int], bytes]:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), str(ref), str(tar), str(radius), "1", str(out),
               "--conv-threshold", "0.001", "--output-format", "raw"] + extra
    run = subprocess.run(command, capture_output=True, text=True, timeout=600)
    log.write_text(" ".join(command) + "\n" + run.stdout + run.stderr)
    if run.returncode != 0:
        raise SystemExit(f"CLI failed (rc {run.returncode}): {' '.join(command)}\n{run.stderr[-2000:]}")
    return read_raw(out.with_suffix(".bin"))


def classify(rows: dict[tuple[int, int], bytes], radius: int) -> dict:
    counts = {"inside": {}, "partial": {}, "outside": {}}
    for (x, y), record in rows.items():
        zncc = fields(record)[8]
        key = str(int(zncc)) if zncc < 0 else ("valid" if zncc == zncc else "nan")
        bucket = counts[block_relation(x, y, radius)]
        bucket[key] = bucket.get(key, 0) + 1
    return counts


def check_rejected(label: str, rows: dict, radius: int, code: float, errors: list[str]) -> int:
    inside = [(k, fields(r)) for k, r in rows.items() if block_relation(*k, radius) == "inside"]
    if not inside:
        errors.append(f"{label}: no POI lies entirely inside a block; the image or radius is wrong")
        return 0
    wrong = [(k, f[8]) for k, f in inside if f[8] != code]
    moved = [(k, f[2], f[3]) for k, f in inside if f[2] != 0.0 or f[3] != 0.0]
    if wrong:
        sample = ", ".join(f"({k[0]},{k[1]}) zncc={z!r}" for k, z in wrong[:6])
        errors.append(f"{label}: {len(wrong)} of {len(inside)} zero-texture POIs are not "
                      f"zncc = {code:g}: {sample}")
    if moved:
        sample = ", ".join(f"({k[0]},{k[1]}) u={u!r} v={v!r}" for k, u, v in moved[:6])
        errors.append(f"{label}: {len(moved)} zero-texture POIs do not carry the incoming "
                      f"guess u = v = 0: {sample}")
    return len(inside)


def check_outside_untouched(label: str, rows: dict, radius: int, errors: list[str]) -> int:
    outside = [(k, fields(r)) for k, r in rows.items() if block_relation(*k, radius) == "outside"]
    bad = [(k, f) for k, f in outside
           if not (f[8] == 1.0 and abs(f[2]) <= 1e-4 and abs(f[3]) <= 1e-4)]
    if bad:
        sample = ", ".join(f"({k[0]},{k[1]}) u={f[2]:.3g} v={f[3]:.3g} zncc={f[8]!r}"
                           for k, f in bad[:6])
        errors.append(f"{label}: {len(bad)} of {len(outside)} POIs outside the blocks are not "
                      f"an exact self-match (zncc 1, u = v = 0): {sample}")
    return len(outside)


def check_identical_outside_inside(label: str, rows: dict, reference: dict, radius: int,
                                   errors: list[str]) -> dict:
    if rows.keys() != reference.keys():
        errors.append(f"{label}: the two binaries wrote different POI sets")
        return {}
    differ = {"partial": 0, "outside": 0}
    inside_differ = 0
    for key, record in rows.items():
        if record == reference[key]:
            continue
        relation = block_relation(*key, radius)
        if relation == "inside":
            inside_differ += 1
        else:
            differ[relation] += 1
    for relation, count in differ.items():
        if count:
            errors.append(f"{label}: {count} POIs {relation} the blocks differ from the reference "
                          "binary; the rule must change only POIs entirely inside a block")
    return {"inside_records_differ": inside_differ, **{f"{k}_records_differ": v for k, v in differ.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path,
                        help="opencorr_cli_cuda_mvp from any preset (CUDA or CPU-only)")
    parser.add_argument("--reference-binary", type=Path,
                        help="optional second CLI; every POI not entirely inside a block must be "
                             "byte-identical between the two")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--radius", type=int, default=7)
    parser.add_argument("--shift", type=int, default=3)
    args = parser.parse_args()

    if args.height < BLOCK_Y + BLOCK_SIDE + 2 * args.radius + 4:
        raise SystemExit("--height too small for the blocks")
    if args.width < BLOCK_X["saturated"] + BLOCK_SIDE + 2 * args.radius + 4:
        raise SystemExit("--width too small for the blocks")

    root = args.work_dir or Path(tempfile.mkdtemp(prefix="opencorr-zero-texture-gate-"))
    root.mkdir(parents=True, exist_ok=True)
    print("evidence:", root)

    base = paint_blocks(args.width, args.height, make_base(args.width, args.height, seed=20261019))
    ref = root / "ref.pgm"
    tar_same = root / "tar_same.pgm"
    tar_shift = root / "tar_shift.pgm"
    write_pgm(ref, args.width, args.height, base)
    write_pgm(tar_same, args.width, args.height, base)
    # The blocks stay where they are in the shifted target: a zero-texture
    # reference subset then faces a textured target subset, the case that used
    # to come out as zncc 0.5.
    write_pgm(tar_shift, args.width, args.height,
              paint_blocks(args.width, args.height,
                           shifted_image(args.width, args.height, base, args.shift, 0, fill=FILL)))

    backend = backend_of(args.binary)
    code = -3.0 if backend == "cuda" else -5.0
    device = ["--gpu-device", str(args.gpu_device)] if backend == "cuda" else []
    variants = [("default", [])]
    if backend == "cuda":
        variants += [("legacy", ["--kernel", "legacy"]), ("unit", ["--input-normalize", "unit"])]

    errors: list[str] = []
    summary = {"backend": backend, "expected_code": code, "runs": {}}
    for name, extra in variants:
        for pair, tar in (("same", tar_same), ("shift", tar_shift)):
            label = f"{name}/{pair}"
            rows = run_cli(args.binary, ref, tar, root / f"out-{name}-{pair}" / "out.csv",
                           root / f"cli-{name}-{pair}.log", args.radius, device + extra)
            counts = classify(rows, args.radius)
            inside = check_rejected(label, rows, args.radius, code, errors)
            outside = check_outside_untouched(label, rows, args.radius, errors) \
                if pair == "same" and name != "unit" else None
            run_summary = {"counts": counts, "inside_pois": inside, "outside_checked": outside}
            if args.reference_binary:
                ref_rows = run_cli(args.reference_binary, ref, tar,
                                   root / f"ref-{name}-{pair}" / "out.csv",
                                   root / f"cli-ref-{name}-{pair}.log", args.radius, device + extra)
                run_summary["reference"] = check_identical_outside_inside(
                    label, rows, ref_rows, args.radius, errors)
                run_summary["reference_counts"] = classify(ref_rows, args.radius)
            summary["runs"][label] = run_summary
            print(f"{label}: inside={inside} {counts['inside']} partial={counts['partial']} "
                  f"outside={counts['outside']}"
                  + (f" reference-diff={run_summary['reference']}" if args.reference_binary else ""))

    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for message in errors:
        print("FAIL:", message)
    if errors:
        return 1
    print(f"zero-texture gate PASS ({backend}: zero-texture reference subsets carry zncc = {code:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
