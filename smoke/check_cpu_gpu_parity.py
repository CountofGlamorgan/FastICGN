#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Check per-frame validity coverage and displacement error on common POIs.

Frames are paired by file name. By default the names are frame_000000.csv,
frame_000001.csv, ... (--frames of them), which is what
run_v9_cpu_gpu_parity.sh writes; --glob PATTERN pairs every matching name that
exists in either directory instead, so pipeline outputs numbered after the
image files (frame_072424.csv, ...) work without renaming.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def read_csv(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(float(row["x"])), int(float(row["y"])))
            if key in rows:
                raise ValueError(f"{path}: duplicate POI {key}")
            rows[key] = {field: float(row[field]) for field in ("u", "v", "zncc")}
    return rows


def finite_valid(row: dict[str, float]) -> bool:
    return row["zncc"] >= 0.0 and all(math.isfinite(row[field]) for field in ("u", "v", "zncc"))


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank quantile, including small samples without interpolation."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def compare_frame(cpu_rows: dict, gpu_rows: dict, *, median_threshold: float = 1e-3,
                  p99_threshold: float = 1e-2, max_threshold: float = .5,
                  min_shared_valid: int = 100,
                  min_valid_retention: float = .95,
                  max_validity_disagreement: float = .05) -> dict:
    common = sorted(cpu_rows.keys() & gpu_rows.keys())
    cpu_valid = gpu_valid = 0
    lost = []
    gained = []
    du, dv, dz = [], [], []
    shared_keys = []
    for key in common:
        cv, gv = finite_valid(cpu_rows[key]), finite_valid(gpu_rows[key])
        cpu_valid += cv
        gpu_valid += gv
        if cv and not gv: lost.append(key)
        if gv and not cv: gained.append(key)
        if cv and gv:
            shared_keys.append(key)
            du.append(abs(cpu_rows[key]["u"] - gpu_rows[key]["u"]))
            dv.append(abs(cpu_rows[key]["v"] - gpu_rows[key]["v"]))
            dz.append(abs(cpu_rows[key]["zncc"] - gpu_rows[key]["zncc"]))
    retention = len(du) / cpu_valid if cpu_valid else 0.0
    disagreement = (len(lost) + len(gained)) / len(common) if common else 1.0
    result = {"shared": len(common), "cpu_valid": cpu_valid, "gpu_valid": gpu_valid,
              "shared_valid": len(du), "valid_retention": retention,
              "validity_disagreement": disagreement, "cpu_valid_gpu_invalid": len(lost),
              "gpu_valid_cpu_invalid": len(gained), "errors": []}
    if lost:
        result["lost_valid_bbox"] = [min(p[0] for p in lost), min(p[1] for p in lost),
                                     max(p[0] for p in lost), max(p[1] for p in lost)]
    if not common: result["errors"].append("no shared POIs")
    if len(du) < min_shared_valid:
        result["errors"].append(f"shared valid POIs below per-frame minimum {min_shared_valid}")
    if retention < min_valid_retention:
        result["errors"].append(f"CPU-valid retention below {min_valid_retention:g}")
    if disagreement > max_validity_disagreement:
        result["errors"].append(f"validity disagreement exceeds {max_validity_disagreement:g}")
    if du:
        for name, values in (("du", du), ("dv", dv), ("dzncc", dz)):
            result[f"median_abs_{name}"] = median(values)
            result[f"p99_abs_{name}"] = percentile(values, .99)
            result[f"max_abs_{name}"] = max(values)
        result["max_du_at"] = shared_keys[max(range(len(du)), key=du.__getitem__)]
        result["max_dv_at"] = shared_keys[max(range(len(dv)), key=dv.__getitem__)]
        if max(median(du), median(dv)) > median_threshold:
            result["errors"].append(f"per-frame median displacement error exceeds {median_threshold:g}")
        if max(percentile(du, .99), percentile(dv, .99)) > p99_threshold:
            result["errors"].append(f"per-frame p99 displacement error exceeds {p99_threshold:g}")
        if max(max(du), max(dv)) > max_threshold:
            result["errors"].append(f"per-frame maximum displacement error exceeds {max_threshold:g}")
    result["pass"] = not result["errors"]
    return result


DEFAULT_FRAMES = 10


def frame_names(cpu_dir: Path, gpu_dir: Path, frames: int | None, pattern: str | None) -> list[str]:
    """File names to compare, in order (see the module docstring)."""
    if pattern is None:
        count = DEFAULT_FRAMES if frames is None else frames
        return [f"frame_{index:06d}.csv" for index in range(count)]
    names = sorted({
        path.name for directory in (cpu_dir, gpu_dir)
        for path in directory.glob(pattern) if path.is_file()
    })
    return names if frames is None else names[:frames]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpu-dir", required=True, type=Path)
    parser.add_argument("--gpu-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=None,
                        help="how many frames to compare. Without --glob: the first N "
                             "0-based names frame_000000.csv .. (default "
                             f"{DEFAULT_FRAMES}); a missing file fails its frame. With "
                             "--glob: the first N matching names in sorted order (default all)")
    parser.add_argument("--glob", default=None, metavar="PATTERN",
                        help="pair files by name: every file matching PATTERN (for example "
                             "'frame_*.csv') in either directory is compared with the file of "
                             "the same name in the other; a name present on one side only "
                             "fails that frame")
    parser.add_argument("--median-threshold", type=float, default=1e-3)
    parser.add_argument("--p99-threshold", type=float, default=1e-2)
    parser.add_argument("--max-threshold", type=float, default=.5,
                        help="Maximum |du| or |dv| in px, including isolated outliers")
    parser.add_argument("--min-shared-valid", type=int, default=100, help="Sequence-wide floor (legacy option)")
    parser.add_argument("--min-frame-shared-valid", type=int, default=100)
    parser.add_argument("--min-valid-retention", type=float, default=.95)
    parser.add_argument("--max-validity-disagreement", type=float, default=.05)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()
    if (args.frames is not None and args.frames < 1) or min(args.min_shared_valid, args.min_frame_shared_valid) < 1:
        parser.error("frames and minimum shared-valid counts must be positive")
    for name in ("median_threshold", "p99_threshold", "max_threshold", "min_valid_retention", "max_validity_disagreement"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.min_valid_retention > 1 or args.max_validity_disagreement > 1:
        parser.error("validity fractions must not exceed 1")
    names = frame_names(args.cpu_dir, args.gpu_dir, args.frames, args.glob)
    if not names:
        print(f"no file in {args.cpu_dir} or {args.gpu_dir} matches --glob {args.glob!r}")
    frames = []
    for name in names:
        try:
            result = compare_frame(read_csv(args.cpu_dir / name), read_csv(args.gpu_dir / name),
                median_threshold=args.median_threshold, p99_threshold=args.p99_threshold,
                max_threshold=args.max_threshold,
                min_shared_valid=args.min_frame_shared_valid,
                min_valid_retention=args.min_valid_retention,
                max_validity_disagreement=args.max_validity_disagreement)
        except (OSError, ValueError, KeyError, TypeError) as ex:
            result = {"pass": False, "shared_valid": 0, "errors": [str(ex)]}
        result["frame"] = name
        frames.append(result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    total = sum(f["shared_valid"] for f in frames)
    report = {"schema_version": 1, "frames": frames, "pooled_shared_valid": total,
              "pairing": "glob" if args.glob else "index", "glob": args.glob,
              "pass": bool(frames) and total >= args.min_shared_valid and all(f["pass"] for f in frames),
              "thresholds": {k: getattr(args, k) for k in ("median_threshold", "p99_threshold", "max_threshold",
                  "min_shared_valid", "min_frame_shared_valid", "min_valid_retention", "max_validity_disagreement")}}
    if args.json_report:
        args.json_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"pooled_shared_valid={total} verdict={'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
