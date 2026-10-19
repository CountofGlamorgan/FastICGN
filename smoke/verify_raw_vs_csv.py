#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Verify OpenCorr-GPU raw binary output against CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import time
from pathlib import Path


RAW_RECORD = struct.Struct("<iifffffff")
RAW_FLOAT_FIELDS = ("u", "v", "ux", "uy", "vx", "vy", "zncc")
CSV_FLOAT_INDEX = {
    "u": 2,
    "v": 3,
    "zncc": 4,
    "uy": 5,
    "vy": 6,
    "ux": 7,
    "vx": 8,
}


def compare_frame(
    binary_path: Path,
    csv_path: Path,
    rel_tol: float,
    abs_tol: float,
) -> dict:
    t0 = time.perf_counter()
    raw_bytes = binary_path.read_bytes()
    binary_read_sec = time.perf_counter() - t0
    if len(raw_bytes) % RAW_RECORD.size != 0:
        raise ValueError(f"{binary_path} size is not a multiple of {RAW_RECORD.size}")

    rows = 0
    max_abs = 0.0
    max_abs_valid = 0.0
    valid_rows = 0
    mismatches = 0
    t1 = time.perf_counter()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_header = ["x", "y", "u", "v", "zncc", "uy", "vy", "ux", "vx"]
        if header != expected_header:
            raise ValueError(f"{csv_path} header {header!r} != {expected_header!r}")
        for rows, (csv_row, raw_row) in enumerate(
            zip(reader, RAW_RECORD.iter_unpack(raw_bytes)), start=1
        ):
            raw = {
                "x": raw_row[0],
                "y": raw_row[1],
                "u": raw_row[2],
                "v": raw_row[3],
                "ux": raw_row[4],
                "uy": raw_row[5],
                "vx": raw_row[6],
                "vy": raw_row[7],
                "zncc": raw_row[8],
            }
            if raw["x"] != int(csv_row[0]) or raw["y"] != int(csv_row[1]):
                mismatches += 1
                continue
            row_valid = raw["zncc"] >= 0.0 and float(csv_row[CSV_FLOAT_INDEX["zncc"]]) >= 0.0
            if row_valid:
                valid_rows += 1
            for field in RAW_FLOAT_FIELDS:
                delta = abs(raw[field] - float(csv_row[CSV_FLOAT_INDEX[field]]))
                max_abs = max(max_abs, delta)
                if row_valid:
                    max_abs_valid = max(max_abs_valid, delta)
                if not math.isclose(
                    raw[field],
                    float(csv_row[CSV_FLOAT_INDEX[field]]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    mismatches += 1
                    break
        expected_rows = len(raw_bytes) // RAW_RECORD.size
        if rows != expected_rows:
            raise ValueError(f"{binary_path} rows {expected_rows} != CSV rows {rows}")
    csv_read_sec = time.perf_counter() - t1
    return {
        "binary": str(binary_path),
        "csv": str(csv_path),
        "rows": rows,
        "bytes": len(raw_bytes),
        "max_abs_delta": max_abs,
        "max_abs_delta_valid_zncc_nonnegative": max_abs_valid,
        "valid_zncc_nonnegative_rows": valid_rows,
        "mismatches": mismatches,
        "binary_read_sec": binary_read_sec,
        "csv_read_sec": csv_read_sec,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rel-tol", type=float, default=5e-6)
    parser.add_argument("--abs-tol", type=float, default=5e-6)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    binary_dir = args.manifest.parent
    frame_results = []
    for frame in manifest["frames"]:
        binary_path = binary_dir / frame["file"]
        csv_path = args.csv_dir / frame["output_name"]
        frame_results.append(compare_frame(binary_path, csv_path, args.rel_tol, args.abs_tol))

    total_rows = sum(item["rows"] for item in frame_results)
    total_bytes = sum(item["bytes"] for item in frame_results)
    total_mismatches = sum(item["mismatches"] for item in frame_results)
    summary = {
        "manifest": str(args.manifest),
        "csv_dir": str(args.csv_dir),
        "frames": len(frame_results),
        "rows": total_rows,
        "bytes": total_bytes,
        "mismatches": total_mismatches,
        "max_abs_delta": max((item["max_abs_delta"] for item in frame_results), default=0.0),
        "max_abs_delta_valid_zncc_nonnegative": max(
            (item["max_abs_delta_valid_zncc_nonnegative"] for item in frame_results),
            default=0.0,
        ),
        "valid_zncc_nonnegative_rows": sum(
            item["valid_zncc_nonnegative_rows"] for item in frame_results
        ),
        "binary_read_sec": sum(item["binary_read_sec"] for item in frame_results),
        "csv_read_sec": sum(item["csv_read_sec"] for item in frame_results),
        "rel_tol": args.rel_tol,
        "abs_tol": args.abs_tol,
        "status": "pass" if total_mismatches == 0 else "fail",
        "frames_detail": frame_results,
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames_detail"}, indent=2))
    return 0 if total_mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
