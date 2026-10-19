#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Convert OpenCorr-GPU raw binary output to legacy CSV files."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Iterable


RAW_FORMAT = "opencorr-gpu-raw-v1"
RAW_RECORD = struct.Struct("<iifffffff")
RAW_FIELD_ORDER = ("x", "y", "u", "v", "ux", "uy", "vx", "vy", "zncc")
DEFAULT_CSV_FIELD_ORDER = ("x", "y", "u", "v", "zncc", "uy", "vy", "ux", "vx")


def format_float(value: float) -> str:
    return format(value, ".6g")


def ensure_under_dir(path: Path, base: Path) -> None:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"refusing to write outside output directory: {path}") from exc


def row_to_dict(raw_row: tuple[int, int, float, float, float, float, float, float, float]) -> dict:
    return {
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


def format_csv_value(field: str, value: int | float) -> str:
    if field in ("x", "y"):
        return str(int(value))
    return format_float(float(value))


def convert_frame(
    raw_path: Path,
    csv_path: Path,
    csv_field_order: Iterable[str],
    expected_rows: int | None,
    expected_bytes: int | None,
) -> dict:
    t0 = time.perf_counter()
    raw_bytes = raw_path.read_bytes()
    read_sec = time.perf_counter() - t0
    if len(raw_bytes) % RAW_RECORD.size != 0:
        raise ValueError(f"{raw_path} size is not a multiple of {RAW_RECORD.size}")
    if expected_bytes is not None and len(raw_bytes) != expected_bytes:
        raise ValueError(f"{raw_path} has {len(raw_bytes)} bytes, manifest says {expected_bytes}")

    rows = len(raw_bytes) // RAW_RECORD.size
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{raw_path} has {rows} rows, manifest says {expected_rows}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(csv_field_order)
    t1 = time.perf_counter()
    with csv_path.open("w", encoding="utf-8", newline="") as out:
        out.write(",".join(fields))
        out.write("\n")
        for raw_row in RAW_RECORD.iter_unpack(raw_bytes):
            row = row_to_dict(raw_row)
            out.write(",".join(format_csv_value(field, row[field]) for field in fields))
            out.write("\n")
    write_sec = time.perf_counter() - t1
    return {
        "raw": str(raw_path),
        "csv": str(csv_path),
        "rows": rows,
        "bytes": len(raw_bytes),
        "read_sec": read_sec,
        "write_sec": write_sec,
    }


def validate_manifest(manifest: dict) -> tuple[tuple[str, ...], list[dict]]:
    if manifest.get("format") != RAW_FORMAT:
        raise ValueError(f"manifest format must be {RAW_FORMAT!r}")
    if manifest.get("endianness") != "little":
        raise ValueError("manifest endianness must be 'little'")
    if manifest.get("record_bytes") != RAW_RECORD.size:
        raise ValueError(f"manifest record_bytes must be {RAW_RECORD.size}")

    csv_field_order = tuple(manifest.get("csv_field_order", DEFAULT_CSV_FIELD_ORDER))
    unknown_fields = [field for field in csv_field_order if field not in RAW_FIELD_ORDER]
    if unknown_fields:
        raise ValueError(f"csv_field_order contains unknown raw fields: {unknown_fields}")

    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("manifest frames must be an array")
    return csv_field_order, frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert OpenCorr-GPU raw .bin frames to legacy per-frame CSV files."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Raw manifest JSON path")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for converted CSVs")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    csv_field_order, frames = validate_manifest(manifest)
    raw_dir = args.manifest.parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    for frame in frames:
        raw_name = frame.get("file")
        output_name = frame.get("output_name")
        if not raw_name or not output_name:
            raise ValueError(f"manifest frame missing file or output_name: {frame}")
        raw_path = raw_dir / raw_name
        csv_path = out_dir / output_name
        ensure_under_dir(csv_path, out_dir)
        frame_results.append(
            convert_frame(
                raw_path=raw_path,
                csv_path=csv_path,
                csv_field_order=csv_field_order,
                expected_rows=frame.get("rows"),
                expected_bytes=frame.get("bytes"),
            )
        )

    summary = {
        "manifest": str(args.manifest),
        "out_dir": str(out_dir),
        "frames": len(frame_results),
        "rows": sum(item["rows"] for item in frame_results),
        "bytes": sum(item["bytes"] for item in frame_results),
        "csv_field_order": list(csv_field_order),
        "read_sec": sum(item["read_sec"] for item in frame_results),
        "write_sec": sum(item["write_sec"] for item in frame_results),
        "frames_detail": frame_results,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
