#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Read a raw multi-frame output directory back through pipeline/fasticgn_io.py.

Used by smoke/selfcheck.sh: it proves that the published reader, the manifest
(including its engine block) and the ledger agree with the bytes on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import numpy as np

import fasticgn_io


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expect-frames", type=int, default=None)
    parser.add_argument("--min-valid-ratio", type=float, default=0.90)
    args = parser.parse_args()

    manifest_path = fasticgn_io.manifest_path(args.output_dir)
    manifest = fasticgn_io.read_manifest(manifest_path)
    frames = manifest["frames"]
    if args.expect_frames is not None and len(frames) != args.expect_frames:
        print(f"FAIL: manifest lists {len(frames)} frames, expected {args.expect_frames}")
        return 1

    engine = fasticgn_io.engine_metadata(manifest)
    required = {"implementation", "icgn_backend", "icgn_kernel", "oob_policy", "fftcc_backend",
                "skip_fftcc", "poi_fallback", "temporal_extrapolation", "frame_fallback_policy",
                "durability", "threads", "git_hash", "build_date", "cuda"}
    missing = sorted(required - set(engine))
    if missing:
        print(f"FAIL: manifest engine block is missing {missing}")
        return 1

    total = 0
    valid = 0
    for frame, records in fasticgn_io.iter_frames(manifest, args.output_dir):
        if records.dtype != fasticgn_io.RAW_DTYPE:
            print(f"FAIL: {frame.get('file')} did not decode to the documented dtype")
            return 1
        mask = fasticgn_io.valid_mask(records)
        total += records.shape[0]
        valid += int(np.count_nonzero(mask))
        columns = fasticgn_io.to_csv_columns(records)
        if columns.shape != (records.shape[0], len(fasticgn_io.CSV_FIELD_ORDER)):
            print(f"FAIL: CSV conversion shape mismatch for {frame.get('file')}")
            return 1

    if total == 0:
        print("FAIL: no POI rows were read")
        return 1
    ratio = valid / total
    if ratio < args.min_valid_ratio:
        print(f"FAIL: valid ratio {ratio:.4f} below {args.min_valid_ratio}")
        return 1

    ledger = fasticgn_io.read_ledger(args.output_dir)
    if not ledger.get("present"):
        print("FAIL: no .opencorr-state ledger next to the frames")
        return 1
    if "complete.json" not in ledger:
        print("FAIL: ledger has no complete.json run record")
        return 1

    print(
        f"OK: {len(frames)} frames, {total} rows, valid {ratio:.4f}, "
        f"engine kernel={engine['icgn_kernel']} fftcc={engine['fftcc_backend']} "
        f"backend={engine['icgn_backend']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
