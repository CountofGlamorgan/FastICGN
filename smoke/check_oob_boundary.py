#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Summarize CPU/GPU differences where target samples leave the image domain."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


def read_csv(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (int(float(row["x"])), int(float(row["y"])))
            rows[key] = {
                "u": float(row["u"]),
                "v": float(row["v"]),
                "zncc": float(row["zncc"]),
            }
    return rows


def predicted_oob(x: int, y: int, subset_radius: int, width: int, height: int, dx: float, dy: float) -> bool:
    min_x = x + dx - subset_radius
    max_x = x + dx + subset_radius
    min_y = y + dy - subset_radius
    max_y = y + dy + subset_radius
    return min_x < 1.0 or min_y < 1.0 or max_x >= width - 2.0 or max_y >= height - 2.0


def finite_valid(row: dict[str, float]) -> bool:
    return row["zncc"] >= 0.0 and all(math.isfinite(row[field]) for field in ("u", "v", "zncc"))


def summarize(label: str, deltas: list[tuple[float, float, float]]) -> None:
    if not deltas:
        print(f"{label}: count=0")
        return
    du = [item[0] for item in deltas]
    dv = [item[1] for item in deltas]
    dzncc = [item[2] for item in deltas]
    print(
        f"{label}: count={len(deltas)} "
        f"median_abs_du={median(du):.9g} max_abs_du={max(du):.9g} "
        f"median_abs_dv={median(dv):.9g} max_abs_dv={max(dv):.9g} "
        f"median_abs_dzncc={median(dzncc):.9g} max_abs_dzncc={max(dzncc):.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-csv", required=True)
    parser.add_argument("--gpu-csv", required=True)
    parser.add_argument("--subset-radius", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, default=0.0)
    args = parser.parse_args()

    cpu_rows = read_csv(Path(args.cpu_csv))
    gpu_rows = read_csv(Path(args.gpu_csv))
    common_keys = sorted(set(cpu_rows) & set(gpu_rows))
    if not common_keys:
        raise SystemExit("no shared POIs")

    interior: list[tuple[float, float, float]] = []
    oob: list[tuple[float, float, float]] = []
    oob_common = 0
    oob_valid = 0
    oob_cpu_rejected = 0
    oob_gpu_rejected = 0

    for x, y in common_keys:
        is_oob = predicted_oob(x, y, args.subset_radius, args.width, args.height, args.dx, args.dy)
        if is_oob:
            oob_common += 1
            # -3 is "terminated before a result existed", which is what both
            # backends now write for a partially out-of-bounds target subset.
            oob_cpu_rejected += cpu_rows[(x, y)]["zncc"] == -3.0
            oob_gpu_rejected += gpu_rows[(x, y)]["zncc"] == -3.0

        cpu = cpu_rows[(x, y)]
        gpu = gpu_rows[(x, y)]
        if not (finite_valid(cpu) and finite_valid(gpu)):
            continue

        delta = (abs(cpu["u"] - gpu["u"]), abs(cpu["v"] - gpu["v"]), abs(cpu["zncc"] - gpu["zncc"]))
        if is_oob:
            oob_valid += 1
            oob.append(delta)
        else:
            interior.append(delta)

    print(f"shared={len(common_keys)} oob_common={oob_common} oob_shared_valid={oob_valid} "
          f"oob_cpu_zncc_minus3={oob_cpu_rejected} oob_gpu_zncc_minus3={oob_gpu_rejected}")
    summarize("interior_shared_valid", interior)
    summarize("oob_shared_valid", oob)

    # A POI whose warped subset leaves the target image's interpolatable range
    # is rejected with zncc = -3 by both backends, so none of them may survive
    # as a measurement and every one of them must carry the sentinel.  These are
    # the assertions; the deltas above are diagnostics.
    problems = []
    if oob_common == 0:
        problems.append("synthetic case did not exercise predicted OOB samples")
    if oob_valid:
        problems.append(f"{oob_valid} predicted-OOB POIs are valid on both sides; "
                        "the rejection rule is not in force")
    if oob_cpu_rejected != oob_common:
        problems.append(f"CPU rejected {oob_cpu_rejected} of {oob_common} predicted-OOB POIs")
    if oob_gpu_rejected != oob_common:
        problems.append(f"GPU rejected {oob_gpu_rejected} of {oob_common} predicted-OOB POIs")
    if problems:
        raise SystemExit("; ".join(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
