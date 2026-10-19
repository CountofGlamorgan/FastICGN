#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Known-truth sequence and direct-library cache regressions; leaves all evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile

from generate_synthetic_dic import make_base, shifted_image, write_pgm


def check_frame(path: Path, shift: int) -> dict:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    interior = [r for r in rows if 24 < float(r["x"]) < 167 and 24 < float(r["y"]) < 103]
    valid = [r for r in interior if float(r["zncc"]) >= 0
             and all(math.isfinite(float(r[k])) for k in ("u", "v", "zncc"))]
    correct = [r for r in valid if abs(float(r["u"]) - shift) < .01 and abs(float(r["v"])) < .01]
    return {"interior": len(interior), "valid": len(valid), "correct": len(correct),
            "pass": len(interior) == 420 and len(correct) >= .95 * len(interior)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-bin", "--binary", dest="gpu_bin", required=True, type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--work-dir", type=Path, help="Must not exist; default creates a unique temp directory")
    args = parser.parse_args()
    root = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="opencorr-cache-regression-"))
    if args.work_dir:
        root.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {root}", flush=True)
    base = make_base(192, 128, 20260916)
    write_pgm(root / "ref.pgm", 192, 128, base)
    write_pgm(root / "ref2.pgm", 192, 128, make_base(192, 128, 20260917))
    for shift in (0, 5):
        write_pgm(root / f"tar{shift}.pgm", 192, 128, shifted_image(192, 128, base, shift, 0, 127))
    results = {}
    modes = {"noskip": ["--no-skip-fftcc", "--no-poi-fallback"], "default": [],
             "no-preload": ["--no-preload", "--no-skip-fftcc", "--no-poi-fallback"]}
    for shifts in ((0, 5, 0), (5, 0, 5)):
        for mode, flags in modes.items():
            name = "seq" + "".join(map(str, shifts)) + "-" + mode
            listing = root / (name + ".txt")
            listing.write_text("".join(f"{root}/tar{s}.pgm frame_{i:06d}.csv\n" for i, s in enumerate(shifts)))
            output = root / name
            command = [str(args.gpu_bin.resolve()), "--multi-frame", "--ref", str(root / "ref.pgm"),
                       "--tar-list", str(listing), "--output-dir", str(output), "7", "5",
                       "--threads", str(args.threads), "--gpu-device", str(args.gpu_device)] + flags
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (root / (name + ".log")).write_text(run.stdout + run.stderr)
            frames = [check_frame(output / f"frame_{i:06d}.csv", s) for i, s in enumerate(shifts)] if run.returncode == 0 else []
            results[name] = {"command": command, "rc": run.returncode, "frames": frames,
                             "pass": run.returncode == 0 and all(f["pass"] for f in frames)}
            print(name, results[name]["pass"], frames, flush=True)
    if args.probe:
        command = [str(args.probe.resolve()), str(root), str(args.gpu_device)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (root / "library.log").write_text(run.stdout + run.stderr)
        results["library"] = {"command": command, "rc": run.returncode, "pass": run.returncode == 0}
        print(run.stdout, flush=True)
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(r["pass"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
