#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Assertion gate: sub-pixel accuracy on noisy speckle, and the CLI's numeric defaults.

Why this exists (review 2026-10-03, section 2.4). The other synthetic checks
move a noise-free image by whole pixels, so FFTCC already lands on the answer
and ICGN stops after one iteration. Historical bugs planted back into a scratch
copy therefore passed ctest, selfcheck.sh and the CPU/GPU parity smoke:

* the legacy kernel leaving its iteration loop on a per-thread convergence
  value ("finalconv"; valid ratio on lab data 0.74 -> 0.09);
* the default --conv-threshold back at 0.05 instead of 0.001;
* --subset-size N used as a radius instead of a size (radius = (N - 1) / 2);
* the legacy kernel loading the 36-entry inverse Hessian in one pass, which
  leaves entries 32..35 stale when the block has 32 threads (subset 63, "s63").

Only the production anchors saw any of them, and those need the lab dataset.

Data: smoke/generate_subpixel_speckle.py (standard library only) renders
Gaussian speckles analytically under an affine displacement field, so u, v
and the four gradients are exact at sub-pixel values; contrast is set in grey
levels, sensor noise is added and the image is quantised to 8 bit.

case        image    radius grid frames contrast noise  purpose
s15         192x144       7    2      4       40     2  production subset (15 px)
s15_lowsnr  192x144       7    2      2       16     4  low SNR: about 20 % of the POIs
                                                        use all 10 iterations
s63         256x224      31    3      3       40     2  32-thread blocks in the legacy kernel

Rigid-body displacements are 0.12 to 1.73 px per component, gradients 0.001 to
0.008 per px.

Runs. Every case: "explicit" (multi-frame, positional radius, --conv-threshold
0.001 --max-iter 10) and, on a CUDA build, "legacy" (the same with --kernel
legacy). s15 adds "default" (multi-frame, --subset-size 15, no threshold, no
iteration cap), "loose" (--conv-threshold 0.05), and "single-explicit" /
"single-default" (single-frame mode on the last target, same two spellings).

Assertions.

1. Grid. Every frame has the POI count and first POI that radius + 2 margins
   give for the radius asked for, so a size read as a radius cannot hide.
2. Accuracy, per frame, against the truth at each POI (valid = finite
   zncc >= 0; the valid ratio is taken over POIs whose subset stays inside the
   image at the true displacement): valid ratio, RMS and mean error of u and
   v, 99th percentile of max(|du|, |dv|), RMS error of ux, uy, vx, vy, median
   zncc, the manifest's iteration_mean, and no POI with zncc >= 0.8 more than
   1 px off.
3. Defaults (s15, multi-frame and single-frame): the "default" manifest says
   conv_threshold 0.001, max_iteration 10, subset_radius 7 and, on CUDA,
   icgn_kernel warp; its frame bytes equal those of the "explicit" run made by
   the same binary; and "loose" differs from "explicit" in at least one frame,
   otherwise the data could not tell the two thresholds apart. In a checkout,
   pipeline/interface_contract.json must document the same two defaults.

Limits. Measured at 3e070b1 on the lab host (RTX 6000 Ada, CUDA 13.2), with
the cuda-fftw build (warp and legacy kernels) and the cpu-only build; the
three agree to within 0.00005 px, one POI in a valid count and 0.001 in
iteration_mean. Worst value over all runs and frames:

case        valid    rmse     |bias|   p99     grad rmse  zncc50  iteration_mean
s15         0.9906   0.01454  0.00431  0.0432  0.003486   0.9973  4.533 .. 4.851
s15_lowsnr  0.7964   0.07173  0.00715  0.2209  0.017589   0.9430  8.351 .. 8.468
s63         1.0000   0.00410  0.00128  0.0106  0.000205   0.9976  2.897 .. 2.950

One rule for every limit, fixed before any mutant was run: allow twice the
worst value. Errors (rmse, bias, p99, gradient rmse): 2 x worst, rounded up to
two significant digits. Valid ratio and median zncc: twice the shortfall from
1, and the valid ratio is never asked to exceed 0.98. iteration_mean: from
min / 1.5 to max x 1.5, capped at the iteration limit of 10. These are
regression limits for this gate, not a statement of measurement uncertainty.

The comparison is against the truth, not against recorded bytes, so the gate
is meant to hold on other GPUs, CUDA versions and the CPU backend. The SHA-256
values in summary.json (images, frame payloads) are provenance only.

Standard-library Python; about 4 s with CUDA, 4 s CPU-only on the lab host.
It is ctest case subpixel_accuracy and a smoke/selfcheck.sh stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_subpixel_speckle as speckle  # noqa: E402

RECORD = struct.Struct("<iifffffff")  # x y u v ux uy vx vy zncc
GOOD_ZNCC = 0.8
HIGH_CONFIDENCE_ERROR_PX = 1.0
DEFAULT_CONV_THRESHOLD = 0.001
DEFAULT_MAX_ITERATION = 10
LOOSE_CONV_THRESHOLD = 0.05
ITERATION_BLOCK = re.compile(r"iteration_block=(\d+)")

CASES = {
    "s15": {
        "width": 192, "height": 144, "radius": 7, "grid": 2, "contrast": 40.0, "noise": 2.0,
        "defaults": True,
        "motions": [(0.37, -0.21, 0.0020, -0.0010, 0.0015, -0.0025),
                    (0.81, 0.12, 0.0030, -0.0015, 0.0020, -0.0030),
                    (1.26, 0.48, 0.0040, -0.0020, 0.0025, -0.0035),
                    (1.73, 0.83, 0.0050, -0.0025, 0.0030, -0.0040)],
    },
    "s15_lowsnr": {
        "width": 192, "height": 144, "radius": 7, "grid": 2, "contrast": 16.0, "noise": 4.0,
        "motions": [(0.43, 0.29, -0.0020, 0.0010, -0.0010, 0.0020),
                    (0.91, 0.64, -0.0030, 0.0015, -0.0015, 0.0030)],
    },
    "s63": {
        "width": 256, "height": 224, "radius": 31, "grid": 3, "contrast": 40.0, "noise": 2.0,
        "motions": [(0.46, -0.33, 0.0030, -0.0020, 0.0020, 0.0040),
                    (0.93, -0.61, 0.0045, -0.0030, 0.0030, 0.0060),
                    (1.38, -0.94, 0.0060, -0.0040, 0.0040, 0.0080)],
    },
}

# Twice the worst value measured at 3e070b1; the module docstring has the table and the rule.
LIMITS = {
    "s15": {"min_valid_ratio": 0.98, "max_rmse_px": 0.030, "max_bias_px": 0.0087,
            "max_p99_px": 0.087, "max_grad_rmse": 0.0070, "min_zncc_median": 0.9946,
            "iteration_mean": (3.0, 7.3)},
    "s15_lowsnr": {"min_valid_ratio": 0.59, "max_rmse_px": 0.15, "max_bias_px": 0.015,
                   "max_p99_px": 0.45, "max_grad_rmse": 0.036, "min_zncc_median": 0.886,
                   "iteration_mean": (5.5, 10.0)},
    "s63": {"min_valid_ratio": 0.98, "max_rmse_px": 0.0082, "max_bias_px": 0.0026,
            "max_p99_px": 0.022, "max_grad_rmse": 0.00041, "min_zncc_median": 0.9952,
            "iteration_mean": (1.9, 4.5)},
}


def read_raw(path: Path) -> list[tuple]:
    blob = path.read_bytes()
    if len(blob) % RECORD.size:
        raise SystemExit(f"{path}: {len(blob)} bytes is not a whole number of records")
    return list(RECORD.iter_unpack(blob))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def json_safe(value):
    """NaN (a frame without valid POIs) is not JSON; write null instead."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def expected_grid(width: int, height: int, radius: int, grid: int) -> tuple[int, int, int]:
    """buildPoiQueue(): margin radius + 2, then every grid-th pixel."""
    margin = radius + 2
    nx = len(range(margin, width - margin, grid))
    ny = len(range(margin, height - margin, grid))
    return margin, nx, ny


def measurable(x: int, y: int, u: float, v: float, case: dict) -> bool:
    """Target subset inside the interpolatable range at the true displacement, 1 px slack."""
    r = case["radius"]
    return (x + u - r >= 2.0 and y + v - r >= 2.0
            and x + u + r < case["width"] - 3.0 and y + v + r < case["height"] - 3.0)


class Runner:
    def __init__(self, binary: Path, gpu_device: int, threads: int):
        self.binary = binary
        self.gpu_device = gpu_device
        self.threads = threads

    def _run(self, root: Path, label: str, command: list[str]) -> str:
        log = root / f"cli-{label}.log"
        with (root / "commands.txt").open("a") as handle:
            handle.write(" ".join(command) + "\n")
        with log.open("w") as handle:
            rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"FAIL: CLI exited {rc}; see {log}")
        return log.read_text(errors="replace")

    def multi(self, root: Path, label: str, subset: list[str], grid: int, extra: list[str]) -> dict:
        out_dir = root / f"out-{label}"
        command = [str(self.binary), "--multi-frame", "--ref", str(root / "ref.pgm"),
                   "--tar-list", str(root / "tar_list.txt"), "--output-dir", str(out_dir),
                   *subset, str(grid), "--threads", str(self.threads),
                   "--gpu-device", str(self.gpu_device), "--output-format", "raw", *extra]
        log = self._run(root, label, command)
        manifest = json.loads((out_dir / "manifest.opencorr-raw.json").read_text())
        return {"label": label, "log": log, "manifest": manifest,
                "files": [out_dir / frame["file"] for frame in manifest["frames"]]}

    def single(self, root: Path, label: str, target: str, subset: list[str], grid: int,
               extra: list[str]) -> dict:
        out_dir = root / f"out-{label}"
        out_dir.mkdir()
        output = out_dir / "frame_single.bin"
        command = [str(self.binary), str(root / "ref.pgm"), str(root / target), *subset, str(grid),
                   str(output), "--threads", str(self.threads), "--gpu-device", str(self.gpu_device),
                   "--output-format", "raw", *extra]
        log = self._run(root, label, command)
        manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
        return {"label": label, "log": log, "manifest": manifest, "files": [output]}


def evaluate(case: dict, run: dict, motions: list, radius: int) -> list[dict]:
    """Per-frame accuracy against the truth, plus the manifest's iteration statistics."""
    width, height = case["width"], case["height"]
    margin, nx, ny = expected_grid(width, height, radius, case["grid"])
    frames = []
    for motion, path, qa in zip(motions, run["files"], run["manifest"]["frames"]):
        rows = read_raw(path)
        eu, ev, emax, grad = [], [], [], [[], [], [], []]
        zncc_valid = []
        status = {"-3": 0, "-4": 0, "-5": 0, "other": 0}
        n_measurable = n_measurable_valid = hiconf = 0
        for x, y, u, v, ux, uy, vx, vy, zncc in rows:
            ut, vt = speckle.displacement(motion, width, height, x, y)
            valid = math.isfinite(zncc) and zncc >= 0.0
            if measurable(x, y, ut, vt, case):
                n_measurable += 1
                n_measurable_valid += valid
            if not valid:
                key = str(int(zncc)) if math.isfinite(zncc) and zncc in (-3.0, -4.0, -5.0) else "other"
                status[key] += 1
                continue
            du, dv = u - ut, v - vt
            eu.append(du)
            ev.append(dv)
            worst = max(abs(du), abs(dv))
            emax.append(worst)
            if zncc >= GOOD_ZNCC and worst > HIGH_CONFIDENCE_ERROR_PX:
                hiconf += 1
            for bucket, value, true in zip(grad, (ux, uy, vx, vy), motion[2:]):
                bucket.append(value - true)
            zncc_valid.append(zncc)
        frames.append({
            "file": path.name, "motion": list(motion), "rows": len(rows),
            "expected_rows": nx * ny, "first_xy": list(rows[0][:2]) if rows else None,
            "expected_first_xy": [margin, margin],
            "measurable": n_measurable,
            "valid_ratio": n_measurable_valid / n_measurable if n_measurable else float("nan"),
            "valid": len(eu), "status": status,
            "bias_u": mean(eu), "bias_v": mean(ev), "rmse_u": rms(eu), "rmse_v": rms(ev),
            "p99_error": percentile(emax, 0.99), "max_error": max(emax, default=float("nan")),
            "grad_rmse": [rms(bucket) for bucket in grad],
            "grad_bias": [mean(bucket) for bucket in grad],
            "zncc_median": percentile(zncc_valid, 0.5),
            "high_confidence_wrong": hiconf,
            "iteration_mean": qa.get("iteration_mean"), "iteration_max": qa.get("iteration_max"),
            "manifest_valid_ratio": qa.get("valid_ratio"),
        })
    return frames


def print_frames(name: str, run: dict, frames: list[dict]) -> None:
    engine = run["manifest"].get("engine", {})
    block = ITERATION_BLOCK.search(run["log"])
    print(f"-- {name}/{run['label']}: kernel={engine.get('icgn_kernel')} "
          f"backend={engine.get('icgn_backend')} radius={run['manifest'].get('subset_radius')} "
          f"conv={run['manifest'].get('conv_threshold')} max_iter={run['manifest'].get('max_iteration')}"
          f"{' iteration_block=' + block.group(1) if block else ''}")
    print(f"   {'frame':<18} {'rows':>5} {'valid':>6} {'bias_u':>8} {'bias_v':>8} {'rmse_u':>7} "
          f"{'rmse_v':>7} {'p99':>7} {'max':>7} {'g_rmse':>8} {'zncc50':>6} {'it_mean':>7} "
          f"{'it_max':>6} {'-3':>4} {'-4':>4} {'-5':>4}")
    for f in frames:
        iteration_mean = float("nan") if f["iteration_mean"] is None else f["iteration_mean"]
        print(f"   {f['file']:<18} {f['rows']:>5} {f['valid_ratio']:>6.4f} {f['bias_u']:>8.5f} "
              f"{f['bias_v']:>8.5f} {f['rmse_u']:>7.5f} {f['rmse_v']:>7.5f} {f['p99_error']:>7.4f} "
              f"{f['max_error']:>7.4f} {max(f['grad_rmse']):>8.6f} {f['zncc_median']:>6.4f} "
              f"{iteration_mean:>7.3f} {str(f['iteration_max']):>6} {f['status']['-3']:>4} "
              f"{f['status']['-4']:>4} {f['status']['-5']:>4}")


def check_limits(name: str, run: dict, frames: list[dict], expected_frames: int, limits: dict,
                 errors: list[str]) -> None:
    tag = f"{name}/{run['label']}"
    if len(frames) != expected_frames:
        errors.append(f"{tag}: {len(frames)} frames in the manifest, expected {expected_frames}")
    for f in frames:
        where = f"{tag} {f['file']}"
        if f["rows"] != f["expected_rows"] or f["first_xy"] != f["expected_first_xy"]:
            errors.append(f"{where}: grid is {f['rows']} POIs from {f['first_xy']}, expected "
                          f"{f['expected_rows']} from {f['expected_first_xy']}")
        if f["high_confidence_wrong"]:
            errors.append(f"{where}: {f['high_confidence_wrong']} POIs with zncc >= {GOOD_ZNCC} "
                          f"and error > {HIGH_CONFIDENCE_ERROR_PX} px")
        if not f["valid_ratio"] >= limits["min_valid_ratio"]:
            errors.append(f"{where}: valid ratio {f['valid_ratio']:.4f} < {limits['min_valid_ratio']}")
        lo, hi = limits["iteration_mean"]
        if f["iteration_mean"] is None or not lo <= f["iteration_mean"] <= hi:
            errors.append(f"{where}: iteration_mean {f['iteration_mean']} outside [{lo}, {hi}]")
        if not f["valid"]:
            errors.append(f"{where}: no valid POI, so no accuracy to measure")
            continue
        for key in ("rmse_u", "rmse_v"):
            if not f[key] <= limits["max_rmse_px"]:
                errors.append(f"{where}: {key} {f[key]:.5f} px > {limits['max_rmse_px']}")
        for key in ("bias_u", "bias_v"):
            if not abs(f[key]) <= limits["max_bias_px"]:
                errors.append(f"{where}: {key} {f[key]:+.5f} px beyond +-{limits['max_bias_px']}")
        if not f["p99_error"] <= limits["max_p99_px"]:
            errors.append(f"{where}: p99 error {f['p99_error']:.4f} px > {limits['max_p99_px']}")
        if not max(f["grad_rmse"]) <= limits["max_grad_rmse"]:
            errors.append(f"{where}: gradient rmse {max(f['grad_rmse']):.6f} /px > "
                          f"{limits['max_grad_rmse']}")
        if not f["zncc_median"] >= limits["min_zncc_median"]:
            errors.append(f"{where}: median zncc {f['zncc_median']:.4f} < {limits['min_zncc_median']}")


def payload_digest(run: dict) -> str:
    digest = hashlib.sha256()
    for path in run["files"]:
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def same_payloads(a: dict, b: dict) -> list[str]:
    differing = []
    for pa, pb in zip(a["files"], b["files"]):
        if pa.read_bytes() != pb.read_bytes():
            differing.append(pa.name)
    if len(a["files"]) != len(b["files"]):
        differing.append("<frame count>")
    return differing


def contract_errors() -> list[str]:
    """The machine contract documents the binary's defaults; they must be the ones asserted here."""
    path = Path(__file__).resolve().parents[1] / "pipeline" / "interface_contract.json"
    if not path.is_file():
        print(f"contract: {path} not found, defaults not cross-checked")
        return []
    documented = {entry.get("name"): entry.get("default")
                  for entry in json.loads(path.read_text())["binary"]["parameters"]}
    errors = []
    for option, expected in (("--conv-threshold", DEFAULT_CONV_THRESHOLD),
                             ("--max-iter", DEFAULT_MAX_ITERATION)):
        if documented.get(option) != expected:
            errors.append(f"{path.name} documents {option} default {documented.get(option)}, "
                          f"this gate asserts {expected}")
    print(f"contract: binary defaults --conv-threshold {documented.get('--conv-threshold')}, "
          f"--max-iter {documented.get('--max-iter')}")
    return errors


def check_defaults(name: str, default: dict, explicit: dict, radius: int, errors: list[str]) -> None:
    """A run that names no threshold, iteration cap or kernel, and gives the subset as a SIZE."""
    tag = f"{name}/{default['label']}"
    manifest = default["manifest"]
    conv = manifest.get("conv_threshold")
    if conv is None or abs(conv - DEFAULT_CONV_THRESHOLD) > 1e-9:
        errors.append(f"{tag}: default conv_threshold is {conv}, expected {DEFAULT_CONV_THRESHOLD}")
    if manifest.get("max_iteration") != DEFAULT_MAX_ITERATION:
        errors.append(f"{tag}: default max_iteration is {manifest.get('max_iteration')}, "
                      f"expected {DEFAULT_MAX_ITERATION}")
    if manifest.get("subset_radius") != radius:
        errors.append(f"{tag}: --subset-size {2 * radius + 1} ran with subset_radius "
                      f"{manifest.get('subset_radius')}, expected {radius}")
    engine = manifest.get("engine", {})
    if engine.get("icgn_backend") == "cuda" and engine.get("icgn_kernel") != "warp":
        errors.append(f"{tag}: default icgn_kernel is {engine.get('icgn_kernel')}, expected warp")
    differing = same_payloads(default, explicit)
    if differing:
        errors.append(f"{tag}: output differs from the run with radius {radius}, --conv-threshold "
                      f"{DEFAULT_CONV_THRESHOLD}, --max-iter {DEFAULT_MAX_ITERATION} spelled out: "
                      f"{differing}")


def run_case(name: str, case: dict, root: Path, runner: Runner) -> tuple[list[str], dict]:
    started = time.monotonic()
    truth = speckle.build_sequence(root, case["width"], case["height"], case["motions"],
                                   contrast=case["contrast"], noise_sigma=case["noise"])
    render_seconds = time.monotonic() - started
    motions, radius, grid = case["motions"], case["radius"], case["grid"]
    size = ["--subset-size", str(2 * radius + 1)]
    spelled = ["--conv-threshold", str(DEFAULT_CONV_THRESHOLD), "--max-iter", str(DEFAULT_MAX_ITERATION)]
    errors: list[str] = []
    report: dict = {"render_seconds": round(render_seconds, 2), "runs": {},
                    "reference_sha256": truth["reference"]["sha256"]}
    print(f"== case {name}: {case['width']}x{case['height']} px, radius {radius}, grid {grid}, "
          f"{len(motions)} frames, contrast {case['contrast']}, noise {case['noise']}; evidence {root}")

    def accept(run: dict, run_motions: list) -> None:
        frames = evaluate(case, run, run_motions, radius)
        print_frames(name, run, frames)
        check_limits(name, run, frames, len(run_motions), LIMITS[name], errors)
        block = ITERATION_BLOCK.search(run["log"])
        report["runs"][run["label"]] = {
            "engine": run["manifest"].get("engine", {}), "frames": frames,
            "iteration_block": int(block.group(1)) if block else None,
            "payload_sha256_of_sha256s": payload_digest(run)}

    explicit = runner.multi(root, "explicit", [str(radius)], grid, spelled)
    accept(explicit, motions)
    cuda = explicit["manifest"].get("engine", {}).get("icgn_backend") == "cuda"
    report["backend"] = explicit["manifest"].get("engine", {}).get("icgn_backend")

    if case.get("defaults"):
        default = runner.multi(root, "default", size, grid, [])
        accept(default, motions)
        check_defaults(name, default, explicit, radius, errors)
        loose = runner.multi(root, "loose", [str(radius)], grid,
                             ["--conv-threshold", str(LOOSE_CONV_THRESHOLD)])
        if not same_payloads(loose, explicit):
            errors.append(f"{name}: --conv-threshold {LOOSE_CONV_THRESHOLD} gives the same bytes as "
                          f"{DEFAULT_CONV_THRESHOLD}; the case cannot see the threshold and is void")
        last = len(motions) - 1
        target = truth["frames"][last]["file"]
        single_explicit = runner.single(root, "single-explicit", target, [str(radius)], grid, spelled)
        accept(single_explicit, [motions[last]])
        single_default = runner.single(root, "single-default", target, size, grid, [])
        accept(single_default, [motions[last]])
        check_defaults(name, single_default, single_explicit, radius, errors)

    if cuda:
        legacy = runner.multi(root, "legacy", [str(radius)], grid, spelled + ["--kernel", "legacy"])
        accept(legacy, motions)
        if legacy["manifest"].get("engine", {}).get("icgn_kernel") != "legacy":
            errors.append(f"{name}/legacy: the manifest does not say icgn_kernel legacy")
        block = report["runs"]["legacy"]["iteration_block"]
        if radius >= 28 and block != 32:
            print(f"   note: iteration_block is {block}, not 32, on this device; the legacy "
                  f"kernel's 32-thread path is not exercised here")
    else:
        print(f"-- {name}/legacy: not run, the legacy kernel exists only in CUDA builds")

    for message in errors:
        print(f"FAIL [{name}]: {message}")
    print(f"case {name}: {'FAIL' if errors else 'PASS'} "
          f"({time.monotonic() - started:.1f} s, of which rendering {render_seconds:.1f} s)")
    report["errors"] = errors
    return errors, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path,
                        help="opencorr_cli_cuda_mvp from any preset (CUDA or CPU-only)")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--case", choices=["all"] + sorted(CASES), default="all")
    parser.add_argument("--work-dir", type=Path,
                        help="evidence directory (must not exist, or be empty)")
    args = parser.parse_args()

    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise SystemExit(f"--work-dir {root} is not empty")
    else:
        root = Path(tempfile.mkdtemp(prefix="opencorr-subpixel-accuracy-gate-"))
    print("evidence:", root, flush=True)
    runner = Runner(args.binary, args.gpu_device, args.threads)
    names = sorted(CASES) if args.case == "all" else [args.case]
    summary = {"binary": str(args.binary), "cases": {}}
    failed = []
    summary["contract_errors"] = contract_errors()
    for message in summary["contract_errors"]:
        print(f"FAIL [contract]: {message}")
    if summary["contract_errors"]:
        failed.append("contract")
    for name in names:
        errors, result = run_case(name, CASES[name], root / name, runner)
        summary["cases"][name] = result
        if errors:
            failed.append(name)
    summary["pass"] = not failed
    (root / "summary.json").write_text(json.dumps(json_safe(summary), indent=1) + "\n")
    if failed:
        print(f"SUBPIXEL ACCURACY GATE FAIL: {', '.join(failed)}")
        return 1
    print("SUBPIXEL ACCURACY GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
