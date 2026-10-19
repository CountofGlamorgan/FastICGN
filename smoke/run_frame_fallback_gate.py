#!/usr/bin/env python3
# Copyright (c) 2026 Chonglang Wang
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Assertion gate for the skip-FFTCC whole-frame fallback (review-2026-10-03 S1-1).

On the default skip-FFTCC path every target after the first is solved by ICGN
seeded from the previous frame. When that attempt leaves too many bad POIs, the
CLI recomputes the frame from scratch (FFTCC from a zero initial guess, then
ICGN) and merges the two per POI: a good skip result is kept, a bad one is
replaced under the POI-level repair rule, a -3 one only by a good from-scratch
result (engine.frame_fallback_policy "per-poi-merge-v1", docs/output-format.md).

Two synthetic cases, standard-library Python only: Gaussian speckles rendered
at the exact sub-pixel displacement, quantised to 8 bit with sensor noise, so
every POI has a known true displacement.

cascade  192x144 px, radius 7: rigid translation growing by 0.5 px per frame
         from 3.25 px to 9.75 px, past the roughly radius - 2 px that FFTCC from a
         zero guess captures reliably, with a 40x40 px occluder (5.8 % of the
         image) on frame 6. A fallback that overwrites the skip attempt makes
         every later frame worse; the pre-merge CLI (1ca731b) ends this case at
         0.049 valid coverage, with high-confidence errors of about 11 px.
jump     320x240 px, radius 15: a 10 px jump between the first two frames. The
         skip attempt starts 10 px from the truth; near the image edges its
         iterate leaves the image (-3) although the subset is inside it at the
         true displacement, and the from-scratch pass solves those POIs. The
         pre-merge CLI took the from-scratch pass wholesale and passes this case
         (0.9997 coverage on the jump frame); the case guards that the merge is
         not worse than that. A merge that never replaced a -3 reached 0.949.

Each case runs the CLI three times: default flags with --threads 4, the same
with --threads 1, and --no-skip-fftcc. The from-scratch pass of a fallback frame
is the --no-skip-fftcc computation of that frame (on the pre-merge CLI every
fallback frame is byte-identical to the no-skip frame), so the no-skip run shows
the candidates the merge chose from. Assertions, per case:

1. at least one frame took the whole-frame fallback, otherwise the case is void;
2. on every fallback frame the bad ratio after the merge is at most that of the
   skip attempt that triggered it (both are printed on the [FALLBACK] line);
3. on every fallback frame every POI that the from-scratch pass resolves as good
   (finite zncc >= 0.8) is good in the output, so none is left at -3;
4. the frame payloads do not depend on the thread count;
5. valid coverage (finite zncc >= 0, over the POIs whose subset is inside the
   image at the true displacement) reaches the case floor: 0.90 on the last
   frame (cascade), 0.99 on every frame from the jump on (jump);
6. valid POIs match the truth: on every frame the median error is at most
   0.02 px and at least 99 % of them lie within 0.1 px, and no POI has
   zncc >= 0.8 with an error above 1 px. On the occluded frame, POIs whose
   target subset overlaps the occluder have no ground truth and are left out.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import struct
import subprocess
import tempfile
from pathlib import Path

RECORD = struct.Struct("<iifffffff")  # x y u v ux uy vx vy zncc
GOOD_ZNCC = 0.8                       # kSkipFftccZnccThreshold in the CLI
HIGH_CONFIDENCE_ERROR_PX = 1.0
TOLERANCE_PX = 0.1
MIN_WITHIN_FRACTION = 0.99
MAX_MEDIAN_ERROR_PX = 0.02
SPECKLE_SIGMA = 1.6
SPECKLE_DENSITY = 0.08
NOISE_SIGMA = 2.0
SEED = 20261003
FALLBACK_LINE = re.compile(r"\[FALLBACK\] (\S+) .*?\sbad_ratio=([^,\s]+),"
                           r".*\spost_fallback_bad_ratio=([^,\s]+),")

CASCADE_TRUTH = [(3.25 + 0.5 * k, -0.4 + 0.05 * k) for k in range(14)]
JUMP_TRUTH = [(0.25, 0.1), (10.25, 0.1), (10.75, 0.1), (11.25, 0.1)]
CASES = {
    "cascade": {"width": 192, "height": 144, "radius": 7, "grid": 2,
                "truth": CASCADE_TRUTH, "occluded_frame": 6, "occluder": (76, 52, 40),
                "floor": 0.90, "floor_from": len(CASCADE_TRUTH) - 1},
    "jump": {"width": 320, "height": 240, "radius": 15, "grid": 4,
             "truth": JUMP_TRUTH, "occluded_frame": None, "occluder": None,
             "floor": 0.99, "floor_from": 1},
}


# ---------------------------------------------------------------- sequence

def make_speckles(seed: int, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
                  density: float) -> list[tuple[float, float, float]]:
    rng = random.Random(seed)
    count = int(round(density * (x_hi - x_lo) * (y_hi - y_lo)))
    return [(rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), rng.uniform(0.5, 1.0))
            for _ in range(count)]


def render(width: int, height: int, speckles, dx: float, dy: float,
           sigma: float) -> list[float]:
    """Sum of Gaussian speckles translated by (dx, dy): exact at sub-pixel shifts."""
    image = [0.0] * (width * height)
    reach = 4.0 * sigma
    scale = -0.5 / (sigma * sigma)
    for sx, sy, amplitude in speckles:
        cx, cy = sx + dx, sy + dy
        x0, x1 = max(0, math.ceil(cx - reach)), min(width - 1, math.floor(cx + reach))
        y0, y1 = max(0, math.ceil(cy - reach)), min(height - 1, math.floor(cy + reach))
        if x0 > x1 or y0 > y1:
            continue
        wx = [math.exp(scale * (x - cx) ** 2) for x in range(x0, x1 + 1)]
        for y in range(y0, y1 + 1):
            wy = amplitude * math.exp(scale * (y - cy) ** 2)
            lo, hi = y * width + x0, y * width + x1 + 1
            image[lo:hi] = [a + wy * w for a, w in zip(image[lo:hi], wx)]
    return image


def quantize(image: list[float], mean: float, std: float, noise_sigma: float,
             rng: random.Random) -> bytearray:
    gain = 40.0 / std
    out = bytearray(len(image))
    for i, value in enumerate(image):
        gray = 128.0 + gain * (value - mean) + rng.gauss(0.0, noise_sigma)
        out[i] = 0 if gray <= 0.0 else 255 if gray >= 255.0 else int(gray + 0.5)
    return out


def write_pgm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)


def build_sequence(root: Path, case: dict) -> None:
    width, height, truth = case["width"], case["height"], case["truth"]
    reach = 4.0 * SPECKLE_SIGMA + 1.0
    lo_x = -reach - max(0.0, max(u for u, _ in truth))
    hi_x = width + reach - min(0.0, min(u for u, _ in truth))
    lo_y = -reach - max(0.0, max(v for _, v in truth))
    hi_y = height + reach - min(0.0, min(v for _, v in truth))
    speckles = make_speckles(SEED, lo_x, hi_x, lo_y, hi_y, SPECKLE_DENSITY)

    reference = render(width, height, speckles, 0.0, 0.0, SPECKLE_SIGMA)
    mean = sum(reference) / len(reference)
    std = math.sqrt(sum((v - mean) ** 2 for v in reference) / len(reference))
    write_pgm(root / "ref.pgm", width, height,
              quantize(reference, mean, std, NOISE_SIGMA, random.Random(SEED + 100)))
    occluder_image = None
    if case["occluder"] is not None:
        foreign = make_speckles(SEED + 1, -reach, width + reach, -reach, height + reach,
                                SPECKLE_DENSITY)
        occluder_image = render(width, height, foreign, 0.0, 0.0, SPECKLE_SIGMA)

    lines = []
    for k, (u, v) in enumerate(truth):
        image = render(width, height, speckles, u, v, SPECKLE_SIGMA)
        if k == case["occluded_frame"]:
            ox, oy, side = case["occluder"]
            for y in range(oy, oy + side):
                row = y * width
                image[row + ox:row + ox + side] = occluder_image[row + ox:row + ox + side]
        pixels = quantize(image, mean, std, NOISE_SIGMA, random.Random(SEED + 101 + k))
        name = f"tar_{k:03d}.pgm"
        write_pgm(root / name, width, height, pixels)
        lines.append(f"{root / name} frame_{k:06d}.csv\n")
    (root / "tar_list.txt").write_text("".join(lines))


# ---------------------------------------------------------------- evaluation

def read_raw(path: Path) -> list[tuple]:
    blob = path.read_bytes()
    if len(blob) % RECORD.size:
        raise SystemExit(f"{path}: {len(blob)} bytes is not a whole number of records")
    return list(RECORD.iter_unpack(blob))


def is_bad(zncc: float) -> bool:
    """isBadPoiForFallback(): -3 is not counted, otherwise non-finite or below 0.8."""
    return zncc != -3.0 and (not math.isfinite(zncc) or zncc < GOOD_ZNCC)


def is_good(zncc: float) -> bool:
    return math.isfinite(zncc) and zncc >= GOOD_ZNCC


def is_valid(zncc: float) -> bool:
    return math.isfinite(zncc) and zncc >= 0.0


def subset_in_range(x: int, y: int, u: float, v: float, case: dict) -> bool:
    """Target subset inside the interpolatable range at the true displacement, 1 px slack."""
    r = case["radius"]
    return (x + u - r >= 2.0 and y + v - r >= 2.0
            and x + u + r < case["width"] - 3.0 and y + v + r < case["height"] - 3.0)


def subset_hits_occluder(x: int, y: int, u: float, v: float, case: dict) -> bool:
    ox, oy, side = case["occluder"]
    r = case["radius"]
    return (x + u + r >= ox and x + u - r < ox + side
            and y + v + r >= oy and y + v - r < oy + side)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def run_cli(binary: Path, root: Path, case: dict, label: str, threads: int,
            gpu_device: int, mode: list[str]) -> tuple[Path, str]:
    out_dir = root / f"out-{label}"
    log = root / f"cli-{label}.log"
    command = [str(binary), "--multi-frame", "--ref", str(root / "ref.pgm"),
               "--tar-list", str(root / "tar_list.txt"), "--output-dir", str(out_dir),
               str(case["radius"]), str(case["grid"]), "--threads", str(threads),
               "--gpu-device", str(gpu_device), "--conv-threshold", "0.001",
               "--output-format", "raw"] + mode
    with (root / "commands.txt").open("a") as handle:
        handle.write(" ".join(command) + "\n")
    with log.open("w") as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"FAIL: CLI exited {rc}; see {log}")
    return out_dir, log.read_text(errors="replace")


def run_case(name: str, case: dict, root: Path, args) -> tuple[list[str], dict]:
    root.mkdir(parents=True)
    build_sequence(root, case)
    skip = ["--skip-fftcc", "--poi-fallback"]
    out, log = run_cli(args.binary, root, case, "default", args.threads, args.gpu_device, skip)
    out_t1, log_t1 = run_cli(args.binary, root, case, "threads1", 1, args.gpu_device, skip)
    out_ns, _ = run_cli(args.binary, root, case, "noskip", args.threads, args.gpu_device,
                        ["--no-skip-fftcc"])

    def fallback_lines(text: str) -> dict:
        found = {}
        for line in text.splitlines():
            match = FALLBACK_LINE.search(line)
            if match:
                found[match.group(1)] = (float(match.group(2)), float(match.group(3)))
        return found

    printed = fallback_lines(log)
    manifest = json.loads((out / "manifest.opencorr-raw.json").read_text())
    engine = manifest.get("engine", {})
    truth = case["truth"]
    errors: list[str] = []
    if len(manifest["frames"]) != len(truth):
        errors.append(f"manifest lists {len(manifest['frames'])} frames, expected {len(truth)}")
    if set(printed) != set(fallback_lines(log_t1)):
        errors.append("--threads 1 took the whole-frame fallback on different frames")

    print(f"== case {name}: {case['width']}x{case['height']} px, radius {case['radius']}, "
          f"grid {case['grid']}, {len(truth)} frames; evidence {root}")
    print(f"{'frame':>5} {'u_true':>6} {'v_true':>6} {'fb':>2} {'trigger':>9} {'after':>9} "
          f"{'valid':>6} {'med_err':>7} {'p99_err':>7} {'max_err':>7} {'>tol':>4} "
          f"{'>0.5px':>6} {'hiconf':>6} {'lost':>4} {'lost-3':>6}")
    report = []
    for k, frame in enumerate(manifest["frames"][:len(truth)]):
        u_true, v_true = truth[k]
        tag = frame["output_name"]
        payload = (out / frame["file"]).read_bytes()
        if payload != (out_t1 / frame["file"]).read_bytes():
            errors.append(f"{tag}: payload differs between --threads {args.threads} and 1")
        rows = read_raw(out / frame["file"])
        scratch = read_raw(out_ns / frame["file"])
        if [r[:2] for r in rows] != [r[:2] for r in scratch]:
            errors.append(f"{tag}: POI order differs from the --no-skip-fftcc run")
            scratch = []
        bad = sum(is_bad(r[8]) for r in rows)
        measurable = [r for r in rows if subset_in_range(r[0], r[1], u_true, v_true, case)]
        valid_ratio = (sum(is_valid(r[8]) for r in measurable) / len(measurable)
                       if measurable else float("nan"))
        occluded = k == case["occluded_frame"]
        errs = []
        hiconf = 0
        for x, y, u, v, _ux, _uy, _vx, _vy, zncc in rows:
            if not is_valid(zncc):
                continue
            if occluded and subset_hits_occluder(x, y, u_true, v_true, case):
                continue
            err = max(abs(u - u_true), abs(v - v_true))
            errs.append(err)
            if zncc >= GOOD_ZNCC and err > HIGH_CONFIDENCE_ERROR_PX:
                hiconf += 1
        within = sum(e <= TOLERANCE_PX for e in errs)
        fallback = tag in printed
        lost = lost_m3 = 0
        if fallback:
            for mine, theirs in zip(rows, scratch):
                if is_good(theirs[8]) and not is_good(mine[8]):
                    lost += 1
                    lost_m3 += mine[8] == -3.0
        trigger, after = printed.get(tag, (None, None))
        stats = {
            "frame": tag, "u_true": u_true, "v_true": v_true,
            "fallback_used": frame.get("fallback_used"),
            "trigger_bad_ratio": trigger, "after_bad_ratio": after, "rows": len(rows), "bad": bad,
            "bad_ratio_recomputed": bad / len(rows) if rows else 0.0,
            "bad_ratio_manifest": frame.get("bad_ratio"), "measurable": len(measurable),
            "valid_ratio_measurable": valid_ratio, "valid_evaluated": len(errs),
            "within_tolerance": within, "median_error_px": percentile(errs, 0.5),
            "p99_error_px": percentile(errs, 0.99), "max_error_px": max(errs, default=float("nan")),
            "over_half_px": sum(e > 0.5 for e in errs), "high_confidence_wrong": hiconf,
            "scratch_good_lost": lost, "scratch_good_left_minus3": lost_m3,
        }
        report.append(stats)
        print(f"{k:>5} {u_true:>6.2f} {v_true:>6.2f} {int(fallback):>2} "
              f"{'-' if trigger is None else f'{trigger:.6f}':>9} "
              f"{'-' if after is None else f'{after:.6f}':>9} {valid_ratio:>6.4f} "
              f"{stats['median_error_px']:>7.4f} {stats['p99_error_px']:>7.4f} "
              f"{stats['max_error_px']:>7.3f} {len(errs) - within:>4} {stats['over_half_px']:>6} "
              f"{hiconf:>6} {lost:>4} {lost_m3:>6}{'  occluded' if occluded else ''}")

        # Bookkeeping: the manifest QA and the [FALLBACK] line describe these bytes.
        manifest_bad = frame.get("bad_ratio")
        if manifest_bad is None or abs(manifest_bad - stats["bad_ratio_recomputed"]) > 1e-5:
            errors.append(f"{tag}: manifest bad_ratio {frame.get('bad_ratio')} != recomputed "
                          f"{stats['bad_ratio_recomputed']:.6f}")
        if bool(frame.get("fallback_used")) != fallback:
            errors.append(f"{tag}: manifest fallback_used disagrees with the [FALLBACK] line")
        if fallback:
            if abs(after - stats["bad_ratio_recomputed"]) > 1e-5:
                errors.append(f"{tag}: logged post-fallback bad ratio {after} != recomputed "
                              f"{stats['bad_ratio_recomputed']:.6f}")
            if after > trigger:  # assertion 2
                errors.append(f"{tag}: the fallback raised the bad ratio {trigger} -> {after}")
            if lost:  # assertion 3
                errors.append(f"{tag}: {lost} POIs the from-scratch pass solved (zncc >= "
                              f"{GOOD_ZNCC}) are not good in the output, {lost_m3} of them -3")
        if k >= case["floor_from"] and not valid_ratio >= case["floor"]:  # assertion 5
            errors.append(f"{tag}: valid coverage {valid_ratio:.4f} < floor {case['floor']}")
        if errs:  # assertion 6
            if stats["median_error_px"] > MAX_MEDIAN_ERROR_PX:
                errors.append(f"{tag}: median error {stats['median_error_px']:.4f} px > "
                              f"{MAX_MEDIAN_ERROR_PX}")
            if within < MIN_WITHIN_FRACTION * len(errs):
                errors.append(f"{tag}: only {within}/{len(errs)} valid POIs within "
                              f"{TOLERANCE_PX} px")
        if hiconf:
            errors.append(f"{tag}: {hiconf} POIs with zncc >= {GOOD_ZNCC} and error > "
                          f"{HIGH_CONFIDENCE_ERROR_PX} px")

    fallback_frames = [s["frame"] for s in report if s["trigger_bad_ratio"] is not None]
    print(f"fallback frames: {len(fallback_frames)} {fallback_frames}")
    if not fallback_frames:  # assertion 1
        errors.append("no frame took the whole-frame fallback; the case is void")
    for message in errors:
        print(f"FAIL [{name}]: {message}")
    print(f"case {name}: {'FAIL' if errors else 'PASS'}")
    return errors, {"engine": engine, "frames": report, "fallback_frames": fallback_frames,
                    "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path,
                        help="opencorr_cli_cuda_mvp from any preset (CUDA or CPU-only)")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--case", choices=["all"] + sorted(CASES), default="all")
    parser.add_argument("--work-dir", type=Path, help="evidence directory (must not exist)")
    args = parser.parse_args()

    if args.work_dir:
        root = args.work_dir
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="opencorr-frame-fallback-gate-"))
    print("evidence:", root, flush=True)
    names = sorted(CASES) if args.case == "all" else [args.case]
    summary = {"binary": str(args.binary), "cases": {}}
    failed = []
    for name in names:
        errors, result = run_case(name, CASES[name], root / name, args)
        summary["cases"][name] = result
        if errors:
            failed.append(name)
    engine = next(iter(summary["cases"].values()))["engine"]
    print(f"binary: {args.binary} git_hash={engine.get('git_hash')} "
          f"icgn_backend={engine.get('icgn_backend')} fftcc_backend={engine.get('fftcc_backend')} "
          f"frame_fallback_policy={engine.get('frame_fallback_policy', '<absent>')}")
    summary["pass"] = not failed
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if failed:
        print(f"FRAME FALLBACK GATE FAIL: {', '.join(failed)}")
        return 1
    print("FRAME FALLBACK GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
