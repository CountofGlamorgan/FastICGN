# Quality checks: can this result be trusted?

A finished run proves only that the process ended. This page lists the checks
that tell you whether the numbers mean anything, what each one costs, and which
of them the tools already do for you. Field names refer to the run summary
written by `--json-summary` and to the machine contract
([`pipeline/interface_contract.json`](../pipeline/interface_contract.json),
printed by `python3 pipeline/dic_pipeline.py --describe`). Output layouts are in
[output-format.md](output-format.md).

The example numbers were measured on 2026-10-03 with the production build at
commit `1ca731b`, except the 12-bit figures in section 4 and the 20 px example
in section 7, which were measured with a build of commit `4b5c50f` (it adds the
per-POI whole-frame fallback merge, `frame_fallback_policy`), and the saturation
figures in section 5, which say where they come from. The datasets named in
the examples (`config1` 10/30/40 kN events, `lowrough`, `control-allpc`,
`30kN/event10`) are the author's laboratory sequences (see `docs/lab/`); the
commit hashes are from the author's private development history and do not
resolve in the public repository. Treat all of the numbers as orders of
magnitude, not as acceptance limits.

## 1. Run-level signals and what they miss

Read these first, in this order:

1. **Exit code.** With `--strict`, 0 means every event has status `ok`, 2 that
   the dry-run checks refused the arguments before anything ran, and 3 that the
   run finished with at least one failed event. Without `--strict` the
   historical codes stay, and 0 does **not** mean success: a campaign
   (`--photo-dir`) exits 0 even when events failed, and a single event whose COD
   came out empty exits 0 too (its status is `fail`). MCP plans and runs always
   use `--strict`; a run reports the exit code as `mcp.job.exit_code`, and its
   `ok` is false whenever an event failed.
2. **`verdict`** — `ok`, `warn` or `fail`, decided by the severities in
   `warning_details` only: `fail` if any event failed, otherwise `warn` if any
   entry has severity `warn`, otherwise `ok`. Entries of severity `info` never
   change it.
3. **`events[].status` and the warnings of every event.** Two lists sit side by
   side: `warnings` keeps its earlier meaning, the warn and fail entries only,
   with codes that already existed in their earlier exact text (so a script that
   accepts `input_8bit_max_255: possible saturation_or_high_bit_truncation` and
   nothing else still works); `warning_details` lists every entry, information
   included, with its severity and the full detail. The top-level lists carry
   only binary, provenance, argument and `--strict` pre-check entries, so they
   can be empty while `verdict` is `warn`. Each code, its severity and what to do
   about it are listed in the contract under
   `outputs.json_summary.warning_vocabulary`.
4. **`events[].valid_ratio`** (pooled fraction of rows with finite
   `zncc >= 0`; the pipeline warns below 0.95) and **`events[].zncc_median`**
   (median `zncc` of valid rows; warns below 0.9). These are fixed heuristics,
   applied the same way in full-field and COD mode, not limits calibrated for
   your data.

What the pipeline now checks for you after a COD run (`events[].cod_stats`):

- **An empty or all-zero COD.** A measurement line outside the image or inside
  the binary margin is skipped by the binary (`binary_warning`, then
  `cod_line_missing`); a reversed `--roi-cols` selects no column; both leave a
  header-only CSV, which is `cod_empty` and makes the event `fail` (its frame
  CSVs are deleted as after any aggregation, so a corrected rerun into the same
  directory starts afresh; `--keep-frames` or `--strict` keep them). `--gauge 0`
  or `1` puts both lines on one row and gives
  `cod_all_zero` (warn; fail with `--strict`). Normal execution still runs with
  such arguments but reports each as an `argument_warning`; `--strict` refuses
  them up front, like `--dry-run`.
- **A COD CSV left over from an earlier run.** A failed event never moves or
  deletes an older CSV at its path, but `events[].cod_csv` is then `null` and
  `stale_cod_csv` names the file; a replaced older file is `cod_csv_replaced`.
  The old file keeps its name (`<event>.csv`, `cod_result.csv`), so a script
  that only checks the exit code (0 for a campaign even with failed events) and
  then globs the CSVs reads the earlier run's result: check `events[].status`,
  or use `--strict` and a new `--output-dir` per run.
- **A 16-bit file read as 8-bit.** Without `--input-depth`, OpenCV converts the
  file to 8 bits before the binary sees it. The dry-run warns
  (`input_mode_differs_from_probe`), the summary warns (`input_8bit_max_255`
  when a pixel reaches 255), and `--strict` refuses the run
  (`input_depth_truncates_container`, with the probe's recommended
  `--input-depth`/`--input-align`).
- **Saturation.** `input_8bit_max_255` is `info` while the graded fraction of
  pixels at 255 is at most `--saturation-warn-fraction` (default 0.05) and
  `warn` above it. COD mode grades the measurement-line bands, the pixels of
  the subsets the COD uses, and the largest band decides; full-field mode
  grades the whole reference frame. The numbers are in
  `events[].input_saturation`, and the dry-run forecasts the level (section 5).

The binary's complete stderr is kept next to the frame directory
(`events[].binary_stderr_log`); a failed event carries the exit code and the
binary's first `Error:` line in `gpu_cli_failed` and one actionable sentence in
`events[].remediation`.

Use a new `--output-dir` for every run, then read the summary:

```bash
python3 - "$OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print("verdict:", s["verdict"], "| top level:", [(w["code"], w["severity"]) for w in s["warning_details"]])
for e in s["events"]:
    print(e["name"], e["status"], e["cod_mode"], e["frames"], e["valid_ratio"], e["zncc_median"],
          [(w["code"], w["severity"]) for w in e["warning_details"]])
    c = e["cod_stats"]
    if c:
        print(f"  COD rows {c['rows']}, frames with points {c['frames_with_points']}/{c['frames']}, "
              f"x points {c['x_points']}, all zero {c['all_zero']}, "
              f"max {c['max_um']} um at frame {c['max_frame']} x={c['max_x_px']} px "
              f"(naive there {c['naive_at_max_um']}), |COD| p95 {c['abs_p95_um']} um")
PY
```

`cod_mode` is the aggregation used: `naive` by default since 2026-10-05, when
`naive there` prints `None` because the maximum is itself the naive value;
it is filled only for `--cod-method fillboundary` (section 6).

Expect every target frame to have points (`frames_with_points` equals
`frames`), about `frames x x points` rows (fewer where a line had no valid
point), and not all zeros. In a campaign the COD files are `$OUT/<event>.csv`.
`events[].geometry` records the lines, the extrapolation target and the gauge
that were actually used.

## 2. Bit depth

By default every image is read as 8-bit (`--input-depth 8`, OpenCV
`IMREAD_GRAYSCALE`). A 12- or 16-bit file is converted to 8 bits by OpenCV
before the engine sees it, so the engine cannot warn: the log reports
`source_depth=CV_8U, source_range=[0,255]` for a 16-bit TIFF read this way.

- Before the run, look at the dry-run's `events[].first_frame`
  (`container_bit_depth`, `pixel_max`, `low_4_bits_all_zero`) and
  `events[].input_recommendation`. A mismatch with what you requested produces
  the warning `input_mode_differs_from_probe`, but the plan stays valid.
- After the run, `events[].input_reference` in the summary records the mode
  actually used for the reference frame, and the raw manifest's `input_image`
  block the requested mode. Under the default mode neither can show that the
  source was 16-bit; the only after-the-fact hint is `input_8bit_max_255`, which
  is a `warn` whenever the reference file's container is deeper than 8 bits.
- `--input-depth auto` reads left-shifted 12-bit data whose maximum exceeds 4095
  as 16-bit. Pass `--input-depth 12 --input-align left` explicitly for that layout.

## 3. Static noise floor

Run a stretch of frames recorded before anything moves (or a deliberately
static sequence) and measure the scatter of valid points: the spatial standard
deviation of `u` and `v` within a frame, and the temporal standard deviation of
each POI across frames. For COD mode, take the spatial standard deviation of
`cod_um` within each pre-event frame. Values within about three times this
floor cannot be told apart from noise. The first five frames of a CSV run,
whatever their frame numbers:

```bash
python3 - "$OUT" <<'PY'
import glob, os, sys
import numpy as np
names = sorted(glob.glob(os.path.join(sys.argv[1], "*", "frame_*.csv")))[:5]
if not names:
    sys.exit(f"no frame_*.csv under {sys.argv[1]}/*/")
for name in names:
    d = np.genfromtxt(name, delimiter=",", names=True)
    ok = np.isfinite(d["zncc"]) & (d["zncc"] >= 0) & np.isfinite(d["u"]) & np.isfinite(d["v"])
    print(name, f"valid {ok.mean():.3f} std u {d['u'][ok].std():.4f} px std v {d['v'][ok].std():.4f} px")
PY
```

The same for a raw run (`--output-format raw`), through the published reader;
run it from the repository root:

```bash
python3 - "$OUT" <<'PY'
import glob, itertools, os, sys
sys.path.insert(0, "pipeline")                    # or pip install .
import fasticgn_io
manifests = sorted(glob.glob(os.path.join(sys.argv[1], "*", "manifest.opencorr-raw.json")))
if not manifests:
    sys.exit(f"no manifest.opencorr-raw.json under {sys.argv[1]}/*/")
for manifest in manifests:
    for frame, records in itertools.islice(fasticgn_io.iter_frames(manifest), 5):
        ok = fasticgn_io.valid_mask(records) & fasticgn_io.finite_displacement_mask(records)
        u, v = records["u"][ok], records["v"][ok]
        print(frame["output_name"], f"valid {ok.mean():.3f} std u {u.std():.4f} px std v {v.std():.4f} px")
PY
```

Reference values from one 8-bit laboratory event with subset 15, before the
event started: along a measurement line the spatial standard deviation of `u`
was 0.0024-0.0031 px, the per-POI temporal standard deviation over 49 frames
0.0019 px, and the single-point COD standard deviation in one frame
2.07 um (naive) and 8.79 um (fill-boundary, see section 6).

## 4. Known-translation self-check

`bash smoke/selfcheck.sh` proves the build on synthetic speckle. To test your
own images, shift your reference image by a known whole number of pixels and
correlate the pair:

```bash
python3 - "$REF" <<'PY'
import sys
import numpy as np
from PIL import Image
with Image.open(sys.argv[1]) as image:
    deep = image.mode.startswith("I")                        # 12/16-bit gray: keep the values
    ref = np.asarray(image).astype(np.uint16) if deep else np.asarray(image.convert("L"))
shifted = np.roll(ref, (-2, 3), axis=(0, 1))                 # content moves +3 px in x, -2 px in y
Image.fromarray(shifted).save("shifted.png")                 # PNG keeps 8 or 16 bits per pixel
Image.fromarray(ref).save("ref.png")
PY
"$BIN" ref.png shifted.png --subset-size 15 2 shift.csv $INPUT_ARGS   # INPUT_ARGS: the run's --input-* flags
python3 - <<'PY'
import numpy as np
d = np.genfromtxt("shift.csv", delimiter=",", names=True)
ok = np.isfinite(d["zncc"]) & (d["zncc"] >= 0)
off = ok & ((np.abs(d["u"] - 3) > 0.5) | (np.abs(d["v"] + 2) > 0.5))
print(f"valid {ok.mean():.3f}; median u {np.median(d['u'][ok]):+.4f} v {np.median(d['v'][ok]):+.4f}; "
      f"valid but off by > 0.5 px: {off.sum()}"
      + (f" (zncc {d['zncc'][off].min():.3f} to {d['zncc'][off].max():.3f})" if off.any() else "")
      + f"; zncc=-3 rows {(d['zncc'] == -3).sum()}")
PY
```

`$BIN` is `opencorr_cli_cuda_mvp`; the positional `2` is the grid step. A 12- or
16-bit file is kept as `uint16` and written as a 16-bit PNG, so set
`INPUT_ARGS` to the flags of the run (for example `--input-depth 12
--input-align left`); leave it empty for 8-bit data. POIs whose shifted subset
would leave the image are rejected with `zncc = -3` (the wrapped strip from
`np.roll` sits there).

On synthetic or well-textured speckle, every valid POI should reproduce
`u = +3`, `v = -2` to within about 5e-4 px. Real images can also contain
subsets that match the wrong place with a lowish `zncc`, and the `zncc >= 0`
mask keeps them:

- A laboratory 8-bit reference image, subset 15: 31 815 of 31 818 valid POIs
  matched to within 5e-4 px. The other 3 sat in the camera's black information
  bar, where the reference subset has no texture at all, and reported `zncc`
  within 1e-7 of 0.5 (0.5 in the CSV's six digits) with `u`, `v` of order
  1e-6 px; at grid step 1 the same image carried 12 such POIs, 10 of them
  with `zncc` exactly 1.0 and `u = v = 0` because the target subset was
  constant too. Since 2026-10-19 a zero-texture reference subset is rejected
  before iterating (`zncc = -3` on the CUDA backend, `-5` on the CPU backend,
  see [output format](output-format.md#negative-zncc-status-codes)), so these
  rows no longer pass the `zncc >= 0` mask; the self-check above then reports
  31 815 valid POIs, all matched.
- One 12-bit left-shifted camera image, subset 15: 9 of 101 829 valid POIs were
  wrong by 6.0-8.1 px in `u`, with `zncc` from 0.60 to 0.86; the largest error
  of the other valid POIs was 2.7e-4 px. With `--subset-size 21` no valid POI was
  wrong and the largest error was 7e-5 px. This is a single measurement on a
  single image.

So state the `zncc` floor you applied when you report a result, and if a floor
removes points you need, re-check with a larger subset. Integer shifts test the
plumbing, not sub-pixel accuracy, and this repository ships no sub-pixel
resampling tool.

A CPU/GPU comparison on your own frames is another cheap check: run the same
full-field CSV command once with `--gpu-bin` pointing at a CUDA build and once
at a `cpu-only` build (two output directories), then
`python3 smoke/check_cpu_gpu_parity.py --cpu-dir CPU_OUT/<event> --gpu-dir GPU_OUT/<event> --glob 'frame_*.csv'`
pairs the frame files by name and applies the per-frame tolerances described in
[smoke/README.md](../smoke/README.md).

## 5. Saturation

`input_8bit_max_255` says that at least one reference pixel equals 255 (read at
effective depth 8). Saturated pixels carry no speckle contrast, so what matters
is how many of them sit where the engine correlates. The warning is `info`
while the graded fraction is at most `--saturation-warn-fraction` (default
0.05) and `warn` above it:

- **COD mode** grades each measurement line's band: the pixels of the subsets
  that enter the COD aggregation, rows `L - r .. L + r` and columns
  `x0 - r .. x1 + r` for `--roi-cols x0,x1` (without it, the binary's first and
  last POI column on the line, `+/- r`), clipped to the image. The largest band
  fraction decides. A line the binary skips (inside its margin) has no band.
- **Full-field mode** grades the whole reference frame
  (`events[].input_saturated_fraction`).
- A file in a container deeper than 8 bits read as 8-bit is always `warn`:
  there a pixel at 255 hints at high-bit truncation (section 2).

The numbers are in `events[].input_saturation` of the summary: `basis`
(`line_bands` or `frame`), the graded `value`, `threshold`, `frame_fraction`,
the `severity` the warning got (`null` when no pixel is 255), and per line `y`,
`band_rows`, `band_cols`, `band_fraction`, `subsets` and the fractions of those
subsets with at least one pixel at 255 (`subsets_with_saturation`) and with
more than 10 % of their pixels at 255 (`subsets_over_10pct`). The
`warning_details` detail repeats them. The dry-run prints the same object for
each event, computed the same way, so it shows the level before anything runs.
The threshold is a heuristic, not a limit calibrated against correlation
error; tune it, and for a `warn` look at which line and how many of its subsets
are affected.

Why COD mode no longer grades the whole frame: on the laboratory frames
(1036 x 186) the whole-frame fraction is set by bright pixels far from the
measurement lines. Each of 159 reference frames surveyed on 2026-10-04 (eight
datasets) has 1,400 or more pixels at 255 in rows 0-108, 796 of them in rows
0-40 in every frame. On their own they make 0.73 % of the frame or more, and
none of them enters a subset of the canonical lines (in config1 10 kN they are
nearly all there is, and both bands hold at most 0.02 %). In the brightest
datasets saturation sits in the bands instead: in every event of config1 40 kN
(runs 1-3) and of lowrough-20260918 20 kN the lower line's band holds 2.2-2.4
times the frame fraction. Four events, measured on 2026-10-04 with the pipeline
of commit `efad739` and the production binary `git_hash=638ad1845d01` (COD
mode, `--max-frames 5`), all with the geometry the 30 kN laboratory command
used until 2026-10-05 (`--interface-y 133` with the default gauge 35, so lines
116 and 150; `--subset 15`, radius 7; `--roi-cols 72,958`, so band rows 109-123
and 143-157, band columns 65-965 and 887 subsets per line). For the other
datasets this geometry is an assumption made to compare the measure:

| Event (reference frame) | Whole frame | Band L0 / L1 | Subsets > 10 % at 255, L0 / L1 | Level at 0.05 |
| --- | ---: | ---: | ---: | --- |
| config1 10kN/event15 | 0.74 % | 0.00 % / 0.00 % | 0 % / 0 % | info |
| config1 30kN/event10 | 1.30 % | 0.02 % / 1.73 % | 0 % / 0 % | info |
| config1 40kN/run3/event04 | 3.11 % | 3.60 % / 7.11 % | 15 % / 38 % | warn |
| config2-symmetric lowrough-20260918 20kN/event16 | 3.42 % | 3.71 % / 7.98 % | 4 % / 38 % | warn |

Since 2026-10-05 the canonical 30 kN command places the lines 8 px either side
of the interface row 131 (`--interface-y 131 --gauge 16`, lines 123 and 139,
band rows 116-130 and 132-146, same columns). The same four reference frames
with that geometry (dry-run on 2026-10-05, same binary and `--roi-cols
72,958`):

| Event (reference frame) | Band L0 / L1 | Subsets > 10 % at 255, L0 / L1 | Level at 0.05 |
| --- | ---: | ---: | --- |
| config1 10kN/event15 | 0.00 % / 0.00 % | 0 % / 0 % | info |
| config1 30kN/event10 | 0.03 % / 0.81 % | 0 % / 0 % | info |
| config1 40kN/run3/event04 | 3.41 % / 5.94 % | 12 % / 30 % | warn |
| config2-symmetric lowrough-20260918 20kN/event16 | 3.60 % / 6.68 % | 2 % / 27 % | warn |

With its own geometry from `dic_processing_config.json` (interface 131, gauge
16, `--roi-cols 106,930`: lines 123 and 139), the last event's bands are 3.80 %
and 6.96 %, still `warn` (dry-run). Over all 159 frames with the 116/150 geometry,
the old rule (whole frame above 0.02) and the new one (largest band above 0.05)
disagree on two datasets: config1 40 kN run1, warn on 13 of 13 events before
and 11 of 13 now, and control-allpc 30 kN, 10 of 12 before and 2 of 12 now
(bands 2.1-4.3 % and 3.2-6.0 %). The reference frame is the only one graded;
later frames can saturate where it does not.

To measure another region, or a whole run's frames, count the pixels at the
maximum yourself (radius, ROI columns, then the rows to inspect):

```bash
python3 - "$REF" 7 72 958 123 139 <<'PY'
import sys
import numpy as np
from PIL import Image
with Image.open(sys.argv[1]) as image:
    if len(image.getbands()) > 1:
        image = image.convert("L")
    pixels = np.asarray(image)
r, x0, x1, *rows = (int(v) for v in sys.argv[2:])
left, right = max(0, x0 - r), min(pixels.shape[1] - 1, x1 + r)
top = int(pixels.max())
print(f"dtype={pixels.dtype} max={top} at-max fraction={np.mean(pixels == top):.4%}")
for y in rows:
    band = pixels[max(0, y - r):y + r + 1, left:right + 1]
    print(f"rows {y - r}..{y + r}, cols {left}..{right}: at-max fraction={np.mean(band == top):.4%}")
PY
```

When `max` is the full-scale code of your camera (255 for 8-bit data), the
at-max fraction is the saturated fraction; for the canonical 30 kN event above
(reference frame of 30kN/event10) this prints the summary's band fractions,
0.0296 % and 0.8139 % (2026-10-05); with rows 116 and 150 it printed
0.0222 % and 1.7314 %.

## 6. Fill-boundary COD and single-point extremes

COD (the column `cod_um`) is the slip across the interface: the relative
displacement of its two sides, `u` on line L0 minus `u` on line L1. It is not
an opening displacement; the name is historical.

The default COD is naive, `1000 * cal * (u_L0 - u_L1)`: naive until
2026-07-09, fill-boundary from 2026-07-09 to 2026-10-05, naive again since
2026-10-05 (summary `events[].cod_mode` says which one a run used). The
optional fill-boundary COD (`--cod-method fillboundary`) extrapolates each
line's `u` to the midpoint `y0` between the lines along that line's own
`uy = du/dy`:

```
COD = 1000 * cal * ([u_L0 + uy_L0 * (y0 - L0)] - [u_L1 + uy_L1 * (y0 - L1)])
```

so the noise of `uy` is multiplied by the extrapolation distance (17 px each
way with the default gauge, 8 px with the laboratory's 30 kN convention of
lines 8 px either side of the interface). Spatial medians are robust to this;
single points are not. One 800-frame laboratory event (subset 15, lines 17 px
either side of the interface):

| | naive | fill-boundary |
| --- | ---: | ---: |
| single-point COD standard deviation, one frame before slip | 2.07 um | 8.79 um |
| event maximum, at the same frame and column | 324.1 um | 413.3 um (x 1.275) |
| spatial median of that frame | - | x 1.041 of naive |

On a second event the fill-boundary maximum was 1.43 times the naive value at
the same point.

Closer lines amplify less. Measured on 2026-10-05 with the production binary
(`git_hash=638ad1845d01`), subset 15:

| Lines (extrapolation) | COD noise floor, naive | fill-boundary | ratio |
| --- | ---: | ---: | ---: |
| 123 / 139 (8 px each way) | 2.52 um | 5.06 um | x 2.0 |
| 116 / 151 (17.5 px each way) | 2.77 um | 9.65 um | x 3.5 |

The noise floor is the mean over 25 static-reference sequences of config1
(the `static-ref` directories, 10-40 kN; first image as reference, up to 200
targets) of the
standard deviation of every single-point COD of a sequence. On 30kN/event10 with lines
123 / 139 (`--interface-y 131 --gauge 16`) the summary gives a fill-boundary
maximum of 368.9 um (frame 625, x = 261 px) where naive is 336.6 um (x 1.10),
and a 95th percentile of |COD| of 327.3 um; with the default gauge (lines
116 / 150) the same event gives 432.2 um where naive is 301.9 um (x 1.43), and
a 95th percentile of 339.8 um. The default (naive) run of the same command
gives a maximum of 340.3 um (frame 801, x = 120 px) and a 95th percentile of
311.0 um (2026-10-05, same binary).

A line inside the interface transition band biases fill-boundary far more than
naive, because the band's `uy` is extrapolated too. On 30kN/event21, 22, 23 and
25, whose interface is at row 133-133.5, lines 123 / 139 (one line in the band)
give, over the last 10 frames, 93-94 % of the plateau jump with naive and
77-79 % with fill-boundary; with clean lines 123 / 141 naive gives 101-102 %
and fill-boundary 103-105 % (fill-boundary with the per-segment lines 125 / 141:
103-105 %). With clean lines fill-boundary is only 1.7-3.4 % above naive on
those events and on event10 (an independent cross-check of 2026-10-05, row medians). These are the reasons the default
went back to naive on 2026-10-05.

When you report an extreme value of a fill-boundary run, give the naive value
at the same point (`cod_stats.naive_at_max_um` / `naive_at_min_um`, filled
only under fill-boundary) and the 95th percentile of |COD|
(`cod_stats.abs_p95_um`) beside it, or report medians or percentiles instead.
Under the default naive those two fields are `null`: `max_um` and `min_um` are
already naive values. Other per-point values come from the frame CSVs, which
COD mode keeps only with `--keep-frames`.

## 7. Large displacements: FFTCC seeding and whole-frame fallback

FFTCC can only represent integer displacements in `[-(r-1), +r]` around its
seed for subset radius `r` (its correlation window is `2r` wide). Two review
runs on synthetic speckle found it reliable only to about `+/-(r-2)` px; that
is a measurement, not a guarantee. In multi-frame runs the binary normally
seeds each frame from the previous one. When that seeded attempt's `bad_ratio`
exceeds the baseline taken from the first target frame by more than 0.03, the
whole frame is recomputed from a zero initial guess and merged with the seeded
attempt point by point (`engine.frame_fallback_policy: per-poi-merge-v1`, see
[output-format.md](output-format.md#whole-frame-fallback)): a good seeded
result (`zncc >= 0.8`) is kept, a bad one is replaced only by a from-scratch
result with at least the same `zncc`, and a `-3` only by a good from-scratch
result. A fallback frame therefore never ends with more bad POIs than its seeded
attempt. Manifests without `frame_fallback_policy` come from earlier builds,
whose fallback kept the from-scratch result even when it was worse, so there one
disturbed frame near the subset radius could lose tracking for every later frame.

Two limits remain. The from-scratch pass still starts from zero, so it cannot
recover motion beyond its capture range. And POIs that neither pass recovers stay
bad and restart from zero on the next frame, so a fallback once triggered can fire
on every later frame, each time adding a full-frame FFTCC + IC-GN.

Raw output records this per frame. CSV runs write no manifest: there the
pipeline keeps the binary's stderr (`[FALLBACK]` lines and the closing totals)
in `events[].binary_stderr_log`, and the summary carries the totals in
`events[].binary_totals` (`frames_with_fallback_fftcc > 0` also adds the info
warning `frame_fallback_used`). For per-frame numbers use `--output-format raw`
(full-field mode) when motion is large (`$BIN` and `$INPUT_ARGS` as in
section 4):

```bash
python3 pipeline/dic_pipeline.py --image-dir "$FRAMES" --output-dir "$OUT" --output-format raw --gpu-bin "$BIN" $INPUT_ARGS
python3 - "$OUT/$(basename "$FRAMES")/manifest.opencorr-raw.json" <<'PY'
import json, sys
frames = json.load(open(sys.argv[1]))["frames"]
fallback = [f["output_name"] for f in frames if f["fallback_used"]]
low = [(f["output_name"], f["valid_ratio"]) for f in frames
       if f["valid_ratio"] is not None and f["valid_ratio"] < 0.9]
print(f"{len(frames)} frames; whole-frame fallback on {len(fallback)}: {fallback[:5]}; "
      f"valid_ratio < 0.9 on {len(low)}: {low[:5]}")
PY
```

A synthetic 20 px jump with subset 15 (radius 7), beyond that range, gave
`whole-frame fallback on 1` and a valid ratio of 9.66e-5 (4 of 41,412 POIs) for
the jump frame while the pipeline still exited 0. All four are wrong: true `u` is
20 px, measured `u` lies between -3.4 and +0.3 px, with `zncc` between 0.27 and
0.55. They are low-confidence matches from the seeded attempt that the
from-scratch pass could not beat (a build without `frame_fallback_policy` kept
one of them: 2.4e-5). In a frame with `fallback_used`, treat valid POIs with
`zncc < 0.8` as suspect. Frames with `fallback_used` followed by a collapsing
`valid_ratio` mean the sequence moved beyond what the seeding can follow; a
larger subset widens the range. The meaning of every per-frame field is in the
contract under `outputs.raw_manifest.frame_qa_semantics`.

## 8. Is the output directory complete?

- `.opencorr-state/complete.json` records that a run once completed. A later run
  that starts changing the directory (resume, append, `--repair`) does not
  invalidate it first, so after a crash it can be present while frames are
  missing or the manifest is older.
- Do not resume a directory that has `.opencorr-state/` with a binary that
  predates the ledger (built before mid-September 2026). Such a binary does not
  read the ledger, so none of its checks apply.
- A quick file-level look that changes nothing:
  `fasticgn_io.run_status("$OUT/<event>")` (Python, `pipeline/fasticgn_io.py`)
  checks that every frame the ledger's contract lists exists with the size its
  commit intent recorded, that the raw manifest lists exactly those frames, and
  that no interrupted append or crash temporary is left. It cannot compare
  content digests (XXH3) or inputs, and says so in `not_verified`.
- The full check that changes nothing:
  `opencorr_cli_cuda_mvp --verify-only "$OUT/<event>"` (binaries whose
  `--version` prints a `version=` line). It compares every payload with its
  ledger digest, the completion record, the manifests and the input
  fingerprints, computes and writes nothing, needs neither the original command
  nor the binary that wrote the directory, prints one JSON verdict and exits 0
  only for `"status": "complete"`
  ([output-integrity.md](output-integrity.md#verifying-a-directory-without-running-it)).
- Rerunning the identical command is the other full check. On a complete
  directory the binary re-reads every output and fingerprints every input,
  prints `Verified complete run; no computation or output rewrite required`
  and changes nothing. Through the pipeline a rerun of a complete full-field run
  exits 0, leaves every frame's inode and mtime as they were, keeps that line in
  `events[].binary_stderr_log`, sets `binary_totals.verified_complete_run`, and
  reports `resumed_verified` (info).
  Preconditions: the same binary, the same inputs unchanged at the same paths,
  and the same output-deciding options. `--threads`, `--preload`,
  `--decode-threads` and `--gpu-device` only schedule the work and do not count,
  except for binaries whose `--version` prints no `version=` line: those
  recorded them, so there the default `--threads`, which follows the CPUs the
  process may use, still matters (a run under `taskset` is a different run).
  On an incomplete directory the same command resumes and computes; on a
  mismatch it is rejected, nothing is changed, and newer binaries name the
  field that differs. COD mode without
  `--keep-frames` keeps no frames, so a rerun simply recomputes.
- Output is deterministic: rerunning the same command into a **new** directory
  and comparing with `cmp` is a cheap reproducibility check (the 8-bit
  canonical event gave a byte-identical `cod_result.csv` on 2026-10-03).

Directory layout, ledger and repair rules are in
[output-integrity.md](output-integrity.md).
