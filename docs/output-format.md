# FastICGN output format

Everything FastICGN writes is described here: the per-frame payloads, their
manifests, and the machine-readable pipeline summary. Nothing in this document
depends on a lab deployment. Integrity, resume and durability semantics are in
[output integrity](output-integrity.md).

## Per-frame payloads

A run writes one file per target frame. `--output-format csv` (default) writes
text, `--output-format raw` writes a little-endian binary record stream and
replaces the `.csv` extension of each output name with `.bin`. Both carry the
same nine quantities per point of interest (POI); the encoding and the field
order differ, and CSV prints floating-point values with six significant digits,
so it is a rounded copy of the float32 values raw stores exactly.

| | CSV | raw |
| --- | --- | --- |
| One frame | `frame_000002.csv` | `frame_000002.bin` |
| Header | `x,y,u,v,zncc,uy,vy,ux,vx` | none |
| Row/record | one text line per POI | one 36-byte record per POI |
| Field order | `x,y,u,v,zncc,uy,vy,ux,vx` | `x,y,u,v,ux,uy,vx,vy,zncc` |
| Manifest | `manifest.opencorr-raw.json` only in raw mode | always |

**The two field orders differ.** The CSV order is historical; the raw order is
the memory layout. Read the raw order from the manifest's `fields` array, or
use `pipeline/fasticgn_io.py`, rather than assuming either one.

### Raw record layout (`opencorr-gpu-raw-v1`)

36 bytes per POI, little-endian, in POI-queue order (row-major: `y` outer, `x`
inner, both ascending; with `--line` the rows follow the order the line values
were given, and repeated values are neither sorted nor removed):

| Offset | Field | Type | Unit |
| --- | --- | --- | --- |
| 0 | `x` | int32 | pixel (image column) |
| 4 | `y` | int32 | pixel (image row) |
| 8 | `u` | float32 | pixel |
| 12 | `v` | float32 | pixel |
| 16 | `ux` | float32 | 1/px |
| 20 | `uy` | float32 | 1/px |
| 24 | `vx` | float32 | 1/px |
| 28 | `vy` | float32 | 1/px |
| 32 | `zncc` | float32 | dimensionless |

### Coordinates, units and signs

* `x` is the image **column**, increasing to the right. `y` is the image
  **row**, increasing **downward** (image convention, not a maths y-axis).
  Both are integer pixel coordinates **in the reference image**, and they are
  the identity of a POI: the same `(x, y)` denotes the same material point in
  every frame of a fixed-reference sequence.
* `u` is the displacement along `+x`, `v` along `+y`, both in **pixels**, both
  measured from the fixed reference image to that frame's target image. The CLI
  never applies a spatial calibration; multiply by mm/px yourself.
* `ux, uy, vx, vy` are the first-order shape-function gradients
  `∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y`, in **1/px**. They are the warp parameters the
  IC-GN solver converged to, not a finite difference taken afterwards over the
  displacement field.
* `zncc` is the zero-mean normalized cross-correlation of the converged subset,
  nominally in `[-1, 1]`.

### Valid state

A row is a **valid measurement** only when `zncc` is finite and `zncc >= 0`.
This is exactly the rule the CLI's own frame QA and the pipeline's
`valid_ratio` use.

Rows with `zncc < 0` are failure/status rows: the negative value mirrors a
status code, and their `u`/`v` are retained diagnostic state (the initial
guess, the previous-frame seed, or the last failed iterate). Excluding them is
not optional — including them silently corrupts any displacement or COD
statistic. Analysis code usually also wants finite `u` and `v`; the two masks
are `fasticgn_io.valid_mask()` and `fasticgn_io.finite_displacement_mask()`.

### Negative `zncc` status codes

| Value | Meaning | `u`, `v`, `ux…vy` carry |
| --- | --- | --- |
| `-3` | Terminated before a result existed — see the two causes below the table. | the incoming guess for that frame — FFTCC output on the first frame, the previous frame's converged value afterwards |
| `-4` | Ran the full iteration budget without meeting the convergence criterion. | the last iterate |
| `-5` | The iteration diverged or produced a non-finite value. | reset to the incoming guess |

All three are excluded by `zncc >= 0`, so the mask is unchanged by anything
described here; the distinction matters only when you are reading *why* a POI
has no measurement.

`-3` has two causes, and only one of them is shared by both backends:

* **Target subset out of range** — the rule below. Identical on the CPU ICGN
  backend and in both CUDA kernels.
* **Reference-side rejection, CUDA only** — before iterating, the CUDA
  precompute kernel rejects a POI whose *reference* subset reaches the edge of
  the reference image, whose Hessian is singular, or (since 2026-10-19) whose
  reference subset has **zero texture**: every sampled value equal, so the
  subset's zero-mean norm is 0. The predicate is exact equality of the samples
  (max == min), independent of `--input-normalize`. The CPU backend has no
  equivalent guard: its entry test covers the reference subset reaching the
  edge, a displacement guess larger than the image, a non-finite guess and a
  negative incoming `zncc`, but **not** a singular Hessian, so a POI in a
  texture-free region iterates on the CPU and is rejected on the GPU. For the
  zero-texture case the two backends agree that the POI is invalid but not on
  the code: the CPU's cost is `0/0`, which its non-finite check reports as
  `-5`, while the GPU reports `-3`. Before 2026-10-19 the GPU rejected most
  zero-texture subsets through the singular Hessian, but a few whose 4-tap
  gradient stencil reached 2 px past the subset edge into texture had a merely
  ill-conditioned Hessian, iterated against a 1e-12 norm clamp and left the
  solver as a "converged" measurement with `zncc` 0.5 (textured target subset)
  or exactly 1.0 (constant target subset, `u = v = 0`); on the reference
  sequence below that was 12 POIs per frame, all in the camera's black
  information bar. `smoke/run_zero_texture_gate.py` (ctest `zero_texture`)
  asserts the rule on both backends.

So the two backends' `-3` counts are not comparable as totals, and a difference
between them is not by itself a parity failure. On this project's reference
sequence (the author's laboratory dataset, `30 kN/event10` reference frame,
radius 7, step 1: 171 024 POI) the
CUDA side carries 37 625 reference-side rejections per frame, 22.0 % of the
POI, that have no CPU counterpart: 37 613 measured on 2026-09-17 (singular
Hessian), plus the 12 zero-texture POIs per frame the 2026-10-19 rule adds,
measured as the only records that differ between the two builds' 60-frame and
800-frame anchor trees. It holds for any later build whose production anchors
(`smoke/run_production_anchors.sh`) are unchanged. It depends on the
reference image and the subset size.

`-3` is also **terminal**: `zncc == -3` is excluded from the frame `bad_ratio`,
is not a POI-level fallback candidate, and a fallback pass that itself ends in
`-3` does not replace the primary result. The practical consequence is that a
POI whose fallback trajectory would have wandered out of the image and come
back is no longer recovered by the fallback; on the reference sequence that
costs nothing measurable (every record affected is invalid on both sides of the
change), but it is a real behavioural rule and not an accident. The one
exception is the whole-frame fallback, which replaces a `-3` left by the
skip attempt when its from-scratch pass resolves that POI as good; see
[Whole-frame fallback](#whole-frame-fallback).

### Out-of-bounds target subsets

A deformation guess can push part of the warped target subset outside the
target image. The bicubic B-spline interpolator cannot evaluate closer than two
pixels to any edge, so a sample at `(gx, gy)` is unavailable whenever

```
gx < 1  ||  gy < 1  ||  gx >= width - 2  ||  gy >= height - 2
```

If **any** of the subset's samples is unavailable, the POI is rejected with
`zncc = -3` at that iteration, before the subset's mean, norm, ZNSSD cost or
Newton step are formed. `u`, `v` and the gradients keep the incoming guess and
the iteration count is reported as zero — the POI is treated as never having
produced a result, not as a converged or a non-converged one.

The alternative — substituting a fabricated intensity for the unavailable
samples and continuing — produces a `zncc` computed partly from values that are
not in either image. That is what upstream OpenCorr did until commit `baa3de6`
(with the interpolator's `-1.f` sentinel) and what this project's CUDA kernels
did until the same rule was ported into them (with `0.0f`). Both now reject
instead. In practice such a POI almost always ended at `-4` anyway, having
burned its whole iteration budget on corrupted data; the visible change is that
the status now names the real cause, and the small number of POIs that used to
*converge* on a partly fabricated subset — and so passed the `zncc >= 0` mask —
no longer do.

This rule is identical on the CPU and CUDA ICGN backends and in both CUDA
iteration kernels (`--kernel warp` and `--kernel legacy`): the same predicate,
evaluated at the same point of the same iteration. `smoke/run_oob_rejection_gate.py`
asserts it against whichever backend a build produced, and fails a build that
has lost it.

A **NaN** sample coordinate is not this rule. It means the iterate itself went
non-finite, and every backend reports it as `-5`, not `-3`. Out of range is
tested first, so a subset that is both gets `-3`.

### Whole-frame fallback

Multi-frame runs with `--skip-fftcc` (the default) run FFTCC only on the first
target. Every later target is solved by IC-GN seeded from the previous frame's
result; a POI that failed there (`zncc < 0`) restarts from zero. A POI is
**bad** when its `zncc` is non-finite or below 0.8 (`-3` is not counted), and
the first target's `bad_ratio` plus 0.03 is the threshold for the whole run.
When the skip attempt's bad fraction exceeds it, the frame is recomputed from
scratch — FFTCC from a zero initial guess, then IC-GN — and the two results are
merged POI by POI (`frame_fallback_policy: per-poi-merge-v1`):

| Skip-attempt result | Replaced by the from-scratch result when |
| --- | --- |
| good: finite `zncc >= 0.8` | never |
| bad | the from-scratch `zncc` is finite, not `-3` and at least as high (the POI-level repair rule) |
| `-3` | the from-scratch result is good |

No rule adds a bad POI, so a fallback frame never ends with more bad POIs than
the attempt that triggered it. The manifest's `bad_ratio` is the merged value;
the triggering one is printed only on the `[FALLBACK]` stderr line. Replacing a
`-3` departs from the POI-level repair, where `-3` stays terminal: after a jump
between frames the seed is far from the truth and the iterate leaves the image
although the subset lies inside it at the true displacement.

The merge exists because FFTCC from a zero guess captures reliably only about
±(radius − 2) px. Near that limit the from-scratch pass is far worse than the
skip attempt. Earlier builds, whose manifests carry no `frame_fallback_policy`,
replaced the whole attempt with it; its failures restarted from zero on the next
frame, the fallback fired on every later frame and valid coverage collapsed.
`smoke/run_frame_fallback_gate.py` reproduces both situations.

**Cost.** POIs that neither pass recovers stay bad and restart from zero on the
next frame, and the threshold never moves, so a fallback once triggered can
fire on every remaining frame, each time adding a full-frame FFTCC + IC-GN to
the skip attempt. `[FALLBACK]` on stderr, `fallback_used` per frame and
`Frames with fallback FFTCC` at the end of the run show it.

### Output cropping: `--output-roi` and `--output-step`

`--output-roi x1,y1,x2,y2` keeps only POIs inside that **inclusive** pixel
rectangle, and `--output-step N` keeps only POIs whose `x` and `y` are both
multiples of N. That is not every N-th grid point (the binary's usage text says
it is): the grid starts at `subset_radius + 2`, so for example radius 7 with
grid spacing 2 and `--output-step 2` keeps nothing, and the run still exits 0
with empty frames. Both change only *what is written*. Computation, fallback
decisions and the QA statistics still cover the whole field, so a cropped file
is a strict subset of the rows of an uncropped run of the same job, never a
different solution. When either is active the raw manifest carries an
`output_filter` block. Neither applies to `--diagnostics` CSV files, which
always cover the full field.

## Reading raw output

```python
import sys
sys.path.insert(0, "pipeline")          # or pip install -e .
import fasticgn_io

manifest = fasticgn_io.read_manifest("out/manifest.opencorr-raw.json")
print(fasticgn_io.engine_metadata(manifest)["icgn_kernel"])

for frame, records in fasticgn_io.iter_frames("out/manifest.opencorr-raw.json"):
    valid = fasticgn_io.valid_mask(records)
    u = records["u"][valid]             # pixels, +x
    v = records["v"][valid]             # pixels, +y
    x = records["x"][valid]             # pixel column
    y = records["y"][valid]             # pixel row
```

`fasticgn_io.iter_frames` yields, for each frame, its entry in the manifest's
`frames` list (a dict: `output_name`, `file`, `rows`, `bytes`, `tar_path` and
per-frame QA such as `valid_ratio`, `bad_ratio` and `fallback_used`; the full
key list is the contract's `outputs.raw_manifest.frame_fields`) together with
its records, after verifying the recorded row and byte count.
`fasticgn_io.to_csv_columns(records)` returns a 2-D
float64 array in the legacy CSV column order, and `smoke/raw_to_csv.py` writes
the legacy per-frame CSV files. `fasticgn_io.read_ledger(output_dir)` returns
the hidden `.opencorr-state/` provenance records.

CSV runs read the same way: `fasticgn_io.read_csv_frame(path)` returns a
structured array with the raw record's field names (float64 values; a gradient
column missing from a legacy 5-column file is NaN, never 0), so `valid_mask` and
the field access above apply unchanged, and `fasticgn_io.iter_frame_dir(dir)`
yields `(path, records)` for every `frame_*.csv` or `frame_*.bin` of a directory
in frame-number order, without a manifest. `fasticgn_io.run_status(dir)` reports,
without changing anything, whether a multi-frame output directory looks complete
(ledger targets present with the sizes their commit intents recorded, the raw
manifest listing exactly them, no interrupted append or crash temporaries); it
cannot check content digests and says so in its `not_verified` list.

## Raw manifest

`manifest.opencorr-raw.json` (multi-frame) or the `.bin` output path plus
`.manifest.json` (single-frame: `out.csv` becomes `out.bin` and
`out.bin.manifest.json`) describes the run and every frame it wrote.

| Key | Meaning |
| --- | --- |
| `format`, `endianness`, `record_bytes` | `opencorr-gpu-raw-v1`, `little`, `36` |
| `ref_path`, `tar_list_path`, `output_dir` | inputs and destination as given |
| `subset_radius`, `grid_space` | the **radius** actually used, and the POI spacing |
| `conv_threshold`, `max_iteration` | IC-GN convergence policy |
| `durability` | effective `--durability` policy |
| `engine` | what actually ran, see below |
| `output_filter` | present only when `--output-roi`/`--output-step` is active |
| `input_image` | requested depth/alignment/normalisation and imread flags, and what the files actually are, see below |
| `fields`, `csv_field_order` | the binary record layout and the CSV column order |
| `frames` | per-frame file name, rows, bytes, timing and QA |

### `engine`

The `engine` block answers "what produced these numbers", so a result can be
replayed or audited without the original command line:

| Key | Meaning |
| --- | --- |
| `implementation` | `FastICGN` |
| `git_hash`, `build_date` | commit the binary was built from (sampled at build time) and the time its build tree was configured |
| `version`, `git_dirty` | semantic version (`--version` prints the same), and whether tracked files differed from `git_hash` when the binary was built: `true`, `false`, or `null` outside a git checkout. A `true` here means the numbers came from uncommitted code. Absent in manifests written before 2026-10-05 |
| `icgn_backend` | `cuda` or `cpu` |
| `icgn_kernel` | `warp`, `legacy`, or `cpu (ICGN2D1)` in CPU builds |
| `oob_policy` | which out-of-bounds rule produced these numbers. `geometric-v1` is the rule described under [Out-of-bounds target subsets](#out-of-bounds-target-subsets). Absent in manifests written before 2026-09-17, which predate the rule |
| `fftcc_backend` | `gpu` (batched cuFFT) or `cpu` (FFTW) |
| `skip_fftcc`, `poi_fallback`, `temporal_extrapolation` | the seeding, repair and temporal policies actually in force |
| `frame_fallback_policy` | how the skip-FFTCC whole-frame fallback combines its from-scratch pass with the skip attempt. `per-poi-merge-v1` is the per-POI merge described under [Whole-frame fallback](#whole-frame-fallback). Absent in manifests of earlier builds, whose fallback replaced the whole skip attempt |
| `preload`, `threads`, `decode_threads`, `dual_gpu`, `gpu_device` | host and device scheduling of the run that wrote the manifest. `preload`, `threads`, `decode_threads` and `gpu_device` never change an output byte and are not part of the resume contract, so a resume may run under other values (and then records those) |
| `durability` | mirrors the top-level value |
| `cuda` | `device_name`, `compute_capability`, `driver_version`, `runtime_version`; `null` in CPU-only builds, and each member is `null` if the query failed |

`engine` is additive metadata. It changes no frame payload byte, and the run
ledger digests whatever manifest it finds on disk, so completion records stay
consistent. Manifests written before 2026-09-17 have no `engine` block;
`fasticgn_io.engine_metadata()` returns `{}` for those.

The diagnostics manifest (`manifest.opencorr-diagnostics.json`, written with
`--diagnostics`) carries the same `engine` block next to its own
`status_semantics` and `columns`.

### `input_image`

| Key | Meaning |
| --- | --- |
| `depth`, `align`, `normalize` | `--input-depth`, `--input-align`, `--input-normalize` as requested (not per-frame `auto` results) |
| `imread_flags` | `IMREAD_GRAYSCALE` for depth 8, `IMREAD_ANYDEPTH\|IMREAD_GRAYSCALE` otherwise |
| `reference_source_depth` | OpenCV depth of the reference file **as stored**: `CV_8U`, `CV_16U`, ... With depth 8 the solver receives 8-bit samples whatever the file holds, so the reference is read once more with `IMREAD_ANYDEPTH` to learn this; `null` if that read failed |
| `reference_source_range` | `[min, max]` of the stored reference values, or `null` |
| `target_source_depth_mismatches` | number of target frames whose stored depth differs from the reference; `0` in a consistent sequence |
| `gray_conversion` | how colour input becomes one channel |

The last three source fields exist in manifests written from 2026-10-05 on.
They are information: they change no payload byte and are not part of the
resume contract. `depth: "8"` together with `reference_source_depth: "CV_16U"`
is the signature of a 16-bit sequence that was reduced to 8 bits (a 16-bit
sample loses its low 8 bits, so right-aligned 12-bit data keeps 4): the run
exits 0 and its `valid_ratio` looks normal, but precision was lost. The binary
warns about it, see [Process output](#process-output).

With depth 8 the targets are compared with the reference through their file
headers (TIFF, PNG, JPEG, PNM and BMP are recognised; a few small reads per
frame, no second decode), otherwise through the depth OpenCV decoded. Every
stored depth up to 8 bits counts as the same grey scale, because OpenCV expands
all of them to full-range 8-bit samples; above 8 bits the depths must be equal.

A frame whose header the probe cannot read is **not compared**, and nothing is
reported for it: another container; a Netpbm file whose header, comments
included, does not end within its first 512 bytes; a TIFF whose `BitsPerSample`
is stored as neither SHORT nor LONG, or whose first directory holds more than
4096 entries; a truncated file. Such frames are read normally. If the
reference's own header cannot be read, no target is compared (the reference's
depth is still known from its second decode, so the warning about a deep
reference does not depend on the header probe).

## Process output

What the binary prints, as opposed to the files it writes.

**stdout** carries exactly one line per computing run, the status line, written
last, also when the run fails and also when the command line was rejected:

```json
{"schema":"fasticgn.run-status/1","status":"ok","exit_code":0,"error":null,"mode":"multi-frame","frames_total":60,"frames_processed":60,"frames_skipped":0,"output_dir":"/data/run-01","output_format":"raw","elapsed_sec":0.694,"tool":"opencorr_cli_cuda_mvp","version":"1.2.0","git_hash":"0123456789ab","git_dirty":false,"build_date":"2026-10-05T07:19:27Z","backend":"cuda","totals":{"skip_fftcc_frames":59,"fallback_frames":0,"poi_fallback_frames":59,"poi_fallback_pois":399259,"poi_fallback_recovered":276,"temporal_seed_frames":0,"temporal_seed_pois":0,"resumed_from_existing_files":0},"warning_count":0,"warnings":[]}
```

| Key | Meaning |
| --- | --- |
| `status`, `exit_code` | `ok` exactly when the exit code is 0, otherwise `error` |
| `error` | text of the first `Error:` line, `null` on success |
| `mode` | `single-frame`, `multi-frame`, or `null` when the command line was rejected before a mode was chosen |
| `frames_total` | target frames requested |
| `frames_processed` | frames computed by this invocation. A resume replays from frame 1, so it counts every frame again |
| `frames_skipped` | frames left alone because the directory verified as a complete run (`frames_processed` is then 0) |
| `output_dir`, `output_path`, `output_format` | absolute output directory; the output file (single-frame mode only); `csv` or `raw`. The line is strict UTF-8: single-frame mode accepts an output path that is not valid UTF-8 and writes the file there, but these two fields then show each undecodable byte as U+FFFD and no longer locate the file (multi-frame mode rejects such paths at startup) |
| `elapsed_sec` | wall time of the whole process |
| `tool`, `version`, `git_hash`, `git_dirty`, `build_date`, `backend` | the binary that ran, as `--version` reports it (`git_dirty` is `true`, `false` or `null`) |
| `totals` | multi-frame runs that computed and succeeded: frames seeded without FFTCC, whole-frame fallbacks, POI-level fallback frames / POIs / recoveries, temporal-seed frames / POIs, and how many existing files a resume started from |
| `warning_count`, `warnings` | number of `Warning:` lines, and the text of the first 20 |

`--version`, `--help` and `--verify-only` print their own answer on stdout and
no status line. `--help` and `-h` are recognised anywhere on the command line
except as the value of an option that takes one (`--output-dir -h` names a
directory) and after `--verify-only`, whose one argument is always a directory.

A process that is killed or crashes prints no status line: **no status line
means the run did not end by itself**, whatever is in the output directory. The
line is written with the reader in mind. A closed pipe does not turn a finished
run into a `SIGPIPE` death. A stdout that is non-blocking and has no room (a
full pipe whose reader is busy) is waited for, up to 5 s in total; this is the
one case in which a run that ended by itself can leave the line out or cut it
short: when the reader makes no room within those 5 s. The exit code is 0
either way, and `--verify-only` on the output directory settles what happened.
A blocking pipe, a file or a terminal is simply written to.

**stderr** carries the run header (image size, options in force, input mode),
one progress line per frame (`[n/N] load-visible ... -> <file>`, plus
`[BASELINE]`, `[FALLBACK]` and `[POI-FALLBACK]` lines), warnings, errors and the
closing totals. `--quiet` drops the per-frame lines and nothing else; without it
stderr is what it always was. With raw output the per-frame QA is in the
manifest either way; with CSV output it exists only in those lines, so keep
them if you need per-frame fallback information.

Warnings never change the exit code. The two input warnings are worth acting
on: `the reference image is stored as CV_16U ... but is read with --input-depth 8`
(the sequence is being reduced to 8 bits; the message names the option to pass)
and `target <name> is stored as ... but the reference as ...` (the frames are
not on one grey scale).

**Exit status** is 0 or 1. For `--verify-only`, 0 means the directory is
complete and every file matches its ledger record
([output integrity](output-integrity.md#verifying-a-directory-without-running-it)).

## Pipeline outputs

`pipeline/dic_pipeline.py` has two modes, selected by whether measurement lines
were given.

**Full-field mode** (no `--interface-y`/`--line-y`) is the general entry point:
it writes `<output-dir>/<event>/frame_NNNNNN.csv` (or `.bin` with
`--output-format raw`, which adds the CLI's `manifest.opencorr-raw.json`) plus
the `.opencorr-state/` ledger, and performs no aggregation. CSV runs write no
manifest.

**COD mode** (`--interface-y` or `--line-y`) is laboratory post-processing: the
CLI computes only the two measurement lines, and the pipeline aggregates them
(into the slip across the interface between the lines, see `cod_um` below)
into `<output-dir>/<event>.csv` (campaign) or `<output-dir>/cod_result.csv`
(single event) with columns `frame,x_mm,cod_um`:

* `frame` is the frame number taken from the target image's file name (the
  number in `frame_NNNNNN.csv`); the reference frame has no rows.
* `x_mm` is `x * cal` for POI column `x`, so its origin is image column 0, not
  the start of `--roi-cols`.
* `cod_um` is the slip across the interface: the relative displacement of its
  two sides, `1000 * cal` times the difference of the horizontal displacement
  `u` between line L0 and line L1, L0 side minus L1 side (L0 is the first
  `--line-y` value, or `--interface-y` minus `gauge // 2`). It is not an opening
  displacement; the column name, `--cod-method` and the abbreviation COD are
  kept as historical names. The sign follows the formula: positive where
  `u_L0 > u_L1`, so it flips when the lines are given in the other order. A
  point is written only where both lines have a valid row.

### COD aggregation modes

`--cod-method naive` (the default since 2026-10-05, and before 2026-07-09)
differences the two lines directly: `u_L0 - u_L1`. `--cod-method fillboundary`
(optional; the default from 2026-07-09 to 2026-10-05) extrapolates `u` on each
line along that line's `uy = ∂u/∂y` to the midpoint `y0 = (L0 + L1) / 2` before
differencing: `[u_L0 + uy_L0 (y0 - L0)] - [u_L1 + uy_L1 (y0 - L1)]`. With
`--interface-y`, `y0` is the interface row. Single-point fill-boundary values
are much noisier than naive ones, because `uy` noise is multiplied by the
extrapolation distance, and a line inside the interface transition band biases
them more; see
[quality checks](quality-checks.md#6-fill-boundary-cod-and-single-point-extremes).
A COD CSV written between 2026-07-09 and 2026-10-05 without `--cod-method` is
fill-boundary; `events[].cod_mode` in its summary says which.

A frame CSV without the `uy` column cannot support fill-boundary. With
`--cod-method fillboundary` the pipeline **fails** with `cod_uy_column_missing`
instead of substituting zero, which would have silently turned fill-boundary
into naive. The default naive needs no `uy` column and aggregates such frames;
the run summary records `cod_mode: "naive"`.

The COD CSV is written to a temporary file in the same directory and renamed
over the target, so the target is always either the previous file or the
complete new one. Next to each frame directory the pipeline keeps the binary's
complete stderr as `<output-dir>/<frame directory name>.binary.log` (outside the
frame directory, because the binary rejects files it did not write when it
resumes a directory).

### Run summary JSON

`--json-summary PATH` writes `schema_version: 1` with `params`, `binary`,
`events`, `global`, `warnings`, `warning_details`, `verdict` and (added
2026-10-05) `event_selection`. NaN and
Infinity are never emitted. New fields are only ever added, and `warnings`
keeps its earlier meaning (warn and fail entries in the earlier wording). Fields
worth knowing:

| Field | Meaning |
| --- | --- |
| `params.field_mode` | `cod` or `full_field` |
| `params.gpu_bin_source` | how the CLI path was resolved: `argument`, `environment:FASTICGN_GPU_BIN`, `config:<file>`, `repository_build`, `legacy_lab_path`, `unresolved`; `params.gpu_bin` is already absolute |
| `verdict` | computed from `warning_details`: `fail` if any event failed or an entry has severity `fail`; otherwise `warn` if any entry has severity `warn`; otherwise `ok`. Severity `info` never changes it |
| `warnings` (top level and per event) | the warn and fail entries only, with the meaning and wording of 638ad18: codes that existed then keep their exact text (`input_8bit_max_255: possible saturation_or_high_bit_truncation`, the bare `gpu_cli_failed`), newer codes read `code: detail`. Information never appears here, so a script that accepts a fixed list or wants an empty one behaves as before |
| `warning_details` (top level and per event) | every entry, information included, as `{code, severity, detail}` with severity `info`, `warn` or `fail` and the full detail (saturated percentage, exit code and `Error:` line, ...); the top-level list holds binary, provenance, argument and `--strict` pre-check entries (the latter with `stage: strict_precheck` and `event`). Every code, its severity and what to do are in the contract under `outputs.json_summary.warning_vocabulary` |
| `events[].status` | `ok` or `fail` for that event, from the same severities; read it for every event, since without `--strict` a campaign exits 0 even when events failed |
| `events[].cod_csv` | the COD CSV this run wrote, only when the event is `ok`; `null` for a failed event (`stale_cod_csv` then names any older file left at that path) |
| `events[].field_dir`, `field_dir_exists` | the event's frame directory, as before whether or not it still exists, and whether it does; COD mode deletes it after the aggregation unless `--keep-frames` (an empty COD included; `--strict` keeps those frames for diagnosis) |
| `events[].cod_mode` | the aggregation that actually ran (`naive` by default since 2026-10-05, or `fillboundary`; always equal to `events[].cod_method`); `null` in full-field mode |
| `events[].valid_ratio` | pooled `finite(zncc) && zncc >= 0` over all parsed frames |
| `events[].geometry` | subset size and radius, step, the lines `[L0, L1]`, their separation in px and mm, the fill-boundary target `(L0 + L1) / 2` and extrapolation distances (`null` under the default naive), the ROI, and (added 2026-10-05) the event's `interface_y`, its `interface_y_source` (`interface_y` = the `--interface-y` default, `event_interface_y` = an `--event-interface-y` option) and the matching options `interface_y_overrides` (`[]` for the default; all three `null` with `--line-y` or in full-field mode) |
| `event_selection` | (added 2026-10-05) campaign only, `null` for a single event: `--events` as given and its `patterns` (`null`: every event), `events_discovered` / `events_selected` / `events_excluded` and the `excluded_events` names, `interface_y_default`, and one entry per `--event-interface-y` option (`argument`, `pattern`, `interface_y`, the selected `events` it applies to) |
| `events[].cod_stats` | COD rows written, frames with and without points, x points, rows found on each line, `all_zero`, the largest and smallest COD with frame and column, the 95th percentile of \|COD\|, (fill-boundary only; `null` under the default naive, whose extremes are already naive) the naive value at those two points, and `unreadable_frames` |
| `events[].input_saturated_fraction` | fraction of the whole reference frame's pixels at 255 at effective depth 8; it grades `input_8bit_max_255` in full-field mode only |
| `events[].input_saturation` | (added 2026-10-04) what `input_8bit_max_255` is graded on, at effective depth 8 (`null` otherwise): `basis` (`line_bands` in COD mode, `frame` in full-field mode), the graded `value` (COD: the largest band fraction), `threshold` (`--saturation-warn-fraction`, default 0.05; `info` up to it, `warn` above), `frame_fraction`, the `severity` the warning got (`null` when no pixel is 255; always `warn` for a container deeper than 8 bits), `container_bit_depth`, and in COD mode `lines`: per measurement line its `y`, the band (`band_rows` = `[L - r, L + r]`, `band_cols` = `[x0 - r, x1 + r]` for `--roi-cols x0,x1`, else the binary's POI range on the line `+/- r`, inclusive and clipped to the image: the pixels of the subsets the COD uses), its `band_fraction` at 255, the number of `subsets` and the fractions of them with any pixel at 255 (`subsets_with_saturation`) and with more than 10 % at 255 (`subsets_over_10pct`). See [quality checks](quality-checks.md#5-saturation) |
| `events[].target_frames_available`, `target_frames_selected` | targets found after the reference, and how many ran (`--max-frames`) |
| `events[].binary_exit_code`, `binary_stderr_log`, `binary_totals`, `remediation` | the binary's exit code, its stderr log (written while the binary runs: the command line first, the exit code appended at the end, so a killed pipeline still leaves it), the end-of-run counters it printed (processed frames, whole-frame and POI-level fallback, verified resume), and one actionable sentence when it failed |

Exit codes: 0, 1 and 2 keep their historical meaning; `--strict` adds 3 (the run
finished but an event failed) and refuses with 2 whatever the dry-run would
refuse. An invalid `--events` or `--event-interface-y` is refused with 2 in every
mode, before anything is read or written. The contract's `exit_codes` has the details.

`--dry-run` prints the same validation plus the exact binary argv it would run,
the state of `--output-dir` and of each event's frame directory
(`output_state`, including whether an existing run ledger accepts the planned
target list), each event's `input_saturation` (the object the summary will
carry, with the level `input_8bit_max_255` will get), each event's `geometry` and the
campaign's `event_selection` (as the summary will give them), and runtime and disk
estimates, to stdout as JSON, without starting a GPU or writing anything. `errors` block the run, `warnings` (severity
`warn`) need a look, `notes` (severity `info`, such as `gpu_bin_outside_checkout`
or `max_frames_applied`) are context only.

The full machine-readable contract — every parameter, default, error code and
output field — is `pipeline/interface_contract.json`, also printed by
`python3 pipeline/dic_pipeline.py --describe`.
