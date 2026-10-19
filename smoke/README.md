# FastICGN Smoke and Regression Checks

## Tiered verification: `scripts/verify.sh` (start here)

```bash
scripts/verify.sh --tier static|quick|gpu|lab [--json out.json] [--build-dir DIR] [--python PY]
                  [--ctest-label REGEX]
```

| Tier | Needs | Runs | Measured |
| --- | --- | --- | --- |
| `static` | Python with `pipeline/requirements.txt` (fastmcp for the MCP tests) | `pipeline/sync_agent_docs.py --check`; `check_license_headers.py` (SKIP outside a git checkout); `dic_pipeline.py --describe` is valid JSON and identical to `pipeline/interface_contract.json`; `python -m unittest discover -s pipeline`; the smoke unit tests that need no binary (`test_check_cpu_gpu_parity.py`, `test_version_metadata.py`; `test_output_integrity.py` and `test_cli_interface.py` run under ctest) | 28 s |
| `quick` | + C++ toolchain, OpenCV, Eigen, FFTW | static, then the `cpu-only` preset: configure, build, `ctest` (11 tests), `selfcheck.sh` | 65 s |
| `gpu` | + CUDA toolkit and a device | static, then `cuda-fftw` (or `--preset cuda-tests`): configure, build, `ctest` (11 tests), `selfcheck.sh` | 75 s |
| `lab` | + the lab dataset | `cuda-tests` configure and build, `run_production_anchors.sh` | 40 s |

Measured on the lab host on 2026-10-04 from fresh build trees (`-j 40`; static
and quick with 138 pipeline and MCP unit tests, gpu with 132); the gpu figure
includes the cuda-fftw build, the lab figure (measured at commit 0c6aae7,
before the build moved into the run
directory) the cuda-tests build and six anchors. Those figures predate the
`subpixel_accuracy` test and self-check stage (2026-10-05), which add about
5 s to the ctest step and 4 s to the self-check of the quick and gpu tiers.
One run on 2026-10-05 with that test, before `cli_interface`,
`version_metadata` and `registry_threads` were added, with other jobs on the
host and 161 unit tests, took 34 s (static), 84 s (quick) and 101 s (gpu).
Exit status: 0 every step
passed, 1 a step failed, 77 nothing failed but a step was skipped (no CUDA
device, no lab data, not a git checkout), 130 interrupted (Ctrl-C or TERM: each
step runs in its own process group, which gets TERM and after 10 s KILL; the
step is recorded as `INTERRUPTED`, later steps do not run, and the report is
still written), 2 usage error. Each step prints its command, exit
code and duration, and its output goes to a log in a fresh run directory
(`mktemp -d` under `$TMPDIR`, left in place: about 40-60 MB with a CPU build,
more with CUDA). That directory also gets `verify-report.json` (`--json FILE`
writes a copy): every step's command, exit code, duration and log,
`git describe --always --dirty`, and the sha256, `git_hash`, `version` and
`git_dirty` of the tested binaries (`git_dirty: "1"` means the binary was built
from uncommitted changes). `HOME`, `XDG_CONFIG_HOME` and `TMPDIR` point into the run directory
and every `FASTICGN_*` variable is unset, so a user configuration cannot change
the result; `PYTHONUSERBASE` keeps pointing at the caller's user site, so
packages installed with `pip --user` stay importable. Pass `--python` for an
interpreter (for example a venv) that has the requirements.

Builds go into the run directory, `<run>/build/<preset>`, never into the
checkout's `build/` trees: `dic_pipeline.py` searches those for its binary
when neither `--gpu-bin`, `FASTICGN_GPU_BIN` nor a config file names one, so a
verification build there would silently become the pipeline's binary.
`--build-dir DIR` builds elsewhere. So every run configures and builds from
scratch (no incremental build) and keeps its tree; do not point `--build-dir` at
the checkout's `build/<preset>` to get incremental builds back, because a
pipeline or MCP run without `--gpu-bin` would then use that binary. Either way the tree is configured with
`cmake --preset P -B DIR` and tested with `ctest --test-dir DIR`. The lab tier reads `ANCHOR_REF`, `ANCHOR_TAR_LIST_60`
and `ANCHOR_TAR_LIST_800` and passes `ANCHOR_FFTW_BIN`, `ANCHOR_GPU_DEVICE`,
`ANCHOR_KEEP_RAW` and `ANCHOR_OUT_ROOT` through (default out root: the run
directory). Without the dataset it reports SKIP; without `ANCHOR_FFTW_BIN` (or
`--preset cuda-fftw`) the `--fftcc cpu` anchor is a SKIP step.
`run_production_anchors.sh` deletes each anchor's `frame_*.bin` after hashing
unless `ANCHOR_KEEP_RAW=1`, which keeps about 15 GB for the six anchors.

`--ctest-label REGEX` (quick and gpu tiers) narrows the ctest step to the tests
whose label matches, as `ctest -L` does, for a faster loop while changing one
area; `verify-report.json` records it as `ctest_label`, and a label that matches
no test fails the step. Without the option every registered test runs, and only
that is the tier's gate. The labels are set in one block of `CMakeLists.txt`:

| Label | Meaning | Tests |
| --- | --- | --- |
| `cpu` / `gpu` | runs without a CUDA device / launches a CUDA binary and needs one; every test has exactly one of the two | CPU-only build: all `cpu`. CUDA build: all `gpu` except `parity_unit` and `version_metadata` |
| `numeric` | compares computed displacements with a known truth | `library_boundaries`, `oob_rejection`, `frame_fallback`, `subpixel_accuracy`, `zero_texture`, `cache_lifecycle` (CUDA), `cpu_sequence_t1`, `cpu_sequence_t4` (CPU-only) |
| `integrity` | output publication, resume and repair; the CLI's own interface | `output_integrity`, `cli_interface` |
| `unit` | runs no CLI: Python unit tests, the library's host-thread probe | `parity_unit`, `version_metadata`, `registry_threads` (CUDA) |

No test needs the lab dataset, so there is no `lab` label: the production
anchors stay outside ctest, in the `lab` tier. `ctest --preset cuda-fftw -LE gpu`
is what the CUDA compile job of the CI runs on a host without a GPU. Every test
also has a `TIMEOUT` of roughly 30 times its duration on the lab host (900 s for
`output_integrity`, 120 s for `parity_unit`, 300 s for the rest), so a hung CLI
fails in minutes rather than after ctest's 1500 s default.

## One-command self-check

```bash
bash smoke/selfcheck.sh                 # add --build-dir DIR or set FASTICGN_GPU_BIN
```

`selfcheck.sh` is the single entry point for "does this build actually work"
(`scripts/verify.sh` runs it in the quick and gpu tiers).
It generates its own synthetic speckle inputs, runs a three-frame raw sequence,
reads it back through `pipeline/fasticgn_io.py` via `selfcheck_read_raw.py`
(which also asserts the manifest `engine` block and the ledger), runs
`run_cache_regression.py`, compares the CPU reference CLI against the sequence
output with `check_cpu_gpu_parity.py`, asserts the out-of-bounds rejection
rule with `run_oob_rejection_gate.py`, and runs `run_subpixel_accuracy_gate.py`
(sub-pixel accuracy on noisy speckle and the CLI's numeric defaults, below).
Its three-frame sequence is requested with `--subset-size 15`; a separate stage
checks that the CLI log and the raw manifest say radius 7. It prints
PASS/FAIL/SKIP per stage plus an evidence directory, and exits non-zero on any
FAIL. A SKIP does not fail the run, and a skipped stage was not checked: the
report ends with the number of skipped stages. The `--subset-size` stage and the
sub-pixel stage use only the standard library and are never skipped.

It touches no dataset and no deployment path: binaries come from `--build-dir`,
`$FASTICGN_GPU_BIN` / `$FASTICGN_CPU_BIN`, or the repository `build/` trees. On
a CPU-only build (no `libcudart`) or a host without a visible CUDA device, the
GPU-specific stages report `SKIP` with the reason instead of failing, and the
parity stage is labelled CPU/CPU.

## The individual checks

These checks are intentionally small. They generate synthetic PGM inputs under
`$TMPDIR`, use a build tree from this repository, and do not read or modify
production datasets or production binaries. No script carries a site-specific
default path.

Build the CLIs first, from the repository root. The `cuda-fftw` preset also
builds `opencorr_cli_cpu`, the CPU reference binary the parity checks need:

```bash
cmake --preset cuda-fftw
cmake --build --preset cuda-fftw -j "$(nproc)"
bash smoke/run_v9_cpu_gpu_parity.sh        # finds build/cuda-fftw by itself
```

How the scripts find their binaries (`smoke/find_build_binary.sh`). The
build-tree order is the one `smoke/selfcheck.sh` and `pipeline/dic_pipeline.py`
also search, but the overrides differ: these scripts read `GPU_BIN`,
`OPENCORR_CPU_BIN` and `CPU_BIN`; `selfcheck.sh` reads `FASTICGN_GPU_BIN` and
`FASTICGN_CPU_BIN`; the pipeline reads `FASTICGN_GPU_BIN` and then the optional
config file before the build trees.

1. `GPU_BIN` / `BUILD_DIR` / `PROBE_BIN` from the environment, when set;
2. the first repository build tree that actually contains the file, searched as
   `build/cuda-release`, `build/cuda-tests`, `build/cuda-fftw`, `build/cpu-only`,
   `build`;
3. otherwise the script exits 1 naming the variable and the build command.

- CPU reference binary: `OPENCORR_CPU_BIN` when set, else the legacy `CPU_BIN`
  override, else `opencorr_cli_cpu` next to `GPU_BIN`, else the same repository
  build-tree search. The scripts fail with an explanation when none of them
  exists. `opencorr_cli_cpu` is built from the same source as upstream
  OpenCorr's `opencorr_cli` (only a licence header differs), so an external
  `opencorr/build/opencorr_cli` stays a valid override.
- `WORK_DIR=${TMPDIR:-/tmp}/fasticgn-smoke-*` (each script has its own tag) and
  is wiped at start, so point it somewhere disposable. On a shared host set
  `TMPDIR` to a directory of your own, or two users running the same script
  will wipe each other's work directory.
- The scripts that need a GPU are `run_c2`, `run_c4`, `run_c6`,
  `run_high_bit_input*`, `run_v9_cpu_gpu_parity` and `run_c5_oob_boundary`;
  `run_c1_reduce_probe.sh` compiles its own probe with `NVCC`.

Checks:

- `run_v9_cpu_gpu_parity.sh`: V9 10-frame CPU vs GPU parity smoke. It joins
  POIs by `(x,y)` and enforces per-frame validity, median, p99, and maximum
  displacement-error bounds (see the cache/parity section below).
- `run_oob_rejection_gate.py`: **assertion** gate for the
  partially-out-of-bounds rejection rule. It builds its own synthetic pair whose
  target is the reference shifted by a known whole-pixel amount, so the band of
  POIs the rule must reject is known in closed form, and then asserts that every
  POI in that band carries `zncc = -3`, that none of them is still valid, and
  that **no** POI outside the band is rejected. The rule is backend-independent,
  so it runs against whichever `opencorr_cli_cuda_mvp` it is given; pass
  `--cpu-binary` as well and it also asserts the two backends reject the
  identical POI set. It is `ctest` case `oob_rejection` in every
  tests-enabled preset and a `selfcheck.sh` stage.
- `run_zero_texture_gate.py`: **assertion** gate for zero-texture reference
  subsets (2026-10-19). It paints three constant blocks -- black, mid-grey and
  saturated -- onto synthetic speckle, so the POIs whose subset lies entirely
  inside a block are known in closed form, and asserts that every one of them
  is rejected before iterating: `zncc = -3` on the CUDA backend (default
  kernel, `--kernel legacy`, and again under `--input-normalize unit`, since
  the predicate is exact equality of the samples), `zncc = -5` on the CPU
  backend, with `u = v = 0` (the incoming guess); and that every POI whose
  subset lies entirely outside the blocks is an exact self-match for the pair
  `ref -> ref` (`zncc = 1`, `|u|, |v| <= 1e-4`). With `--reference-binary`
  it also asserts, record by record, that the POIs not entirely inside a
  block -- the partly-covered ones included -- are byte-identical between the
  two binaries. It is `ctest` case `zero_texture` in every tests-enabled preset.
- `run_frame_fallback_gate.py`: **assertion** gate for the skip-FFTCC
  whole-frame fallback, which merges its from-scratch pass with the skip
  attempt per POI (`frame_fallback_policy: per-poi-merge-v1`, see
  [`docs/output-format.md`](../docs/output-format.md#whole-frame-fallback)).
  Standard-library Python renders two small speckle sequences with exact
  sub-pixel truth and noise: `cascade` (192x144, radius 7, +0.5 px per frame
  from 3.25 to 9.75 px, a 40x40 occluder on frame 6) and `jump` (320x240,
  radius 15, a 10 px jump between the first two frames). Each case runs the CLI
  with 4 threads, with 1 thread and with `--no-skip-fftcc`, and asserts that a
  fallback happened; that no fallback frame ends with more bad POIs than its
  skip attempt; that every POI the from-scratch pass solves (`zncc >= 0.8`) is
  good in the output, none left at `-3`; that the output does not depend on the
  thread count; that valid coverage stays above the case floor; and that valid
  POIs match the truth, with no `zncc >= 0.8` point off by more than 1 px. On
  the pre-merge CLI (`1ca731b`), whose fallback overwrote the skip attempt,
  `cascade` fails: the bad ratio rose on 7 of 8 fallback frames and last-frame
  coverage fell to 0.049, with high-confidence errors of about 11 px. `jump`
  passes there (0.9997 on the jump frame) and guards that the merge is no worse
  than taking the from-scratch pass wholesale; a merge that never replaced a
  `-3` reached 0.949. It is `ctest` case `frame_fallback` in every
  tests-enabled preset: about 3 s with CUDA, 10 s CPU-only.
- `run_subpixel_accuracy_gate.py`: **assertion** gate for sub-pixel accuracy
  and for the CLI's numeric defaults, with no lab data. The other synthetic
  checks shift a noise-free image by whole pixels, so ICGN stops after one
  iteration; this one uses `generate_subpixel_speckle.py` (below): noisy 8-bit
  speckle under an affine displacement field with exact truth. Three cases:
  `s15` (192x144, radius 7), `s15_lowsnr` (the same at contrast 16 and noise 4
  grey levels, where about 20 % of the POIs use all 10 iterations) and `s63`
  (256x224, radius 31, which puts the legacy kernel on 32-thread blocks). Each
  runs the default kernel and, on a CUDA build, `--kernel legacy`. Per frame it
  asserts the POI grid that the requested radius implies, the valid ratio, the
  RMS and mean error of `u` and `v` against the truth, the 99th-percentile
  error, the RMS error of the four gradients, the median `zncc` and the
  manifest's `iteration_mean`. On `s15` it also runs the CLI with
  `--subset-size 15` and no `--conv-threshold` / `--max-iter`, in multi-frame
  and single-frame mode, and asserts that the manifest says `conv_threshold`
  0.001, `max_iteration` 10, `subset_radius` 7 and that the bytes equal those of
  a run with radius 7 and both values spelled out (and that `--conv-threshold
  0.05` gives different bytes, so the data can tell the two apart). Limits allow
  twice the worst value measured at `3e070b1` (CUDA warp, CUDA legacy and
  CPU-only agree to within 0.00005 px); the table and the rule are in the script's
  docstring. They are accuracy limits against the truth, not byte hashes, so the
  gate is meant to hold on other GPUs and CUDA versions; the SHA-256 values in
  its `summary.json` are provenance only. Planted back into a scratch copy
  (2026-10-05), each of these turns it red while every older ctest case stays
  green: the legacy kernel leaving its loop on a per-thread convergence value
  (`s15` legacy valid ratio 0.000), the default `--conv-threshold` at 0.05,
  `--subset-size N` used as the radius (single-frame and multi-frame), and the
  legacy kernel loading the inverse Hessian in a single pass. That last bug is
  not deterministic: over 110 runs the `s63` legacy valid ratio ranged from 0.04
  to 0.90 per frame, but `vy` never converged (gradient error more than 18
  times the limit in every run), so all 110 runs were red. A change to one of those defaults
  must update the constants at the top of the script in the same commit. It is
  `ctest` case `subpixel_accuracy` in every tests-enabled preset and a
  `selfcheck.sh` stage: about 4 s with CUDA, 4 s CPU-only.
- `generate_subpixel_speckle.py`: the generator behind that gate, usable on its
  own (`--out-dir DIR [--width W --height H] [--motion u0,v0,ux,uy,vx,vy ...]
  [--contrast C --noise N] [--seed S]`). Standard-library Python: Gaussian
  speckles evaluated analytically at the back-projected position of every
  target pixel, so no image is interpolated and the displacement and its
  gradients are exact; then grey-level contrast, Gaussian sensor noise and 8-bit
  quantisation, all seeded. It writes `ref.pgm`, `tar_NNN.pgm`, a multi-frame
  `tar_list.txt` and `truth.json`. Bytes are reproducible on one interpreter but
  not promised across C libraries (`math.exp`), which is why the gate compares
  with the truth instead of hashing.
- `run_c1_reduce_probe.sh`: C1 reduction probe. It demonstrates the old
  96-thread reduction failure and the fixed arbitrary-block reduction.
- `run_c2_fallback.sh`: C2 skip-FFTCC fallback guard. It uses a synthetic
  large-shift second frame and requires one frame-level fallback.
- `run_c4_shared_memory.sh`: C4 dynamic shared-memory launch check. It runs
  subset radii 20, 25, and 31 and prints the runtime launch configuration.
  On GPUs where radius 31 exceeds the default shared-memory limit but fits the
  opt-in limit, the log should include `dynamic shared-memory opt-in applied`.
  The sizes are those of the legacy iteration kernel's shared-memory layout,
  which is checked and configured whichever kernel runs; the default warp kernel
  itself uses no dynamic shared memory for iteration. The script has no
  assertion: it prints the launch configuration and whether opt-in happened.
- `run_c5_oob_boundary.sh`: C5 near-boundary synthetic shift. It verifies that
  the sample contains target OOB reads and summarizes CPU/GPU differences in
  the predicted OOB band. Both backends now reject a POI whose warped target
  subset leaves the interpolatable range with `zncc = -3`
  (see [`docs/output-format.md`](../docs/output-format.md)), so
  `oob_shared_valid` should be 0 and the printed `oob_*` deltas empty; the
  interior band is what carries the real CPU/GPU comparison. It asserts that
  the synthetic case exercises predicted-OOB samples at all, that
  `oob_shared_valid` is 0, and that both backends rejected every one of them.
  For the single-binary version of the same assertions, which needs no CPU
  reference, see `run_oob_rejection_gate.py` above.
- `run_c6_subset_capacity.sh`: C6 library-style capacity smoke. It links a
  small probe against the scratch build and runs a small-subset then
  large-subset sequence, using `compute-sanitizer` when available.
- `run_high_bit_input.sh`: high bit-depth input smoke. It checks exact
  16-bit and 12-bit left/right readback values, legacy 8-bit compatibility,
  CLI preload, `--no-preload` synchronous loading, and raw manifest input
  metadata.
- `run_high_bit_input_extra.sh`: the 2026-07-10 follow-up. Spaced-path
  tar-lists, `--input-depth auto` inference anchors including the documented
  misjudgment cases, and the `--input-normalize` output ranges.
- `check_installed_headers.sh`: compiles every header `cmake --install` put
  under `<prefix>/include/opencorr-gpu/` as the single include of its own
  translation unit, and checks each one is declared in
  [`docs/library-contracts.md`](../docs/library-contracts.md). Run it after
  `cmake --install <build-dir> --prefix DIR`:
  `bash smoke/check_installed_headers.sh --prefix DIR`.
- `find_build_binary.sh`: sourced helper, not a check. It is the shared
  environment-then-repository-build-tree lookup described above.
- `raw_to_csv.py`: converts a raw output manifest and its `.bin` frame files
  back to legacy per-frame CSV files with the existing
  `x,y,u,v,zncc,uy,vy,ux,vx` column order.
- `verify_raw_vs_csv.py`: compares raw output against CSV output from the same
  job and reports numeric deltas for regression checks.
- `selfcheck_read_raw.py`: reads a raw output directory back through the
  published `pipeline/fasticgn_io.py` reader and checks the manifest `engine`
  block, the per-frame row/byte counts and the `.opencorr-state` ledger.

For release provenance, configure a fresh build tree from the intended release
commit or tag. `opencorr_gpu_mvp_version.h` is generated when the tree is
configured and refreshed by every build, so `--version` reports the `git_hash`
and `git_dirty` of the tree that was compiled (`build_date` stays the configure
time); a release binary must say `git_dirty=0`.
# Cache correctness and per-frame parity (2026-09-16)

Build the real CUDA library/CLI cache regressions in an isolated build tree:

```bash
cmake -S . -B build-cache -DOPENCORR_GPU_MVP_BUILD_TESTS=ON
cmake --build build-cache -j 4
python3 smoke/run_cache_regression.py \
  --gpu-bin build-cache/opencorr_cli_cuda_mvp \
  --probe build-cache/cache_lifecycle_probe --gpu-device 0
python3 -m unittest discover -s smoke -p test_check_cpu_gpu_parity.py
```

The CMake option defaults OFF. When enabled in a CUDA build,
`ctest --test-dir build-cache --output-on-failure` runs eleven tests,
`library_boundaries`, `output_integrity`, `cli_interface`, `parity_unit`,
`version_metadata`, `oob_rejection`, `frame_fallback`, `cache_lifecycle`,
`registry_threads`, `subpixel_accuracy` and `zero_texture`, using device 0 (use
`CUDA_VISIBLE_DEVICES` to select a physical GPU).

CPU-only hosts can enable the same test option without finding a CUDA compiler:

```bash
cmake -S . -B build-cpu -DOPENCORR_GPU_MVP_ENABLE_CUDA=OFF \
  -DOPENCORR_GPU_MVP_BUILD_TESTS=ON
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure
```

This runs eleven tests: `library_boundaries`, `output_integrity`, `cli_interface`,
`parity_unit`, `version_metadata`, `oob_rejection`, `frame_fallback`,
`cpu_sequence_t1`, `cpu_sequence_t4`, `subpixel_accuracy` and `zero_texture`. The
CPU sequences cover both three-frame orders, default fallback, no-skip and no-preload. The CPU
library probe also compares against a freshly prepared native CPU `ICGN2D1` across
thread counts 1 -> 4 -> 2, reference/target changes, explicit/lazy preparation,
and the repair path. `run_cache_regression.py --binary` is a backend-neutral alias
for the existing `--gpu-bin`, with `--threads` selecting the CLI worker count.
The Python runner creates a unique evidence directory and leaves inputs, commands, logs,
per-frame coverage/truth checks, and `summary.json`. An explicit `--work-dir` must not exist.

The sequences are `0 -> 5 -> 0` and `5 -> 0 -> 5` pixels, with no-skip, default fallback,
and no-preload modes. Each frame must recover at least 95% of the 420 interior POIs within
0.01 px in both displacement components. The C++ probe separately checks both CUDA classes
using setters plus lazy `compute()`, with a mutable target, a mutable reference, and reference
object replacement. Reused objects must match fresh objects; indexed FFTCC is covered too.
The cache contract probe also verifies that a reference change really rebuilds the
retained cache (ready-state checks), reordered/repeated slots work, and invalid indices
fail both before and after invalidation. `WITH_FFTW=ON` runs the same contract checks
against CPU FFTW, including invalidation through the base `DIC` setter. An unprepared
unindexed call intentionally stays uncached; indexed calls require `prepareRef(pois)`.
These tests are measurably red on the pre-fix `da3bd11` core: all six CLI regression
sequences report `valid=0 correct=0` on the second frame and the runner exits 1, and the
probe built against that core reports 34 PASS / 14 FAIL, every failure a second-frame case
covering both `fft-*` and `icgn-*`. After the fix the runner exits 0 and the probe reports
65 PASS (default build) or 83 PASS (`OPENCORR_GPU_MVP_WITH_FFTW=ON`) with no failures.
See `docs/validation-cache-validity-2026-09-16.md` for the full evidence index. These tests
do not use the pre-fix multi-frame hashes as expected results.

Since 2026-10-05 the probe also covers `ICGN2D1CudaMvp::computeGathered` with both
iteration kernels (14 more checks, `icgn-gather-*`): a recipient prepared for the full
layout gathers 32 POIs from a donor, then computes the full layout again. The gathered
queue, the later full-layout `compute()` and a third solver that tries to gather from the
overwritten recipient must all equal fresh solvers, and `hasPreparedReference` must be
false between the gather and that `compute()`. Before the fix the recipient kept its
layout cache: all 32 overwritten slots came back wrong in both later cases, with both
kernels (6 FAIL).

`check_cpu_gpu_parity.py` now checks every frame before accepting a sequence:

- At least 100 shared-valid POIs per frame (`--min-frame-shared-valid`).
- At least 95% of CPU-valid common POIs remain GPU-valid (`--min-valid-retention`).
- At most 5% validity disagreement on common coordinates (`--max-validity-disagreement`).
- Per-frame median |du|/|dv| <= 1e-3 px and nearest-rank p99 <= 1e-2 px.
- Maximum |du|/|dv| <= 0.5 px (`--max-threshold`), including isolated outliers below p99.
- Non-finite displacement/ZNCC is invalid; duplicate coordinates are rejected.
- Malformed CSV rows, including missing columns producing `TypeError`, are reported as
  failed frames in the JSON report rather than unhandled tracebacks.
- Frames are paired by file name. `--frames N` alone compares `frame_000000.csv` to
  `frame_{N-1:06d}.csv` (the 0-based names `run_v9_cpu_gpu_parity.sh` writes; default 10).
  `--glob 'frame_*.csv'` instead pairs every matching name found in either directory, so
  pipeline outputs numbered after the image files work as they are; a name on one side only
  fails its frame, and `--frames N` then keeps the first N names.

The p99 bound is deliberately looser than the median and the archived synthetic-test tails;
it is a regression threshold, not a measurement-uncertainty claim. The optional `--json-report`
records thresholds, both directions of validity loss/gain, lost-point bounding boxes, and
worst-error coordinates. The legacy `--min-shared-valid` remains a sequence-wide floor.
The 0.5 px maximum is a synthetic-regression default. Rare ambiguous low-texture POIs
in real data may exceed it; their previous pooled/p99 pass does not imply a pass under
this new maximum. Different protocols must explicitly report any different threshold.
These checks operate on common coordinates (CPU/GPU margins can differ); known-truth tests
also enforce the expected grid count so loss of coordinates cannot masquerade as accuracy.

## Atomic output and resume regression

`BUILD_TESTS=ON` also registers `output_integrity`, for CPU-only and CUDA builds.
The existing CPU `parity_unit`/`cpu_sequence_t1`/`cpu_sequence_t4` and CUDA
`parity_unit`/`cache_lifecycle` tests remain registered (11 CPU / 11 CUDA tests in all,
including `library_boundaries`, `oob_rejection`, `frame_fallback`, `cli_interface`,
`version_metadata`, `subpixel_accuracy`, `zero_texture` and the CUDA-only `registry_threads`).
To retain evidence at an explicit, new location:

```bash
python3 smoke/test_output_integrity.py --binary build/cpu-only/opencorr_cli_cuda_mvp \
  --fault-lib build/cpu-only/libio_faults.so --work-dir /tmp/opencorr-integrity-new-run
```

The 33 end-to-end cases cover CSV/raw replay and append, temporal extrapolation,
cropping/diagnostics, complete-run no-rewrite, Unicode/escaped JSON, malformed or
duplicate-key JSON, zero-byte/untracked/truncated/same-size-corrupt output,
changed inputs/parameters/version/order, missing files, valid empty filtered
payloads, unsafe names/symlinks, non-UTF-8 output directories, and directory
locking. Linux `LD_PRELOAD` hooks inject short writes, fsync/close/publication
errors, process exit immediately before/after publication, and manifest
replacement failure. The crash and short-write cases run under **both**
durability policies. The hooks are a separate test library, never linked into
the CLI. Each command, return code, log, and summary is retained in a fresh test
directory (printed at startup). Without `--fault-lib`, fault-injection cases are
reported as skipped.

Durability and repair cases:

- `test_rename_policy_costs_no_fsync_per_frame` counts `fsync`/`fdatasync` inside
  the run directory with the same `LD_PRELOAD` library
  (`OPENCORR_TEST_IO_COUNT`, `OPENCORR_TEST_IO_COUNT_ROOT`) for 2- and 4-frame
  jobs. `rename` must show the same count for both (zero per frame; observed 5
  for the completion barrier), `fsync` at least four more per added frame
  (observed 15 -> 23). Both policies must produce identical frame bytes. Raw
  counts are written to `fsync-counts.json` in the evidence directory.
- `test_durability_choice_is_not_part_of_the_resume_contract` resumes a `fsync`
  run under `rename` and rejects an invalid policy name.
- `test_repair_regenerates_only_the_damaged_frame` truncates one frame, checks
  the plain rejection names `--repair`, then verifies that `--repair` quarantines
  the damaged file, regenerates exactly that frame, and leaves every other
  frame's inode and mtime untouched.
- `test_repair_adopts_missing_intent_and_crash_temporaries` and
  `test_repair_still_rejects_foreign_files_symlinks_and_changed_options` fix the
  safety boundary: ledger damage is repairable, foreign files, symlinked outputs,
  reordered lists and changed options are not.
- `test_failed_append_can_still_resume_the_original_target_list` kills an append
  before its first new frame is published and resumes with the original shorter
  list.

[Output and resume contract](../docs/output-integrity.md). These tests exercise
process interruption and syscall failures, not physical power-loss recovery.

## CLI interface regression

`BUILD_TESTS=ON` registers `cli_interface` (`smoke/test_cli_interface.py`) for
CPU-only and CUDA builds, and `version_metadata`
(`smoke/test_version_metadata.py`, which needs no binary and also runs in the
static tier). Both use synthetic inputs only:

```bash
python3 smoke/test_cli_interface.py --binary build/cpu-only/opencorr_cli_cuda_mvp \
  --fault-lib build/cpu-only/libio_faults.so
python3 -m unittest discover -s smoke -p test_version_metadata.py
```

The 26 `cli_interface` cases cover:

- `--help` / `-h`: exit 0, usage and every option of `parseCommonOption` with a
  default on stdout, nothing on stderr; usage errors: exit 1, `Error:` and the
  usage text on stderr, the status line on stdout. After any option that takes
  a value (the list in the source is bound to the options that read one), and
  after `--verify-only`, the two words are a value: the run or the audit goes
  ahead with a directory of that name.
- `--version`: the first four lines unchanged, then `version`, `git_dirty`,
  `backend` and `fftw` (ctest passes the values CMake configured as
  `--expect-version`, `--expect-backend`, `--expect-fftw`).
- The status line: exactly one JSON line on stdout after a successful run, a
  verified rerun, a failed run and a single-frame run, with the exit code, the
  frame counts and the first error; it survives a reader that closed the pipe;
  with a non-blocking stdout whose pipe is full it waits for the reader and
  arrives whole, and gives up after 5 s when the reader never comes; 25
  warnings are counted as 25 with the first 20 kept; a non-UTF-8 single-frame
  output path stays valid JSON with U+FFFD.
- `--quiet`: identical frame bytes, no per-frame lines, the header, the closing
  totals, warnings and errors still there; without it the per-frame line keeps
  its format.
- Bit depth: a 16-bit reference read with the default `--input-depth 8` warns
  once (the three advice variants), an 8-bit one does not, a target stored at
  another depth than the reference warns with and without `IMREAD_ANYDEPTH`,
  and the raw manifest records `reference_source_depth`,
  `reference_source_range` and `target_source_depth_mismatches` while the
  contract records none of it. The header probe is checked per container in
  its own sub-test (PGM, PNG, TIFF in both byte orders, TIFF with a LONG
  `BitsPerSample`): a container this OpenCV build cannot decode is reported as
  a skip, and PGM, which needs no codec, must have been checked. A 4-bit PNG
  target with an 8-bit reference raises no warning, and a 16-bit PGM whose
  header comment hides it from the probe is not compared (the documented
  limit).
- The resume contract: the four scheduling options are absent from
  `contract.json`; a complete run verifies and an interrupted one finishes under
  other `--threads` / `--preload` / `--decode-threads`; an invalid
  `--gpu-device` no longer poisons the directory (CUDA builds); a contract that
  still lists the four options resumes; a rejected resume names the field with
  both values (an option, the target list, the binary, the reference).
- `--verify-only`: a complete raw run and a complete CSV + diagnostics run pass
  (also from a read-only directory) and not one inode, size or mtime in the
  directory changes; a missing frame or completion record is `incomplete`; a
  truncated or flipped payload, a missing ledger record, a foreign frame file, a
  changed manifest and a broken contract are `damaged`; changed inputs are
  `inputs_changed`; a held lock is `locked`; an empty, absent or single-frame
  directory is `not_a_run` / `not_found`; a missing, empty or extra argument
  still yields one JSON line with status `error`; a run killed by the fault
  library and an abandoned append are reported as what they are.
- `--verify-only` and appends: an append killed right after its first new frame
  was published (`crash-after-publish`) is `incomplete` with that frame under
  `appended_published`, the short list is then refused and the long list
  continues without `--repair`; the same directory with the appended frame
  altered, without its ledger record, replaced by a symlink, or with a pending
  contract that is unreadable or does not extend the recorded one stays
  `damaged`; the other `pending_append_state` values are reported.
- `--verify-only` and foreign entries: a file, a symlink, a dangling symlink or
  a directory with the extension of the run's outputs, next to the frames or in
  the diagnostics directory, is `damaged`; other names are only listed, and the
  original command then refuses the directory although the verdict is
  `complete`.

`version_metadata` binds the three places the version is written:
`CMakeLists.txt` (project `VERSION` plus `OPENCORR_GPU_MVP_VERSION_SUFFIX`),
`CITATION.cff` (the same with a leading `v`) and `pyproject.toml` (its PEP 440
spelling).

## 2C library boundaries and inherited cleanup

`library_boundary_probe` runs in all supported builds. It checks both rectangular
orientations and invalid radii are rejected, square translation truth, a changed
reference with a mismatched unindexed queue does not rebuild an unused cache,
and a later matching queue does rebuild it. In CUDA builds it also gives
`FFTCC2DCuda` a target smaller and one larger than the reference: both `compute`
overloads must throw `std::invalid_argument` naming the two sizes and leave the
queue untouched, and the solver must work again once a matching target is set
(a smaller target used to be read out of bounds and the queue filled with
garbage). The FFTW variant also checks
inherited geometry changes are rejected. `cpu_sequence_t1` and `cpu_sequence_t4`
now both run `cpu_lifecycle_probe`: repeated adopt reuse, new donor preparation,
and recipient OpenMP pool rebuild are covered in addition to native comparisons.
CPU-only plus `WITH_FFTW=OFF` is rejected at configure time because that CLI has
no available FFTCC backend.

## Solvers on several host threads

`registry_thread_probe DEVICE [THREADS=4] [ROUNDS=30]` (ctest `registry_threads`, CUDA
builds, no data files) runs solvers that belong to different host threads. Every round
constructs, uses and destroys an `FFTCC2DCuda` and two `ICGN2D1CudaMvp` (one borrows the
other's target and gathers from it) next to a pair each thread keeps, so both process-wide
buffer registries are searched, grown and shrunk concurrently; every result must equal the
single-threaded one bit for bit. Before the registries were locked this ended in a
segmentation fault within about a second, and AddressSanitizer reported a
heap-use-after-free through a reference into the registry vector, in
`fftcc2dCudaEnsureRefImage` and, with only the FFTCC side fixed, in the ICGN
`enqueueDeviceWork`. To run it under AddressSanitizer add `-fsanitize=address` to
`CMAKE_CXX_FLAGS` and the linker flags, `-Xcompiler=-fsanitize=address` to
`CMAKE_CUDA_FLAGS`, and set `ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0` (the CUDA
driver maps memory into the shadow gap).

The additional integrity cases exercise normal-resume residue archiving, broken
completion repair while preserving frame inode/mtime, and asynchronous
diagnostics publication failure/recovery under both durability policies.
`test_resume_skips_reserved_names_that_are_not_regular_files` puts a directory
and a symlink under reserved private names next to real residue and checks that
both sweeps (the one before the complete-run fast path and the one after the
completion record is written) warn, skip them, archive only the real residue,
and still exit 0 with unchanged frame bytes.
`test_reserved_names_that_are_not_regular_files_survive_both_modes` repeats that
with and without `--repair`, adding a malformed reserved name, and checks that
all three are only reported, keep their inode and mtime, and never fail the run:
one classifier now serves the resume scan and the post-run sweep.
`test_unreadable_metadata_is_an_io_fault_not_damage` makes `complete.json`
unreadable and checks that both modes exit 1 naming the path and `errno=13`,
quarantine nothing, and do not suggest `--repair`; the same file corrupted in
content is still rejected without `--repair` and repaired with it.
