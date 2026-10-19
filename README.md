# FastICGN

**GPU-accelerated 2D-DIC built on [OpenCorr](https://www.opencorr.org/).**

FastICGN measures full-field displacement from image sequences by digital image
correlation (DIC): FFT cross-correlation seeding plus inverse-compositional
Gauss-Newton (IC-GN) refinement with a first-order shape function. It keeps the
reference image, its gradients and the per-POI Hessians resident on the GPU
across a whole sequence, so a long high-speed recording is processed as one
streaming job rather than as N independent image pairs.

The numerical definition is upstream OpenCorr's, which is vendored in `src/`
(see [`NOTICE.md`](NOTICE.md)); FastICGN adds the CUDA IC-GN kernel, a batched
cuFFT FFTCC, checked and resumable multi-frame output, and a Python/agent
pipeline. Both are MPL-2.0.

中文说明见 [`docs/README.zh.md`](docs/README.zh.md)。

## What and why

- **Sequences, not pairs.** `--multi-frame` walks a target list against a fixed
  reference, reusing the prepared reference state and seeding each frame from
  the previous result. One process, one upload of the reference, one POI layout.
- **Checked, resumable output.** Every payload and manifest is published
  atomically through a temporary file; a hidden ledger lets an interrupted job
  resume, and refuses to adopt output it cannot verify. See
  [output integrity](docs/output-integrity.md).
- **Auditable results.** Raw manifests record the engine that produced them —
  kernel mode, FFT backend, seeding/fallback policy, build identity and device.
  See [output format](docs/output-format.md).
- **Checked against the CPU reference.** The CUDA kernels are not
  bit-identical to CPU OpenCorr: they sum in a different order (the default
  warp kernel reduces with warp shuffles), so agreement is enforced by
  tolerance instead, per frame (median 1e-3 px, p99 1e-2 px, max 0.5 px) by the
  CPU/GPU parity stage of `smoke/selfcheck.sh` and by
  `smoke/run_v9_cpu_gpu_parity.sh`.
- **Not** a GUI, a stereo/3D-DIC tool, or a strain post-processor. It writes
  displacement and shape-function gradient fields; analysis is downstream.

## Requirements

Linux, out of a plain `git clone`. Nothing in the build depends on a lab path.

| Dependency | Minimum | Debian/Ubuntu package | Needed for |
| --- | --- | --- | --- |
| CMake | 3.18 (3.21 to use `CMakePresets.json`) | `cmake` | always |
| C++17 compiler with OpenMP | GCC 11 / Clang 14 (untested floor; GCC 13.3 validated) | `build-essential` | always |
| Eigen | 3.3 | `libeigen3-dev` | always (header-only) |
| OpenCV | 4.x | `libopencv-dev` | always (image I/O) |
| CUDA Toolkit | 11.8 (untested floor; 13.2 validated) | NVIDIA installer | CUDA builds |
| FFTW3 single precision | 3.3 | `libfftw3-dev` | CPU FFTCC, CPU-only builds |
| Python | 3.10 (3.11 for the optional TOML config) | `python3` | tests, pipeline, helper scripts |

The compiler and CUDA floors are read off the code, not from a build: the CLI
formats and parses `float` with `std::to_chars`/`std::from_chars`
(`<charconv>`), which libstdc++ provides from GCC 11, and the default
`CMAKE_CUDA_ARCHITECTURES=89` (Ada) needs CUDA 11.8. Clang 14 also needs a
standard library with floating-point `from_chars`. No build has been made or
tested with GCC 11, Clang 14 or CUDA 11.8; the validated toolchain is GCC 13.3,
CUDA 13.2 and CMake 3.28.

Validated on Ada-class GPUs (`CMAKE_CUDA_ARCHITECTURES=89`). Python
dependencies are `pipeline/requirements.txt` (numpy, Pillow) and
`pipeline/requirements-mcp.txt` (adds FastMCP); `pyproject.toml` installs the
pipeline as `fasticgn-pipeline` if you prefer. The installed entry point does not
run from this checkout and cannot find its `build/`: pass `--gpu-bin` or set
`FASTICGN_GPU_BIN`. Otherwise it takes `~/.config/fasticgn/config.toml`, then a
deprecated hard-coded lab path where that exists, with only a `path_resolution`
warning and an info note `gpu_bin_outside_checkout`; the dry-run still exits 0
unless `--strict` is given, which refuses the lab path.

## Build

```bash
git clone https://github.com/CountofGlamorgan/FastICGN.git opencorr-gpu
cd opencorr-gpu
cmake --preset cuda-release
cmake --build --preset cuda-release -j "$(nproc)"
./build/cuda-release/opencorr_cli_cuda_mvp --version
```

`CMakePresets.json` ships four configurations; each has a matching build preset
and test preset of the same name, and builds into `build/<preset>/`.

| Preset | CUDA | FFTW | Tests | Purpose |
| --- | --- | --- | --- | --- |
| `cuda-release` | ON | OFF | OFF | the production configuration |
| `cuda-tests` | ON | OFF | ON | `ctest --preset cuda-tests`, needs a GPU |
| `cuda-fftw` | ON | ON | ON | dual FFTCC backend for `--fftcc cpu` vs `gpu`, plus `opencorr_cli_cpu` |
| `cpu-only` | OFF | ON | ON | no CUDA toolkit required |

Plain CMake works the same way if you prefer not to use presets:

```bash
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release \
  -DOPENCORR_GPU_MVP_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda -j "$(nproc)"
```

Options:

- `OPENCORR_GPU_MVP_ENABLE_CUDA` (default `ON`) — CUDA ICGN plus cuFFT FFTCC.
  `OFF` builds the CPU backend described above.
- `OPENCORR_GPU_MVP_WITH_FFTW` (default `OFF` with CUDA, `ON` without) — the
  legacy FFTW CPU FFTCC path and the `opencorr_cli_cpu` reference CLI.
- `OPENCORR_GPU_MVP_BUILD_TESTS` (default `OFF`) — registers the ctest suite.
- `CMAKE_BUILD_TYPE` is set to `Release` when the caller leaves it empty.
- `CMAKE_CUDA_ARCHITECTURES` defaults to `89` (Ada, the validated production
  target). Pass `-DCMAKE_CUDA_ARCHITECTURES=native` to match the build host, or
  an explicit list such as `"80;86;89"`. `nvcc` is taken from `PATH`, from
  `CUDACXX`, or from `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

Eigen is located in this order: `-DOPENCORR_GPU_MVP_EIGEN_INCLUDE_DIR=<dir
containing Eigen/Core>`, then `find_package(Eigen3 3.3 CONFIG)` (point it at a
non-default copy with `-DEigen3_DIR=<prefix>/share/eigen3/cmake`), then a
`find_path` for `Eigen/Core` with the `eigen3` suffix. Configuration fails with
an explicit message when none of the three works. Both `<eigen root>` and
`<eigen root>/Eigen` are added to the include path, because the vendored
upstream sources use OpenCorr's non-standard `#include <Eigen>` spelling.

FFTW is **GPL-2.0-or-later** and optional. The production CUDA build does not
link it; CPU-only builds and `-DOPENCORR_GPU_MVP_WITH_FFTW=ON` builds do, and
distributing such a binary puts the combined work under the GPL (NOTICE.md §6).

Tests need `OPENCORR_GPU_MVP_BUILD_TESTS=ON` (already set by the `cuda-tests`,
`cuda-fftw` and `cpu-only` presets):

```bash
cmake --preset cpu-only && cmake --build --preset cpu-only -j "$(nproc)"
ctest --preset cpu-only
python3 -m pip install -r pipeline/requirements.txt   # numpy, Pillow (the unit tests import them)
python3 -m unittest discover -s pipeline
python3 pipeline/sync_agent_docs.py --check
python3 smoke/check_license_headers.py
```

`pipeline/requirements-mcp.txt` (FastMCP) is optional here: without it the MCP
server's test cases are reported as skipped, not as failures.
`smoke/check_license_headers.py` reads `git ls-files`, so it needs a git
checkout; in an unpacked archive it says so and exits 77 (skipped).

`.github/workflows/ci.yml` has three jobs. `cpu-linux` runs that CPU sequence
on `ubuntu-latest`, plus `smoke/selfcheck.sh`, the smoke unit tests and an
install/header check. `cuda-compile` runs automatically as well, inside an
`nvidia/cuda` devel container without a GPU: it configures and builds the
`cuda-release` and `cuda-fftw` presets and runs only the ctest cases that need
no CUDA device (`ctest -LE gpu`), so it proves the CUDA sources compile and link
but launches no kernel. `gpu-cuda` (CUDA build, full ctest and
`smoke/run_production_anchors.sh`) is `workflow_dispatch`-only and targets a
`[self-hosted, gpu]` runner; with no such runner registered it is never
scheduled.

Installing:

```bash
cmake --install build/cuda-release --prefix /opt/opencorr-gpu
/opt/opencorr-gpu/bin/opencorr_cli_cuda_mvp --version
```

This installs `bin/opencorr_cli_cuda_mvp` (plus `bin/opencorr_cli_cpu` when
FFTW is enabled), the `libopencorr_gpu_mvp_core` static library, the embedding
headers under `include/opencorr-gpu/`, and `LICENSE`, `NOTICE.md`, `README.md`,
`CITATION.cff` and the `docs/*.md` files under `share/doc/opencorr_gpu_mvp/`;
`docs/lab/` (one laboratory's deployment manual) stays in the repository and is
not installed. See [library contracts](docs/library-contracts.md) for which
classes are supported public API.

## Quickstart

**1. Prove the build works — synthetic data, one command, no dataset needed:**

```bash
bash smoke/selfcheck.sh
```

It generates its own speckle images, runs a three-frame sequence in raw mode,
reads the result back through the published NumPy reader, runs the known-truth
cache regression, compares against the CPU reference CLI when
`opencorr_cli_cpu` has been built (the `cpu-only` and `cuda-fftw` presets build
it; otherwise that stage is `SKIP`), and asserts the out-of-bounds rejection
rule. It prints PASS/FAIL/SKIP per stage and an evidence directory. On a
CPU-only build the GPU-specific stages are reported as `SKIP`, not as failures.
Point it at a specific build with `--build-dir <dir>`; `FASTICGN_GPU_BIN`, when
set, takes precedence over `--build-dir`. The raw-reader stage needs numpy.

Steps 2 to 4 use a synthetic speckle sequence made by the smoke tests'
generator (standard-library Python, no GPU); step 5 uses the one real sequence
the repository ships, `examples/smooth-30kN-0725B-B2L2/`:

```bash
python3 smoke/generate_synthetic_dic.py --out-dir frames --frames 4
```

It writes 256 x 192 8-bit `.pgm` images into `frames/`: the reference
`ref.pgm`, the targets `frame_000000.pgm` to `frame_000003.pgm`, each `ref.pgm`
shifted by whole pixels (`metadata.json` lists the shifts), and a few more
targets the smoke tests use. Your own `tif`/`png`/`jpg`/`bmp` images take the
same commands.

**2. One image pair:**

```bash
./build/cuda-release/opencorr_cli_cuda_mvp \
  frames/ref.pgm frames/frame_000001.pgm --subset-size 15 1 out.csv
```

The positional arguments are `<ref_image> <tar_image> <subset_radius>
<grid_space> <output_path>`; here `--subset-size 15` replaces the radius and the
`1` is the grid step in pixels between points of interest. `--subset-size N`
takes the odd subset **size**; the positional radius takes the **radius**.
`--subset-size 15` and a positional `7` are the same subset and produce
byte-identical output — pass one, never both. See
[radius versus size](#subset-radius-versus-subset-size).
`out.csv` holds one row per point of interest (`x,y,u,v,zncc,uy,vy,ux,vx`;
for this pair `u = 1`, `v = 0`, the shift that frame was made with).
`opencorr_cli_cuda_mvp --help` prints every option with its default.

**3. A sequence:**

```bash
# tar_list.txt: one "<target image path> <output name>" line per frame.
# The path may contain spaces; the last whitespace-separated token is the name.
i=0; for f in frames/frame_*.pgm; do
  printf '%s frame_%06d.csv\n' "$f" "$i"; i=$((i + 1))
done > tar_list.txt

./build/cuda-release/opencorr_cli_cuda_mvp --multi-frame \
  --ref frames/ref.pgm --tar-list tar_list.txt --output-dir out/ \
  --subset-size 15 1 --threads 8 --conv-threshold 0.001 --output-format raw
```

`--output-format raw` writes `out/frame_000000.bin` and friends plus
`out/manifest.opencorr-raw.json`; the `.csv` names in the target list get the
`.bin` extension automatically.

**4. Or drive it from Python, over a directory of frames:**

```bash
python3 pipeline/dic_pipeline.py --image-dir frames/ --output-dir results/ \
  --gpu-bin "$PWD/build/cuda-release/opencorr_cli_cuda_mvp" \
  --image-pattern '(\d{6})\.pgm$'
```

With no measurement-line argument the pipeline runs **full-field mode**: the
first frame found is the reference (`frame_000000.pgm` here), every other frame
is a target, and the displacement fields land in `results/<event>/`
(`results/frames/frame_000001.csv` to `frame_000003.csv` for this directory).
`--image-pattern` is only there because the synthetic images are `.pgm`, see
the first bullet below. Adding `--interface-y` or `--line-y` switches to the
laboratory COD post-processing described under [Lab workflow](#lab-workflow).

- By default an image is found only if its file name ends in a six-digit frame
  number and a `jpg`, `jpeg`, `tif`, `tiff`, `png` or `bmp` extension
  (`frame_000123.tif`); otherwise pass `--image-pattern REGEX` (and possibly
  `--image-order name`). `.pgm` is not in the default list: without the
  `--image-pattern` above the synthetic directory yields `insufficient_images`
  and exit code 1.
- Images are read as 8-bit unless told otherwise, and a 16-bit file is then
  reduced to 8 bits. The binary prints a warning that names the stored depth and
  the option to pass, but it still runs and exits 0. For 12- or 16-bit camera
  data run the same command with `--dry-run` first and follow its
  `input_recommendation`.
- The pipeline picks the binary from `--gpu-bin`, `FASTICGN_GPU_BIN`,
  `~/.config/fasticgn/config.toml`, then this checkout's `build/` trees, and
  last a deprecated hard-coded lab path. A config file or that path can win
  without failing the run, which is why the command above passes `--gpu-bin`.

**5. Real data — one laboratory rupture, 122 frames at 50 000 fps:**

[`examples/smooth-30kN-0725B-B2L2/`](examples/smooth-30kN-0725B-B2L2/) holds
the reference frame and target frames 443 to 564 of one stick-slip event on a
frictional interface (1024 x 112 px, 8-bit JPEG, 7.7 MB; CC BY 4.0, see its
`MANIFEST.md`). The interface is the horizontal line at image row 56; the slip
across it is measured as the displacement difference between the rows 8 px
above and below (`--interface-y 56 --gauge 16`, lines 48 / 64), over columns
106 to 892, at 0.635527061 mm/px. This is the laboratory COD post-processing
run through the production entry point, `pipeline/run_dic.sh`:

```bash
pipeline/run_dic.sh --image-dir examples/smooth-30kN-0725B-B2L2 --output-dir out/B2L2 \
  --subset 15 --step 1 --interface-y 56 --gauge 16 --cal 0.635527061 --roi-cols 106,892 \
  --gpu-bin "$PWD/build/cuda-release/opencorr_cli_cuda_mvp" \
  --strict --json-summary out/B2L2/summary.json
```

`run_dic.sh` runs `pipeline/dic_pipeline.py` with `python3` (or
`FASTICGN_PYTHON`, when numpy and Pillow live in a virtual environment).
Expected, on a CUDA build: exit 0 in about one second of GPU time, `verdict`
`ok` with an empty `warnings` list, 122 frames x 787 points in
`out/B2L2/cod_result.csv` (`frame,x_mm,cod_um`), `valid_ratio` 0.959 and
`zncc_median` 0.995 in the summary. The spatial median of `cod_um` per frame
stays near 1 µm up to frame 462, leaves that baseline at frames 463 to 464,
climbs at up to 24 µm per frame (1.2 µm per µs, frame 466), passes 110 µm at
frame 471 and 256 µm at frame 542, and levels off near 260 µm (261 µm at
frame 564):

```bash
python3 -c 'import csv,statistics,sys; r={}
for d in csv.DictReader(open(sys.argv[1])): r.setdefault(int(d["frame"]),[]).append(float(d["cod_um"]))
print({f: round(statistics.median(v),1) for f,v in sorted(r.items()) if f in (443,463,471,501,542,564)})' out/B2L2/cod_result.csv
```

prints `{443: 1.4, 463: 7.3, 471: 110.2, 501: 213.8, 542: 256.5, 564: 261.3}`.
The default `naive` method subtracts the two lines' `u`; add
`--cod-method fillboundary` to extrapolate each line to the interface first
(same displacement field, medians a few percent higher, 267 µm at frame 564,
noisier per point; see [Lab workflow](#lab-workflow)). One full-field frame
with the binary alone, subset radius 7 on a 2 px grid:

```bash
./build/cuda-release/opencorr_cli_cuda_mvp \
  examples/smooth-30kN-0725B-B2L2/B2L2_C001H001S0001000001.jpg \
  examples/smooth-30kN-0725B-B2L2/B2L2_C001H001S0001000471.jpg 7 2 frame471.csv
```

writes 23 641 rows (x 9 to 1013, y 9 to 101) of which 95.9 % are valid
(`zncc >= 0`, median 0.997); the rejected points lie in the blank columns
outside the specimen (x below about 40 and above about 970).

### Subset radius versus subset size

This has caused a real false failure, so it is worth one paragraph. The binary
positional argument is the subset **radius** in pixels. `pipeline --subset` and
the binary `--subset-size` take the odd full **size**. The conversion is
`radius = (size - 1) / 2`, so `--subset-size 15` = radius 7, and passing 15
where a radius is expected silently computes a 31-pixel subset. The binary
rejects an even `--subset-size`, and rejects supplying both spellings at once.

## Machine interface (scripts & agents)

For automated use, the interfaces below are machine-readable.

- `python3 pipeline/dic_pipeline.py --describe` prints the contract,
  [`pipeline/interface_contract.json`](pipeline/interface_contract.json): parameters,
  units, defaults, output schemas, error/warning codes, exit codes, known limitations.
- `--dry-run` validates a pipeline command without writing or starting the GPU:
  one JSON plan on stdout (exact binary argv, measurement lines, bit-depth probe,
  the state of `--output-dir` and its frame directories, runtime range);
  exit 0 = runnable, 2 = see `errors[]`; `warnings[]` need a look, `notes[]` are
  context only.
- `--json-summary PATH` writes strict JSON: `verdict` (`ok`/`warn`/`fail`, decided
  by the severities `info`/`warn`/`fail` in `warning_details`) and, per event,
  `status`, `valid_ratio`, `zncc_median`, `warnings`, `warning_details`, the
  geometry used and COD statistics. `warnings` keeps its earlier meaning (warn and
  fail entries only, in the earlier wording); `warning_details` has every entry,
  information included, with its severity and full detail. Top-level `warnings`
  can be empty under `warn`.
- `--strict` applies every dry-run check before running (exit 2; it also refuses a
  16-bit file that would be read as 8-bit, and checks a campaign as a whole) and
  exits 3 when any event failed; without it a campaign exits 0 even with failed
  events. `--max-frames N` runs only the first N targets. In a campaign,
  `--events LIST` (comma-separated event names or globs) runs only some events,
  and `--event-interface-y PATTERN=Y` (repeatable) gives matching events their
  own interface row while the others keep `--interface-y`; both are refused with
  exit 2 before anything is read when a name or pattern matches nothing or two
  rows conflict, and the summary records the selection (`event_selection`) and
  each event's row (`events[].geometry`).
- `pipeline/fasticgn_io.py` reads raw and CSV frames into the same field names and
  reports whether an output directory looks complete (`run_status`).
- The binary on its own: `--help` prints every option with its default (exit 0);
  `--version` prints `git_hash`, `build_date`, `version`, `git_dirty`, `backend`
  and `fftw`. Every computing run, successful or not, ends with one JSON status
  line on stdout (`status`, `exit_code`, `error`, frame counts, `output_dir`,
  `elapsed_sec`, build identity, `warnings`); progress, warnings and errors go to
  stderr, and `--quiet` drops the per-frame lines
  ([process output](docs/output-format.md#process-output)).
  `--verify-only <output_dir>` audits a multi-frame output directory without
  computing or writing anything and prints one JSON verdict, exit 0 only when the
  directory is complete
  ([output integrity](docs/output-integrity.md#verifying-a-directory-without-running-it)).
- Raw manifests carry an `engine` block (build hash, kernel, FFT backend,
  out-of-bounds policy `oob_policy`, whole-frame fallback policy
  `frame_fallback_policy`, device).
- `pipeline/mcp_server.py` is a FastMCP stdio server over the pipeline, COD mode
  only, with a configured read allowlist and job root. Its tools publish typed
  parameter schemas (including `max_frames`, and for campaigns `events` and
  `event_interface_y`, the pipeline's `--events` / `--event-interface-y`) and the server sends usage
  instructions; plans and runs always use `--strict`, so a failed event returns
  its summary with `verdict: fail` and `ok: false`. Rejections come back as
  `ok: false` with `code` and `hint`, and a run keeps its GPU lock until the job
  ends, even if the client session closes. Plans and campaign results with more
  than 10 events list one compact line per event, and every plan or run reply,
  rejections included, stays within 48 KiB, saying what it left out (the full
  summary stays in `summary.json`); `dic_describe` without arguments returns what
  matters before a run (a section index, the run params and the guardrails, about
  15 KB, held under 20 480 bytes by a unit test) and lists the parts it left out, each read by name
  (`section="mcp.results"`, `"mcp.errors"`, ...; `section="all"` returns all of it).
- Whether a finished result can be trusted: [docs/quality-checks.md](docs/quality-checks.md).

## Output format

Per-frame CSV or raw 36-byte records, a hidden integrity ledger for
multi-frame runs, and, in raw mode, a JSON manifest describing the record
layout and the engine that produced it.
Coordinate directions, units, the valid-state rule (`zncc >= 0`), the gradient
columns, `--output-roi`/`--output-step` semantics and the NumPy reader are all
documented in **[docs/output-format.md](docs/output-format.md)**.

The reader is `pipeline/fasticgn_io.py`:

```python
import sys; sys.path.insert(0, "pipeline")    # from the repository root, or pip install .
import fasticgn_io
for frame, records in fasticgn_io.iter_frames("out/manifest.opencorr-raw.json"):
    valid = fasticgn_io.valid_mask(records)          # finite zncc and zncc >= 0
    u, v = records["u"][valid], records["v"][valid]  # pixels, +x right / +y down
```

`frame` is that frame's entry in the manifest's `frames` list, a dict with
`output_name`, `file`, `rows`, `tar_path` and the per-frame QA such as
`valid_ratio`, `bad_ratio` and `fallback_used`.

## Validation

| Check | Command | What it proves |
| --- | --- | --- |
| Tiered gate | `scripts/verify.sh --tier static` (then `quick`, `gpu`, `lab`) | the checks below grouped by what the machine has, one exit code and a JSON report; see [`smoke/README.md`](smoke/README.md) |
| Self-check | `bash smoke/selfcheck.sh` | the build runs end to end on synthetic data; GPU and CPU agree when `opencorr_cli_cpu` is built |
| ctest | `ctest --preset cpu-only` / `--preset cuda-tests` | library boundaries, output integrity, the CLI's own interface (`--help`, `--version`, status line, `--quiet`, bit-depth warnings, resume contract, `--verify-only`), version metadata, parity-checker unit tests, out-of-bounds rejection, whole-frame fallback merge, sub-pixel accuracy on noisy speckle with the CLI's numeric defaults, zero-texture reference subsets rejected before iterating, plus CPU sequences at 1 and 4 threads (`cpu-only`, 11 tests) or cache lifecycle and solvers on several host threads (`cuda-tests`, 11 tests); labels for `ctest -L` are in [`smoke/README.md`](smoke/README.md) |
| Interfaces | `HOME="$(mktemp -d)" python3 -m unittest discover -s pipeline` | pipeline and MCP contracts against a fake binary; they also pass with your real `HOME` (the pipeline tests ignore `~/.config/fasticgn/config.toml`), the throwaway one just keeps the run clean |
| Generated docs | `python3 pipeline/sync_agent_docs.py --check` | the pitfall list in `docs/lab/simulacrum.md` matches `interface_contract.json` |
| Licences | `python3 smoke/check_license_headers.py` | headers present, vendored files unmodified |
| Installed headers | `bash smoke/check_installed_headers.sh --prefix DIR` | every header `cmake --install` ships compiles on its own |
| Production anchors | `smoke/run_production_anchors.sh` | SHA-256 of the frame tree against recorded values (needs the lab dataset) |

Additional targeted smokes: `run_v9_cpu_gpu_parity.sh` (CPU/GPU parity),
`run_c1_reduce_probe.sh`, `run_c2_fallback.sh`, `run_c4_shared_memory.sh`,
`run_c5_oob_boundary.sh`, `run_c6_subset_capacity.sh`,
`run_high_bit_input.sh`, `run_high_bit_input_extra.sh`, and
`run_cache_regression.py`. They take their binaries from `GPU_BIN` /
`BUILD_DIR` when set and otherwise discover the repository's own `build/`
trees, so building any preset first is enough. See
[`smoke/README.md`](smoke/README.md); the dated release records are
[`docs/release-v1.0.0.md`](docs/release-v1.0.0.md) and
[`docs/release-v1.1.0.md`](docs/release-v1.1.0.md).

## Lab workflow

These parts are specific to the laboratory rupture experiments FastICGN was
written for. They are not needed to use the DIC engine, and they are kept out
of the general entry point.

- **COD post-processing.** `pipeline/dic_pipeline.py --interface-y Y --cal
  MM_PER_PX` (or `--line-y L0,L1`) computes two measurement lines and
  aggregates them into the slip across the interface between them: `cod_um`,
  the relative displacement of the two sides (`u` on line L0 minus `u` on line
  L1). It is not an opening displacement; "COD" is a historical name kept in
  the column and option names. `--cod-method naive`
  (default since 2026-10-05) differences the lines directly;
  `--cod-method fillboundary` (optional; the default from 2026-07-09 to
  2026-10-05) extrapolates each line along its `∂u/∂y` to the midpoint between
  the lines (the interface row when `--interface-y` is used). Under
  fillboundary a frame CSV without the `uy` column is rejected with
  `cod_uy_column_missing` rather than silently degrading to naive.
- **Agent/MCP interface.** `pipeline/mcp_server.py` is a guarded FastMCP stdio
  facade with a write jail, a read allowlist and an audit log. It contains no
  numerics and runs the COD mode only. Runs go through a job runner that writes
  `<output_dir>/.mcp-job.json` (state and progress) and holds the GPU/output
  locks for the job's lifetime. A run tool blocks until the job ends unless
  `wait: false` is passed: then it returns a job handle at once, the job record
  is polled with `dic_query_manifest`, and `dic_job_cancel` stops the job. The
  client configuration is in [`docs/lab/simulacrum.md`](docs/lab/simulacrum.md).
- **Deployment paths.** `pipeline/run_dic.sh`, the MCP binary and the read
  allowlist all resolve through `FASTICGN_*` environment variables, then an
  optional `~/.config/fasticgn/config.toml` (template:
  `pipeline/config.example.toml`), then the repository's own `build/` trees,
  and only then the historical `/home/chonglang` paths — which still work but
  emit a deprecation warning. See
  [`pipeline/DIC_PIPELINE_GUIDE.md`](pipeline/DIC_PIPELINE_GUIDE.md) and
  [`docs/lab/simulacrum.md`](docs/lab/simulacrum.md).
- **Lab-specific documents.** These describe that one deployment — host name,
  dataset paths, historical run records — and not general usage. Every
  remaining absolute `/home/chonglang` path in the repository is inside one of
  them, or inside the documented deprecated fallback:
  [`docs/lab/simulacrum.md`](docs/lab/simulacrum.md) (lab runbook),
  [`pipeline/DIC_PIPELINE_GUIDE.md`](pipeline/DIC_PIPELINE_GUIDE.md)
  (deployment guide),
  [`docs/agent-interface-validation-2026-07-10.md`](docs/agent-interface-validation-2026-07-10.md),
  [`docs/release-v1.0.0.md`](docs/release-v1.0.0.md),
  [`docs/release-v1.1.0.md`](docs/release-v1.1.0.md) and
  [`docs/validation-cache-validity-2026-09-16.md`](docs/validation-cache-validity-2026-09-16.md)
  (dated validation records). `cmake --install` ships the dated records under
  `share/doc/` (they are historical evidence, not instructions) but not
  `docs/lab/`.

## Reference

CLI behaviour:

- `opencorr_cli_cuda_mvp` keeps the existing CLI flow: CPU image I/O, FFTCC initial guesses (batched cuFFT by default in CUDA builds; `--fftcc cpu` selects FFTW in FFTW-enabled builds), GPU ICGN refinement, CSV or raw binary output.
- `opencorr_cli_cuda_mvp --version` prints a title line and then `name=`, `git_hash=`, `build_date=`, `version=`, `git_dirty=`, `backend=` and `fftw=`, in that order. `version` is the semantic version (the CMake project version, with `-dev` between releases; `CITATION.cff` carries it with a leading `v`), `git_dirty` is `1` when tracked files differed from `git_hash` when the binary was built (`unknown` outside a git checkout), `backend` is `cuda` or `cpu`, and `fftw` says whether the FFTW CPU FFTCC is built in. `--help` (or `-h`) prints the usage text and every option with its default on stdout and exits 0; it is recognised anywhere except as the value of an option and after `--verify-only`. Usage errors print the usage text on stderr and exit 1.
- Every computing run ends with one JSON status line on stdout, also when it fails; `--quiet` drops the per-frame progress lines from stderr. See [process output](docs/output-format.md#process-output).
- `OPENCORR_GPU_MVP_ENABLE_CUDA=OFF` builds the CPU backend through OpenCorr's `ICGN2D1` and FFTW. It supports the same multi-frame lifecycle, including default skip/fallback, without CUDA. Runtime ICGN timing labels identify the CPU backend.
- CPU mode accepts `--gpu-device` and `--kernel` for command compatibility but does not use GPU devices or CUDA kernels; `--dual-gpu` is rejected in a CPU-only build.
- The CUDA path precomputes target-image bicubic coefficient matrices, reference gradients, and per-POI Hessian data before the ICGN loop, which runs one warp per POI by default (`--kernel warp`) or one block per POI with `--kernel legacy`. Launch configuration is logged at runtime, including dynamic shared-memory requirements and any opt-in above the default device limit.
- CSV rows with `zncc < 0` are failure/status rows, not valid measurements. Their `u` and `v` values are retained diagnostic state (initial seed, previous-frame seed, or last failed iteration) and must be excluded from displacement/COD analysis.
- Numeric CLI arguments are parsed strictly. Empty values, non-numeric suffixes such as `15px`, non-finite values, zero or negative subset/grid/thread/iteration values, and unknown options fail with a non-zero exit code.
- `--output-format raw` replaces the `.csv` extension with `.bin` and writes one little-endian 36-byte record per POI, plus a JSON manifest (`manifest.opencorr-raw.json` in multi-frame mode) describing the record layout and per-frame files. Use raw for production throughput, then run `smoke/raw_to_csv.py --manifest <raw-dir>/manifest.opencorr-raw.json --out-dir <csv-dir>` when downstream tools need the legacy per-frame CSV layout. `smoke/verify_raw_vs_csv.py` cross-checks raw output against a CSV run of the same job. The default remains CSV.
- Image input bit depth is selected with `--input-depth 8|12|16|auto`, `--input-align right|left|auto`, and `--input-normalize none|to-8|to-16|unit`. The default `--input-depth 8 --input-normalize none` matches legacy OpenCorr `IMREAD_GRAYSCALE`; high-bit modes read with `IMREAD_ANYDEPTH|IMREAD_GRAYSCALE`. For 12-bit data `right` keeps values in `0..4095` (and rejects inputs above 4095), while `left` recovers samples stored left-shifted into a 16-bit container (true value `<< 4`, low four bits zero) by dividing the stored `uint16` by 16; `--input-align` is only meaningful with `--input-depth 12`. `auto` infers 12-bit only when a 16-bit source has maximum `<= 4095`, so left-shifted 12-bit data (maximum `65504 = 4094 * 16`) is read as 16-bit — pass `--input-depth 12 --input-align left` explicitly for that layout. Because ZNCC is invariant to a global intensity scale, a correctly-aligned 12-bit run and the raw 16-bit run are bitwise-identical. The requested and effective input mode is logged at runtime and written to the raw manifest `input_image` block. With the default depth 8 a deeper file is reduced to 8 bits before correlation and the run still exits 0: the binary therefore reads the reference once more to learn its stored depth, warns on stderr and in the status line when it is deeper than 8 bits (naming the option to pass) or when a target frame is stored at another depth than the reference, and records `reference_source_depth`, `reference_source_range` and `target_source_depth_mismatches` in the same manifest block. The probe changes no output byte and is not part of the resume contract.
- In `--multi-frame` mode each `--tar-list` line is split on the last whitespace, so target image paths may contain spaces; the final token is the per-frame output CSV name (when the line contains a tab, the last tab is the separator instead, and a trailing CR is stripped for CRLF files).
- `--subset-size N` is an alias for the positional subset radius: it takes the odd full size and uses `radius = (N - 1) / 2`. Supplying both is rejected, an even `N` is rejected, and the computed output is byte-identical to the equivalent radius.

Validation and release helpers:

- `smoke/run_v9_cpu_gpu_parity.sh` generates a synthetic 10-frame sequence, runs CPU OpenCorr and FastICGN, joins common valid POIs by `(x,y)`, and checks every frame: median `|du|`, `|dv|` at most `1e-3 px`, nearest-rank p99 at most `1e-2 px`, maximum at most `0.5 px`, plus validity-retention and disagreement limits (`smoke/README.md`).
- `smoke/run_c1_reduce_probe.sh`, `run_c2_fallback.sh`, and `run_c6_subset_capacity.sh` preserve the phase-1 C1/C2/C6 minimal repros as repo-local smoke checks.
- `smoke/run_high_bit_input.sh` exercises high bit-depth input end to end: exact 16-bit and 12-bit left/right readback through `image_input_depth_probe`, legacy 8-bit compatibility, CLI preload and `--no-preload` synchronous loading, and the raw-manifest input metadata.
- `smoke/run_c4_shared_memory.sh` records dynamic shared-memory requirements for s41/s51 plus a larger synthetic subset that should exercise opt-in on Ada-class GPUs when the requested bytes exceed the default per-block limit.
- `smoke/run_c5_oob_boundary.sh` creates a near-boundary translation case and asserts that both the CPU and the GPU backend reject every POI of the predicted out-of-bounds band with `zncc = -3`; `smoke/run_oob_rejection_gate.py` (ctest `oob_rejection`) asserts the same rule for a single binary.
- `smoke/run_zero_texture_gate.py` (ctest `zero_texture`) paints black, mid-grey and saturated constant blocks onto synthetic speckle and asserts that every POI whose reference subset lies entirely inside a block is rejected before iterating (`zncc = -3` on the CUDA backend, `-5` on the CPU backend) while the POIs outside the blocks are untouched; with `--reference-binary` it also asserts that every other record is byte-identical between two builds.
- `smoke/run_frame_fallback_gate.py` (ctest `frame_fallback`) asserts the whole-frame fallback's per-POI merge (`frame_fallback_policy`) on two synthetic sequences with known truth, one whose motion grows past FFTCC's capture range with an occluded frame and one with a 10 px jump: no fallback frame ends with more bad POIs than its seeded attempt, every POI the from-scratch pass solves (`zncc >= 0.8`) stays good, output does not depend on the thread count, and coverage and accuracy hold.
- `smoke/run_subpixel_accuracy_gate.py` (ctest `subpixel_accuracy`, also a `selfcheck.sh` stage) renders noisy 8-bit speckle under an affine sub-pixel displacement field with exact truth (`smoke/generate_subpixel_speckle.py`, standard-library Python) and asserts, for the default kernel and on CUDA builds `--kernel legacy`, at radius 7 and radius 31: the POI grid the radius implies, valid ratio, RMS and mean error against the truth, gradient error, median `zncc` and the iteration count; and that a run with `--subset-size 15` and no `--conv-threshold` / `--max-iter` is byte-identical to radius 7 with 0.001 and 10 spelled out. It is the check that needs no lab data for a convergence-loop bug, a changed default threshold or a size taken as a radius.
- `docs/release-v1.0.0.md` and `docs/release-v1.1.0.md` are dated records of those two releases (lab paths of that time), not current instructions.

Boundary behavior:

- A POI whose warped target subset has any sample outside the bicubic interpolation domain (`x < 1`, `y < 1`, `x >= width - 2`, or `y >= height - 2`) is rejected with `zncc = -3` in that iteration, before any substituted value can enter the correlation, by the CPU backend and by both CUDA kernels alike; raw manifests record this as `"oob_policy": "geometric-v1"`. The rule, its history and the separate CUDA-only reference-side rejection are documented in [output format: out-of-bounds target subsets](docs/output-format.md#out-of-bounds-target-subsets).

Dynamic shared-memory notes:

- Runtime logs include `precompute_shared`, `iteration_shared`, `default_shared_limit`, and `optin_shared_limit`.
- If a kernel needs more than the default limit but not more than the opt-in limit, FastICGN applies `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` before launch.
- Before adding larger subsets to production sweeps, run `smoke/run_c4_shared_memory.sh` on the target GPU and confirm that the requested `iteration_shared` is below `optin_shared_limit`. On an Ada GPU with 49152 B default and about 101376 B opt-in limit, the practical square-subset ceiling depends on the selected block size; for 32-thread iteration blocks `iteration_shared` is `1960 + 16 * N` bytes, where `N` is the number of pixels in the subset. These numbers describe the legacy block-per-POI iteration kernel; the default warp kernel uses no dynamic shared memory for iteration, but the same limit check still runs when a reference is prepared, so it caps the subset size in either mode.

Library-embedding notes:

- In CPU-only builds the ICGN wrapper performs actual reference-gradient and target-interpolation preparation. It sizes its CPU instance pool to the current OpenMP maximum and safely rebuilds preparation when that setting changes between calls. Callers must not concurrently use one solver from multiple host threads. Reference-only preparation is valid before setting a target; `prepare()`, explicit per-side preparation, and lazy `compute()` are supported.
- CPU `adoptPreparedTar()` prepares its own interpolation storage from the donor's current target image, rather than borrowing CUDA buffers. `computeGathered()` returns false in CPU mode, so the existing fallback caller uses normal CPU compute.
- Known CPU cost: that preparation re-runs `BicubicBspline::prepare()` over the whole target, allocating 16 floats (64 bytes) per pixel, so a CPU-only run that repairs POIs keeps two full coefficient tables resident (about 1.4 GB each at 21.5 MPx). The CLI now calls `adoptPreparedTar()` only when a frame actually has repair candidates, but every such frame still rebuilds the table once; the CUDA path borrows the donor buffer instead and pays nothing. Repeated CPU adoption of the same successful target-preparation generation is now reused. A donor prepareTar call, recipient target change, or CPU worker-pool rebuild forces fresh interpolation. The generation is process-unique, so image/donor address reuse is not an identity check.
- CUDA solvers owned by different host threads may run at the same time; one solver, or an ICGN solver together with the solvers that borrow its target or gather from it, stays on one thread at a time. A successful `computeGathered()` leaves the recipient without a prepared POI layout (its next `compute()` rebuilds one), and `FFTCC2DCuda::compute()` throws `std::invalid_argument` when the reference and target sizes differ. See [library contracts](docs/library-contracts.md).
- CUDA image setters are change notifications, not pointer-identity checks. Reusing an `Image2D` at the same address is supported: after changing its contents, call the appropriate setter before `compute()`. `setRefImage()` invalidates the uploaded reference and its derived caches; `FFTCC2DCuda` also invalidates prepared reference spectra. `setImages()` invalidates both reference and target state. Keep image objects alive while the solver uses them.
- For a fixed-reference sequence, call `setRefImage(ref)` and `prepareRef(pois)` once, then `setTarImage(target)` for each new target. `compute()` lazily prepares invalidated CUDA state. The CLI uses this pattern so the cache correctness fix does not recompute the reference every frame. Direct mutation without a subsequent setter is outside this contract.
- Both FFTCC backends retain the POI layout established by `prepareRef(pois)`. After `setRefImage()`, a matching unindexed queue or a valid indexed request rebuilds its reference spectra once. A mismatched unindexed queue goes directly uncached, leaving the retained layout invalid until actually requested. Call `prepareRef(pois)` explicitly to replace the retained layout. The CPU setter also dispatches through `DIC::setImages`; the upstream CPU checkout is unchanged.
- The 2D FFTCC public engines accept positive **square** subset radii only. Non-square/invalid radii throw before allocation or CUDA work; rectangular FFT memory layout and CPU/GPU numerical equivalence remain unsupported. CPU `FFTCC2D` geometry is fixed at construction: inherited `setSubset` or public-field edits are rejected by preparation/compute when they change it. See [library contracts](docs/library-contracts.md).
- ABI note: making `DIC::setImages` virtual inserts a slot in the vendored `DIC` vtable. C++ clients that link this static library must be recompiled against the matching headers; object files compiled before this change must not be mixed with the new library, even though `DIC` already had a vtable and object sizes are unchanged.
- `compute(queue, indices)` requires a previously prepared layout. Wrong index counts, out-of-range indices, or mismatched integer reference coordinates throw; this overload never silently ignores indices. Reordered and repeated slots are supported. Unindexed calls without a prepared layout, or with a different queue, retain the documented uncached path; the CPU single-POI overload also remains uncached. Lazy rebuilding refers to an existing prepared layout, not automatic cache creation for every call.
- The CLI checks that both engines hold the same prepared reference before processing a frame. "Prepared" is checked per engine: the FFTCC backend must still hold the reference spectra layout from `prepareRef(pois)`, and the CUDA ICGN engine must hold the uploaded reference plus the per-POI precompute layout on every device `prepareRef(pois)` populates (both devices under `--dual-gpu`). This runtime check remains active in Release builds.
- The 2026-09-16 cache regression covers `0 -> 5 -> 0` and `5 -> 0 -> 5` pixel shifts, mutable target/reference objects, reference replacement, and indexed FFTCC. See `smoke/README.md`. Pre-fix multi-frame raw hashes are historical reproducibility records, not correctness gates for the repaired implementation.
- The CUDA buffer registry is keyed by `ICGN2D1CudaMvp::Impl*` and device id, and the wrapper destructor releases registry entries for that instance.
- Reusing one `ICGN2D1CudaMvp` object across different subset sizes is supported by capacity tracking on both POI count and subset size, but callers must still run `setImages`/`setRefImage`, `prepareRef`, `prepareTar`, and `compute` in that order after changing images or subset configuration.
- The CLI still requires reference and target images to have identical dimensions. The CUDA iteration uses `ref_width` for reference-gradient indexing and `tar_width` for target interpolation, but mixed-size image pairs remain outside the supported public CLI contract.

Release provenance:

- `opencorr_gpu_mvp_version.h` is generated when CMake configures the build tree and refreshed by every build: `git_hash` and `git_dirty` describe the tree that was compiled (editing a tracked file or moving `HEAD` and rebuilding updates them), `build_date` is the time the tree was configured, and `version` comes from `CMakeLists.txt`. For a release, configure a fresh build tree from the tagged commit and check that `git_dirty=0`.
- A release build should verify that `opencorr_cli_cuda_mvp --version` prints a `git_hash` equal to both `git rev-parse --short=12 HEAD` and the release tag, for example `v1.0.0`.
- The public repository starts at a single initial commit. Commit hashes and the tags `v1.0.0` / `v1.1.0` that the documentation, the smoke scripts and the code comments cite from before that commit (for example `638ad18`, the 2026-10-03 pipeline and binary that the summary's `warnings` wording is held compatible with) belong to the author's private development history and do not resolve in the public repository; they date a measurement or a behaviour, nothing in the tree depends on them. Upstream OpenCorr hashes (`994337a`, `5f84be6f`) resolve in the upstream repository.

Numerical parity:

- The CUDA ICGN results are not bit-identical to CPU OpenCorr: both CUDA kernels sum in a different order from the CPU code (the default warp kernel with warp-shuffle reductions, the legacy kernel with block-wide trees), and the two kernels are separate numerical baselines with their own production anchors. CPU/GPU agreement is held by the per-frame tolerances of `smoke/check_cpu_gpu_parity.py`.
- Numerical changes must pass CTest, CPU/GPU parity and the production anchors. Any intended anchor change must be documented; never relax expected hashes or thresholds merely to make a test pass.

## Checked output and resume

Multi-frame jobs validate inputs, configuration, binary identity, and committed
frame contents before resuming. Partial or extended jobs replay from frame 1 to
restore temporal state, preserving verified files; complete jobs return after
validation. Old untracked outputs are rejected. CSV/raw payload layouts and
solver calculations are unchanged. A rejected resume names the contract field
that differs with its recorded and requested value. Options that only schedule
the work (`--threads`, `--preload`, `--decode-threads`, `--gpu-device`) are not
part of that contract: they never change an output byte, so a run may be
resumed or verified under other values.

`--verify-only <output_dir>` answers "is this directory complete, and does every
file still match the ledger?" without computing or writing anything: one JSON
object on stdout, exit 0 only for a complete directory.

`--durability rename` (default) publishes every file through a private temporary
plus atomic rename/link and issues **no fsync per frame**: killing the process
still never leaves a partial file. `--durability fsync` additionally forces every
payload, ledger record and directory to the device for power-loss durability, at
roughly four fsync per frame. `--repair` moves damaged tracked frames into
`.opencorr-state/quarantine/<timestamp>/` and regenerates them by replay instead
of refusing the whole directory; changed inputs, parameters, binary, reordered or
shortened target lists, unknown files and symlinks stay rejected either way.

See [output integrity](docs/output-integrity.md) for atomic publication,
durability policies, repair, failure retention, compatibility, and replay costs.

## License

FastICGN is released under the **Mozilla Public License, version 2.0**
(MPL-2.0). The full text is in [`LICENSE`](LICENSE); every first-party source
file carries the MPL-2.0 Exhibit A notice.

```
Copyright (c) 2026 Chonglang Wang
```

The example image sequence under [`examples/`](examples/) is data, not code:
it is released under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0, text in `examples/smooth-30kN-0725B-B2L2/LICENSE`),
copyright Chonglang Wang; its `MANIFEST.md` gives the attribution line and the
provenance of every file (NOTICE.md §10).

[`NOTICE.md`](NOTICE.md) is the authoritative third-party inventory. In short:

- **Vendored upstream OpenCorr** (`src/oc_*.{h,cpp}`, `src/opencorr.h`) is
  MPL-2.0, Copyright (C) 2021-2025 Zhenyu Jiang, taken from upstream commit
  `994337a74b051f5b28a305299df3bfc12d08288f` (2026-08-28, untagged on `main`;
  the previous baseline was `5f84be6`).
- **Six of those files are modified** — `src/oc_fftcc.{h,cpp}` (reference
  spectrum caching, `prepareTar`, setter invalidation, square-subset
  enforcement), `src/oc_icgn.cpp` (upstream's out-of-bounds rejection kept,
  with a geometric detector), `src/oc_image.{h,cpp}` (12/16-bit input, bit
  alignment, normalisation) and `src/oc_dic.h` (`setImages` made `virtual`).
  Each keeps its original upstream notice, carries a
  `// Modified for OpenCorr-GPU` marker, and **continues to be offered under
  MPL-2.0**. The other 38 vendored files are byte-identical to the baseline.
- **Vendored third-party headers**: nanoflann v1.9.0 (BSD-2-Clause),
  nlohmann/json 3.12.0 (MIT), xxHash 0.8.3 (BSD-2-Clause).
- **External dependencies**: Eigen (MPL-2.0), OpenCV (Apache-2.0), CUDA Toolkit
  (NVIDIA EULA), OpenMP runtime (GPL + GCC Runtime Library Exception).
- **FFTW is GPL-2.0-or-later and optional.** The production CUDA build does not
  link it (`WITH_FFTW` defaults to `OFF` when CUDA is enabled; FFTCC runs on
  cuFFT). CPU-only builds and `-DOPENCORR_GPU_MVP_WITH_FFTW=ON` builds do link
  it, and **distributing such a binary puts the combined binary under the GPL**
  — permitted by MPL-2.0 §3.3 (Secondary Licenses). See NOTICE.md §6.

`python3 smoke/check_license_headers.py` is the machine-checkable gate: it fails
if a first-party file is missing its notice, if an unmodified vendored file
drifts from the recorded upstream SHA-256, or if a modified vendored file loses
its upstream notice or modification marker.

Please cite this software using [`CITATION.cff`](CITATION.cff), and cite the
upstream library as Z. Jiang, *Optics and Lasers in Engineering* (2023) 165:
107566, <https://doi.org/10.1016/j.optlaseng.2023.107566>.
