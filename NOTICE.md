# NOTICE

This file records the licensing of FastICGN and of every third-party
component it vendors, links against, or requires at run time. It complements
[`LICENSE`](LICENSE), which carries the verbatim text of the Mozilla Public
License 2.0.

## 1. Project license

FastICGN is licensed under the **Mozilla Public License, version 2.0**
(MPL-2.0). The whole repository is covered, including both the first-party
sources listed in section 3 and the vendored OpenCorr sources described in
section 2, with one exception: the example image sequence under `examples/`
is data, released under CC BY 4.0 (section 10).

```
Copyright (c) 2026 Chonglang Wang

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

MPL-2.0 is a per-file weak copyleft licence. Modifying a file that carries the
notice above obliges you to publish the modified file under MPL-2.0; combining
these files with separately licensed code in a Larger Work does not.

**A note on Git history and authorship.** The public repository starts at a
single initial commit. The development history before it is kept in the
author's private repository and is not published; part of it was recorded
under tooling identities rather than under a person's name. All of that work
was produced for, and its copyright is held by, the copyright holder named
above. Commit hashes and the tags `v1.0.0` / `v1.1.0` that the documentation,
the smoke scripts and the code comments cite from before the initial commit
refer to that private history and do not resolve here; they date a
measurement or a behaviour, and nothing in the tree depends on them.

## 2. Upstream OpenCorr (vendored, MPL-2.0)

| Field | Value |
| --- | --- |
| Project | OpenCorr — open source C++ library for 2D, 3D/stereo and volumetric digital image correlation |
| Home | <https://www.opencorr.org/> |
| Repository | <https://github.com/vincentjzy/OpenCorr> |
| Copyright | Copyright (C) 2021-2025, Zhenyu Jiang \<zhenyujiang@scut.edu.cn\> |
| License | MPL-2.0 (same as this project) |
| Vendored baseline | commit `994337a74b051f5b28a305299df3bfc12d08288f`, branch `main`, authored 2026-08-28 00:16:53 +0800 by `vincentjzy`, subject *"Revert \"debug: fix the knn_search in feature_affine without correct k-value\""* |
| Release tag | **none** — the baseline is an untagged commit on `main`, and upstream has issued no library release tag since `v1.0` (2022-12-23). The only tag between `v1.0` and this commit is `GUI_3.0` (2026-05-19), which releases the separate GUI program and changes no `src/` file. A commit hash is therefore the only precise identifier available. |
| Previous baseline | commit `5f84be6fe5520d92e9eb6a555fe3caf1d10e4a2a` (2025-05-06). Section 2.4 records what the move to `994337a` brought in. |
| Vendored into | `src/oc_*.{h,cpp}` (43 files) and `src/opencorr.h` |

Citation for the upstream library:

> Z. Jiang, *OpenCorr: An open source library for research and development of
> digital image correlation.* Optics and Lasers in Engineering (2023) 165:
> 107566. <https://doi.org/10.1016/j.optlaseng.2023.107566>

### 2.1 How the baseline was determined

The original baseline was identified by hashing every `src/oc_*` file and
`src/opencorr.h` as a Git blob and matching them against the `src/` tree of all
83 upstream commits that touch `src/`. Commit `5f84be6f` was the unique maximum:
**37 of 37** files that we did not modify matched it byte for byte, with no
other commit matching more than 36.

The current baseline `994337a` is the upstream `main` HEAD that tree was then
realigned onto (section 2.4). It is verified the same way and holds the same
property: **38 of 38** vendored files we did not modify — the 36 originals that
were already unmodified plus the two `oc_region_fit` files added by `be61daf` —
are byte-identical to `994337a`, and the only remaining differences are the six
files in section 2.2. `smoke/check_license_headers.py` pins all 38 digests.

### 2.2 Vendored OpenCorr files modified in this project

The following six files differ from the `994337a` baseline. Each retains its
original upstream copyright and MPL-2.0 notice, and carries an added marker line
`// Modified for OpenCorr-GPU; see NOTICE.md for the list of changes.`
(the marker keeps its original spelling; renaming it would invalidate every
recorded modification marker and the `smoke/check_license_headers.py` gate)
**These files continue to be made available under MPL-2.0.**

| File | Diff vs. baseline | What changed |
| --- | --- | --- |
| `src/oc_fftcc.cpp` | +290 / −18 | Reference-spectrum caching, a `prepareTar` entry point (on the CPU it only validates the subset geometry), setter invalidation rules that drop the cache when geometry or images change, and rejection of rectangular (non-square) subsets. |
| `src/oc_fftcc.h` | +52 / −20 | Declarations and cache/ownership members for the changes in `oc_fftcc.cpp`. |
| `src/oc_icgn.cpp` | +123 / −11 | The out-of-bounds target-subset rejection upstream added in `baa3de6`, with its detector replaced by a geometric test, and a NaN sample coordinate reported as `-5` instead of `-3`. Section 2.5. |
| `src/oc_image.cpp` | +253 / −25 | 12- and 16-bit high-bit-depth image input, left/right bit alignment for 12-bit data stored in a 16-bit container, and optional normalisation of the loaded samples. |
| `src/oc_image.h` | +86 / −21 | Declarations, enums and members for the high-bit-depth input path in `oc_image.cpp`. |
| `src/oc_dic.h` | +2 / −1 | `setImages` made `virtual` so a GPU subclass can override image binding. |

Total: 806 insertions, 96 deletions across 6 files, measured with
`git diff --numstat` against the `994337a` blobs. Every count includes the
one-line modification marker itself. Most of the deletions are line-ending
changes: ignoring a carriage return at end of line, `oc_fftcc.cpp`,
`oc_fftcc.h`, `oc_image.cpp` and `oc_image.h` count +273 / −1, +32 / −0,
+228 / −0 and +65 / −0. `src/oc_fftcc.cpp` additionally carries the
`num_threads(thread_number)` pin described in section 2.4 on the three parallel
regions this project added, with a five-line comment at each.

The remaining **38** vendored files (37 `oc_*` files plus `src/opencorr.h`) are
byte-identical to the baseline and carry only their original upstream notices.
`smoke/check_license_headers.py` pins their SHA-256 digests and fails if any of
them drifts.

### 2.3 Vendored headers redistributed by `cmake --install`

The install step ships the public C++ API named in
[`docs/library-contracts.md`](docs/library-contracts.md) plus the vendored
OpenCorr headers those three classes include, into
`<prefix>/include/opencorr-gpu/`. All of them are OpenCorr files under
MPL-2.0 and are covered by section 2 above:

`oc_array.h`, `oc_cubic_bspline.h`, `oc_deformation.h`, `oc_dic.h`,
`oc_fftcc.h` (FFTW builds only), `oc_gradient.h`, `oc_icgn.h`, `oc_image.h`,
`oc_interpolation.h`, `oc_poi.h`, `oc_point.h`, `oc_subset.h`.

Of those, `oc_dic.h`, `oc_fftcc.h` and `oc_image.h` are among the six files
modified in this project (section 2.2) and carry the modification marker.
`src/nanoflann.hpp` (section 4) is **not** installed, because no public header
includes it. Neither is `oc_region_fit.h` (section 2.4): upstream does not list
it in `src/opencorr.h`, this project does not build or call it, and no installed
header includes it. The full unmodified vendored sources remain in this repository's
`src/` tree, which is how MPL-2.0 source availability is satisfied.

### 2.4 Moving the baseline from `5f84be6f` to `994337a`

Upstream produced 20 commits between the two, of which nine touch `src/`. The
realignment took the net `5f84be6f..994337a` diff of `src/` wholesale, so the
vendored tree is a faithful snapshot of upstream `main` at `994337a` except for
the six files in section 2.2. Eight of the nine applied without conflict; the
ninth, `0a795a0`, was applied by hand because it edits `src/oc_fftcc.cpp`, which
this project has modified.

| Upstream commit | Effect on this project |
| --- | --- |
| `df60be0` | `delete2D`/`delete3D` now free the backing data block, not just the pointer table. Fixes a genuine leak; changes no numerical result. |
| `c486663` | Adds a missing `#include <random>` to `oc_feature_affine.cpp`. Compile portability only; this project does not call `FeatureAffine`. |
| `baa3de6` | ICGN rejects a target subset that is partially out of the target image's interpolatable range with `zncc = -3` instead of feeding the interpolator's `-1.f` sentinel into the correlation cost. **Changes numerical output** on the CPU ICGN backend. The same rule was ported to this project's CUDA kernels, which had the equivalent gap; see [`docs/output-format.md`](docs/output-format.md). Upstream's **detector** is not kept — section 2.5. |
| `0a795a0` | Pins `num_threads(thread_number)` on the parallel-for loops that pick a worker out of a fixed-size instance pool with `omp_get_thread_num()`, which otherwise throws `CPU thread ID over limit` when OpenMP's default team is larger than the pool. Applied by hand to `src/oc_fftcc.cpp` and **extended to the three parallel regions this project added there** (`FFTCC2D::prepareRef`, `FFTCC2D::compute(queue, cache_indices)`, and the cached fast path of `FFTCC2D::compute(queue)`), which upstream's patch cannot reach. The two pool-construction loops index by the loop variable rather than by thread id and are left as upstream has them. |
| `9fdbc62`, `e85d781` | `FeatureAffine3D` fixes (a `wy`/`wz` copy-paste error and an uninitialised `subset_radius_z`). 3D/DVC only; unused here. |
| `385d809` + `994337a` | A `knnSearch` k-value change and its complete revert the next day. Net diff zero. |
| `be61daf` | Adds `src/oc_region_fit.{cpp,h}` (`RegionFit2D`/`RegionFit3D`, a RANSAC neighbourhood fit for POIs that ICGN failed from a poor initial guess). Vendored for completeness. Upstream does not list it in `src/opencorr.h`; neither do we, it is not compiled into any target here, and it is not installed. |

The remaining eleven commits are documentation, examples, the `GUI_3.0` release
or merge commits with no changes of their own, and touch no `src/` file.
`LICENSE` is unchanged between the two baselines, and `src/opencorr.h` is
byte-identical in both.

### 2.5 The one deliberate deviation from upstream: how `baa3de6` detects the condition

`src/oc_icgn.cpp` is the vendored file where this project changes an upstream
*numerical* decision. (`src/oc_fftcc.cpp` also overrides one, by rejecting
rectangular subsets that upstream accepts, but that is a guard over an
unimplemented case rather than a different answer to the same question — see
section 2.2.) The rule upstream introduced in `baa3de6` is kept exactly — a warped target
subset that leaves the target image's interpolatable range rejects its POI with
`zncc = -3`, in the iteration that produced it, before the subset's zero-mean
norm, leaving the deformation at the incoming guess. What changes is how that
condition is **detected**.

Upstream scans the reconstructed subset for a negative value, because
`BicubicBspline::compute()` returns exactly `-1.f` for a coordinate it cannot
interpolate, and "a real pixel intensity is never negative". The premise does
not survive the interpolation: a cubic B-spline undershoots below zero wherever
the data has a sharp intensity edge, and a dark background with bright speckle
is exactly such data. Measured on this project's reference sequence (ten frames,
1 710 240 shared POIs, 8-bit, a dark band where 850-915 of each row's 1036
pixels are exactly zero):

| | |
| --- | ---: |
| POIs the value test newly rejects | 277 083 |
| of those, every sample inside the image | **273 619 (98.75 %)** |
| of those, previously valid measurements (`zncc >= 0`) | 200 976 |
| valid records, before → after | 1 267 708 → 1 066 732 (**−15.9 %**) |
| in-bounds samples that interpolate negative, probed directly with OpenCorr's own B-spline coefficients | 5 723 of 30 459 (**18.8 %**), most negative −149.1 |

This project therefore tests the coordinate instead of the value, using
`BicubicBspline::compute()`'s own out-of-range condition
(`x < 1 || y < 1 || x >= width - 2 || y >= height - 2`) evaluated as each sample
is placed. It is the same predicate the CUDA kernels apply in
`sampleTargetValue()`, so the CPU and GPU backends reject the same subsets and
stay comparable. The helper is `outOfInterpolatableRange()` at the top of
`src/oc_icgn.cpp`, and it is used in all four 2D `ICGN2D1`/`ICGN2D2` overloads.

A **NaN** sample coordinate is deliberately outside that predicate. Upstream's
value test cannot separate it from an out-of-range one, because the
interpolator returns the same `-1.f` for both, so under `baa3de6` a non-finite
iterate is reported as `-3`. Here it keeps its own flag and the `-5` that this
file already assigns after the loop for a non-finite result, which is also what
the CUDA kernels assign for the same cause. Out of range is tested first, so a
subset that is both is reported as `-3` on either backend.

`ICGN3D1::compute()` keeps upstream's value test unchanged. This project builds
no 3D path and has no volumetric data to measure the same effect on, so leaving
it is the smaller claim; the same argument would apply to `TricubicBspline`.

## 3. First-party sources

Written for this project and licensed MPL-2.0 under the copyright in section 1:

- `opencorr_cli_cuda_mvp.cpp`, `opencorr_cli.cpp`
- `src/cli_output_integrity.{h,cpp}`, `src/cli_image_probe.{h,cpp}`
- `src/fftcc2d_cuda.{h,cpp,cu}`
- `src/icgn2d1_cuda_mvp.{h,cpp,cu}`
- `src/opencorr_gpu_mvp_version.h.in`
- `CMakeLists.txt`, `cmake/fasticgn_version_header.cmake`,
  `smoke/c6_subset_capacity_probe/CMakeLists.txt`
- `smoke/*.{cpp,cu,py,sh}`
- `pipeline/*.py`, `pipeline/run_dic.sh`
- `docs/*.md`, `docs/lab/*.md`, `README.md`, `smoke/README.md`

`opencorr_cli.cpp` is a first-party command-line wrapper written against the
public OpenCorr API. Its call sequence follows the pattern of the upstream 2D
DIC examples (`examples/test_2d_dic_fftcc_icgn1.cpp` and siblings), but it is
not a copy of any upstream file, and no upstream example is vendored here.

## 4. Third-party components vendored in this repository

These are redistributed unmodified, under their own licenses. Their licenses are
compatible with MPL-2.0 and impose only attribution obligations.

| Component | Version | Path | License | Notes |
| --- | --- | --- | --- | --- |
| [nanoflann](https://github.com/jlblancoc/nanoflann) | v1.9.0 | `src/nanoflann.hpp` | BSD-2-Clause | Header-only KD-tree. Upstream OpenCorr treats nanoflann as an *external* dependency and does not vendor it; this project vendors the header so the tree builds without a separate install. SHA-256 `e91b910b6fcdeaefb0253d50438c519de675980cbf3374397da730dfc23bbfe7`, byte-identical to `include/nanoflann.hpp` at tag `v1.9.0`. Copyright 2008-2009 Marius Muja; 2008-2009 David G. Lowe; 2011-2025 Jose Luis Blanco. |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | `third_party/nlohmann/json.hpp` | MIT | License text at `third_party/nlohmann/LICENSE.MIT`. Copyright (c) 2013-2025 Niels Lohmann. Used only by the CLI output/metadata layer. |
| [xxHash](https://github.com/Cyan4973/xxHash) | 0.8.3 | `third_party/xxhash/xxhash.h` | BSD-2-Clause | License text at `third_party/xxhash/LICENSE`. Copyright (c) 2012-2021 Yann Collet. XXH3-128 is used as a non-cryptographic integrity fingerprint only; release gates use SHA-256. |

Download URLs, byte counts and SHA-256 digests for the `third_party/` headers
are recorded in `third_party/sources.json`.

## 5. External dependencies (not distributed with this repository)

These must be present to build or run FastICGN. They are *not* redistributed
here; the versions are those validated on the project's reference machine.

| Dependency | Validated version | License | How it is used |
| --- | --- | --- | --- |
| [Eigen](https://eigen.tuxfamily.org/) | 3.4.0 | MPL-2.0 (with some modules under BSD-3 / LGPL-2.1+; none are used here) | Header-only linear algebra, used by the vendored OpenCorr solvers. |
| [OpenCV](https://opencv.org/) | 4.6.0 | Apache-2.0 (OpenCV 4.5.0 and later) | Image decoding and grayscale/high-bit-depth input. |
| [FFTW](https://www.fftw.org/) (`fftw3`, `fftw3f`, `fftw3l`) | 3.3.10 | **GPL-2.0-or-later** | Optional. See section 6. |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (`cudart`, `cufft`) | 13.2 | NVIDIA CUDA Toolkit EULA (proprietary) | See section 7. |
| OpenMP runtime (GCC `libgomp`) | as shipped with the system compiler | GPL-3.0-or-later **with the GCC Runtime Library Exception** | Host-side parallel loops. The Runtime Library Exception is what lets non-GPL programs link it. |

### 5.1 Python components

Used by `pipeline/` and `smoke/`; not redistributed here.

| Package | License | Where required |
| --- | --- | --- |
| [NumPy](https://numpy.org/) | BSD-3-Clause | `pipeline/dic_pipeline.py`, `smoke/generate_synthetic_dic.py` and other analysis helpers |
| [Pillow](https://python-pillow.org/) (`PIL`) | MIT-CMU (HPND) | `pipeline/dic_pipeline.py` (first-frame probes for `--dry-run` and the run summary) and the pipeline tests |
| [FastMCP](https://github.com/jlowin/fastmcp) | Apache-2.0 | `pipeline/mcp_server.py`; pinned as `fastmcp==3.4.4` in `pipeline/requirements-mcp.txt` |

Everything else imported by `pipeline/` and `smoke/` is Python standard library
(PSF License, bundled with the interpreter).

## 6. FFTW and the GPL — read this before distributing binaries

FFTW is licensed **GPL-2.0-or-later**. It backs only the legacy CPU FFTCC path.

- FFTW is an **optional** dependency, controlled by the CMake option
  `OPENCORR_GPU_MVP_WITH_FFTW`.
- The **production CUDA build does not link FFTW at all.** With
  `OPENCORR_GPU_MVP_ENABLE_CUDA=ON`, `WITH_FFTW` defaults to `OFF`, FFTCC runs
  on batched cuFFT (`src/fftcc2d_cuda.*`), and CMake neither searches for nor
  links `fftw3*`. Source distribution and binary distribution of that build are
  governed by MPL-2.0 alone.
- CPU-only builds (`ENABLE_CUDA=OFF`) require FFTW, and a CUDA build configured
  with `-DOPENCORR_GPU_MVP_WITH_FFTW=ON` links it as well.
- **If you distribute a binary that links FFTW, the combined work must be
  distributed under the GPL.** MPL-2.0 explicitly permits this: MPL-2.0 §3.3
  makes the GPL a "Secondary License", so MPL-licensed files may be combined
  with and redistributed under the GPL. The MPL-covered files stay available
  under MPL-2.0; the *combined binary* is what carries GPL terms. Nothing in
  this repository is marked "Incompatible With Secondary Licenses" (MPL-2.0
  Exhibit B), so this combination is allowed.
- Using an FFTW-linked build privately, or for internal research, triggers no
  distribution obligation at all.

An alternative for downstream redistributors who need the CPU FFT path without
GPL terms is to purchase a non-free FFTW licence from MIT, or to substitute a
permissively licensed FFT.

## 7. CUDA Toolkit

`cudart` and `cufft` are covered by the **NVIDIA CUDA Toolkit End User License
Agreement** (<https://docs.nvidia.com/cuda/eula/>), a proprietary licence. The
CUDA Toolkit is not redistributed by this project — you obtain it from NVIDIA
and accept the EULA yourself. If you redistribute binaries that bundle CUDA
runtime libraries, the EULA's redistribution terms (Attachment A) apply to those
libraries; they do not alter the MPL-2.0 terms of this project's own source.

## 8. License compatibility summary

| Component | License | Compatible with an MPL-2.0 repository? |
| --- | --- | --- |
| OpenCorr (vendored) | MPL-2.0 | Yes — identical licence; modified files stay MPL-2.0. |
| nanoflann (vendored) | BSD-2-Clause | Yes — permissive; attribution retained in the file header. |
| nlohmann/json (vendored) | MIT | Yes — permissive; attribution retained in `third_party/`. |
| xxHash (vendored) | BSD-2-Clause | Yes — permissive; attribution retained in `third_party/`. |
| Eigen | MPL-2.0 | Yes — identical licence, and header-only. |
| OpenCV | Apache-2.0 | Yes — permissive; MPL-2.0 works may link Apache-2.0 code. |
| CUDA Toolkit | NVIDIA EULA (proprietary) | Yes for our purposes — MPL-2.0 is file-scoped, so linking a proprietary runtime is permitted. NVIDIA's own redistribution terms still apply to NVIDIA's files. |
| OpenMP runtime (libgomp) | GPL-3.0-or-later + GCC Runtime Library Exception | Yes — the Runtime Library Exception exists precisely to permit this. |
| NumPy / Pillow / FastMCP | BSD-3 / MIT-CMU / Apache-2.0 | Yes — permissive, and invoked as separate processes rather than linked. |
| **FFTW** | **GPL-2.0-or-later** | **Conditionally.** Source distribution is unaffected. Building and using it is unaffected. *Distributing a binary that links FFTW* puts the combined binary under the GPL (permitted by MPL-2.0 §3.3, Secondary Licenses). The production CUDA build does not link FFTW. |
| Example image sequence (`examples/`) | CC BY 4.0 | Not code: data shipped beside the software, outside the MPL-2.0 covered files; redistributing it requires the attribution in section 10. |

## 9. Verifying this file

`smoke/check_license_headers.py` is the machine-checkable companion to this
notice. It asserts that every first-party source file carries the MPL-2.0
Exhibit A notice and the project copyright line, that the 38 unmodified vendored
OpenCorr files and `src/nanoflann.hpp` still match their recorded upstream
SHA-256 digests, and that the six modified vendored files still carry both the
upstream notice and the modification marker.

```
python3 smoke/check_license_headers.py
```

It checks source files only; the example data of section 10 is outside its
scope (its own `MANIFEST.md` lists the SHA-256 of every file).

## 10. Example data in `examples/` (CC BY 4.0)

`examples/smooth-30kN-0725B-B2L2/` holds 123 JPEG frames (7 698 361 bytes)
of one laboratory stick-slip event recorded in the author's laboratory on
2023-07-25 with a Photron FASTCAM SA1.1 at 50 000 fps: the reference frame and
target frames 443 to 564. The pixels are byte-identical copies of the camera
files; `MANIFEST.md` beside them records the acquisition parameters, the
calibration and measurement-line conventions, the frame-selection rule, the
expected results and the SHA-256 of every file.

| Field | Value |
| --- | --- |
| Copyright | Copyright (c) 2026 Chonglang Wang |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), full text in `examples/smooth-30kN-0725B-B2L2/LICENSE`, <https://creativecommons.org/licenses/by/4.0/> |
| Attribution | "FastICGN example sequence smooth-30kN-0725B-B2L2, laboratory stick-slip experiment, Chonglang Wang, CC BY 4.0", with a link to this repository; a publication that uses it should also cite the software (`CITATION.cff`) |
| Scope | the image files, `MANIFEST.md` and `README.md` under `examples/`. The software that reads them stays MPL-2.0; `cmake --install` does not install the example data |
