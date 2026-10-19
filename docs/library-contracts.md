# Supported 2D library contracts

The public engines used by the CLI are `FFTCC2DCuda`, the vendored `FFTCC2D`,
and `ICGN2D1CudaMvp`. This document does not qualify every upstream OpenCorr
class as a supported public API. The original CPU checkout remains unchanged.

## Installed headers

`cmake --install` ships exactly these headers into
`<prefix>/include/opencorr-gpu/`, and nothing else. The list is the transitive
include closure of the three supported engines, so each one compiles on its
own against what is installed — `smoke/check_installed_headers.sh --prefix DIR`
proves it, one translation unit per header, and also fails on any header that
is installed without appearing in this table.

| Header | Why it ships | Configuration |
| --- | --- | --- |
| `icgn2d1_cuda_mvp.h` | `ICGN2D1CudaMvp` | always |
| `fftcc2d_cuda.h` | `FFTCC2DCuda` | CUDA builds |
| `oc_fftcc.h` | vendored `FFTCC2D` (includes `<fftw3.h>`) | `OPENCORR_GPU_MVP_WITH_FFTW=ON` |
| `oc_icgn.h`, `oc_dic.h`, `oc_poi.h`, `oc_subset.h`, `oc_deformation.h`, `oc_point.h` | base classes and POI/subset types of those engines | always |
| `oc_image.h`, `oc_array.h`, `oc_gradient.h`, `oc_interpolation.h`, `oc_cubic_bspline.h` | image, gradient and interpolation types the engines take | always |
| `opencorr_gpu_mvp_version.h` | generated build identity | always |

One upstream caveat: `oc_point.h` calls `sqrt` in `Point2D::vectorNorm` and
`Point3D::vectorNorm` but includes only `<iostream>`, so including it as the
very first header of a translation unit does not compile. Include `<cmath>`
(or any other engine header) before it. The vendored sources are not patched
for this, so the baseline in NOTICE.md section 2 stays intact; the checker
knows about this one header and still compiles everything else in isolation.

Deliberately **not** installed: `src/cli_output_integrity.h` (it needs
`third_party/nlohmann` and the internal `opencorr_cli_io` target, neither of
which is installed, and the CLI I/O layer is not a supported API),
`src/opencorr.h`, and the vendored headers only that umbrella header pulls in
(`oc_feature.h`, `oc_sift.h`, `oc_nearest_neighbor.h` with `nanoflann.hpp`,
`oc_io.h`, `oc_strain.h`, `oc_stereovision.h`, `oc_calibration.h`,
`oc_epipolar_search.h`, `oc_feature_affine.h`, `oc_iclm.h`, `oc_nr.h`).
Embedding one of those classes means building against this repository's
`src/` tree, not against the install prefix.

## Contracts

| Item | Contract |
| --- | --- |
| Coordinates / displacement | POI x/y and u/v are pixels. FFTCC extraction uses integer-truncated coordinates; callers supply finite, int-representable values. |
| Subset dimensions | FFTCC radius r means an even 2r by 2r FFT window; ICGN uses 2r+1. CLI radius is distinct from pipeline subset size. |
| FFTCC geometry | Only equal positive x/y radii, up to 23170 (signed indexing bound), are accepted. The bound is not a memory-capacity promise. Both rectangular orientations throw `std::invalid_argument` before FFT buffer allocation/launches. Rectangular transforms remain unsupported, and the packed GPU geometry behind that rejection is itself square-only (see below). |
| CPU geometry lifetime | `FFTCC2D` allocation is fixed at construction. Changing inherited/public radius fields is detected before prepare/compute; construct a new engine instead. |
| Images | Images must outlive all calls; setters notify content changes, even at the same address. Direct writes without a setter are unsupported. Reference and target must have the same width and height: the CLI refuses a mismatched frame, and `FFTCC2DCuda::compute` (both overloads) throws `std::invalid_argument` naming both sizes before any kernel runs, leaving the queue untouched. Its target extraction is bounds-checked against the reference size, so a smaller target would otherwise be read out of bounds. The vendored CPU `FFTCC2D` keeps upstream's behaviour and does not check. |
| Reference layout | `prepareRef(queue)` retains the integer coordinate layout. A reference setter invalidates spectra. Matching unindexed or valid indexed calls lazily rebuild it. A mismatched unindexed call stays uncached without rebuilding the retained layout. |
| Indexed FFTCC | One index per POI; indices must be in range and identify the same integer coordinates. Count/coordinate errors throw `invalid_argument`, missing layout `logic_error`, range errors `out_of_range`. No silent fallback. |
| Thread ownership | Do not call one solver concurrently from multiple host threads. Different CUDA solvers may be used from different host threads at the same time: the process-wide device-buffer registries behind `FFTCC2DCuda` and `ICGN2D1CudaMvp` are mutex-protected (lookup, insertion and removal only; the lock is never held across a CUDA call or a kernel). An ICGN solver and the solvers that borrow its target (`adoptPreparedTar`) or gather from it (`computeGathered`) share device buffers and count as one solver for this rule: keep such a group on one thread at a time. `smoke/registry_thread_probe.cpp` (ctest `registry_threads`) holds this. CPU FFTCC pool capacity is set by its constructor; keep the OpenMP worker count within that capacity. CPU ICGN wrapper rebuilds its pool when OpenMP maximum changes. |
| CPU target adoption | The recipient owns its interpolation table. Repeated adoption of one successful donor prepareTar generation reuses that table. New donor preparation, recipient target preparation, or a worker-pool rebuild invalidates reuse. Process-unique generation IDs avoid address-reuse ambiguity. |
| Gathered ICGN compute | `computeGathered` fills the recipient's per-POI precompute slots with the donor's records for the queued POIs. After a call that returned true the recipient holds no prepared POI layout: `hasPreparedReference` is false, the next `compute()` or `prepareRef(queue)` recomputes the layout it is given, and until then the recipient is refused as a gather donor (the caller falls back to `compute()`). A call that returned false wrote no slot. |
| Invalid results | Negative ZNCC is status/failure state; u/v may be seeds. Filter finite values and zncc >= 0 for measurements. `-3` means terminated before a result existed, `-4` non-convergence, `-5` divergence or a non-finite value; see [`docs/output-format.md`](output-format.md). |
| Out-of-bounds target subsets | A warped target subset whose samples are not all inside the interpolatable range of the target image (`gx >= 1 && gy >= 1 && gx < width - 2 && gy < height - 2`) rejects its POI with `zncc = -3` at that iteration, before the subset mean, norm, ZNSSD cost or Newton step are formed. The deformation fields keep the incoming guess and the iteration count is zero. The CPU ICGN backend and both CUDA iteration kernels evaluate the identical predicate at the identical point, so the two backends reject the same subsets and a POI is never scored against fabricated intensities. A NaN sample coordinate is not covered by this predicate: it means the iterate went non-finite and every backend reports `-5`. Upstream OpenCorr introduced the rule in `baa3de6` but detects it by scanning the reconstructed subset for a negative value, which also fires on in-bounds B-spline undershoot; this project tests the coordinate instead (NOTICE.md section 2.5). |
| `-3` is terminal | A POI already carrying `zncc = -3` is excluded from the frame `bad_ratio`, is not a POI-level fallback candidate, and a fallback pass that ends in `-3` does not replace the primary result. The one exception is the CLI's whole-frame fallback, which replaces a `-3` left by the seeded attempt when its from-scratch pass resolves that POI as good (finite `zncc >= 0.8`); see [`docs/output-format.md`](output-format.md#whole-frame-fallback). `-3` also has a CUDA-only cause (the reference-side precompute gate, which includes a singular-Hessian test the CPU backend does not have), so the two backends' `-3` totals are not comparable. See [`docs/output-format.md`](output-format.md). |
| Exceptions | Host API guards throw standard exceptions as specified above. Older vendored CPU methods still include string exceptions (e.g. missing reference in prepareRef); applications using those interfaces must handle both. Errors in an OpenMP worker are not a supported exception transport. |

Rejecting rectangles at the constructor is a guard over an implementation that
is square-only in two independent places, so the guard is load-bearing and not
merely a scope decision. `ensureChunkScratch` in `src/fftcc2d_cuda.cu` sizes the
R2C spectra as `freq_len = subset_width * (subset_height / 2 + 1)`, while the
cuFFT plan in the same file is built with `n = {height, width}` and therefore
needs `subset_height * (subset_width / 2 + 1)`. The two expressions agree only
when width equals height; a rectangular subset would under-allocate `d_freq_a`,
`d_freq_b` and `d_ref_freq` and let cuFFT write past the end of a device buffer.
Separately, upstream FFTW calls `fftwf_plan_dft_r2c_2d(width, height, ...)`,
which reinterprets the packed subset with swapped dimensions, so the non-square
behavior of the CUDA class has never been cross-validated against the CPU path.
Opening a rectangular path means correcting `freq_len` together with the
`d_ref_freq` stride and the per-slot offsets that index it, not just relaxing
the guard.

The CPU adoption change does not share interpolation memory or remove its first
copy per target. It therefore does not lower the two-table peak memory of the
CPU repair path. Profile `prepare_tar_calls` counts actual CPU preparations;
CUDA profiling and kernels are unchanged. This is an observable stderr change in
CPU-only builds: the counter was never incremented there before, so the
end-of-run `prepareTar calls:` line reported 0 regardless of the work actually
done. It now reports the real count, the same value a CUDA build reports for the
same job (measured: 2 calls for a two-frame run in both builds). Frame bytes,
CUDA builds and the production binary are unaffected.

Changing CPU-only wrapper/FFTCC class layout requires recompiling C++ clients
against the matching headers/library. Do not mix object files from old builds.
The CLI still uses square subsets and one fixed reference layout. Existing
square-output SHA-256 gates must remain unchanged; rejecting rectangles is not
a claim that a corrected rectangular numerical implementation has shipped.
