# Cache-validity fix: validation evidence index (2026-09-16)

Index of the evidence behind the two cache-validity commits on
`codex/cache-validity-20260916`. Every number below comes from a logged run; the
"Where the evidence lives" section points at the log that produced it.

## The bug

`FFTCC2DCuda::setTarImage()` decided whether the cached target image was still
valid by comparing the incoming `Image2D` **address** with the cached pointer.
A multi-frame run reuses one `Image2D` object and reloads pixels into it, so the
address never changes and the upload was skipped: from the **second frame on,
FFTCC correlated every frame against the first frame's target image**.
`ICGN2D1CudaMvp` carried the same address-keyed guard, so the defect was present
in both CUDA classes at library level. The CLI masked the ICGN half because it
calls `prepareTar()` unconditionally on every frame; the FFTCC half was live.
（中文注：缓存按对象地址判新旧，多帧复用同一个 `Image2D` 时缓存永不失效。）

## What the two commits did

### `dfd246e` — Fix CUDA image cache invalidation and validate parity per frame

- Image setters became **content-change notifications** rather than
  pointer-identity checks, in both `FFTCC2DCuda` and `ICGN2D1CudaMvp`;
  `compute()` prepares invalidated CUDA state lazily.
- `setRefImage()` invalidates the uploaded reference and its derived caches;
  `setImages()` invalidates reference and target state.
- `smoke/check_cpu_gpu_parity.py` gained **per-frame** acceptance (shared-valid
  floor, valid-retention, validity-disagreement, median/p99 bounds) instead of
  sequence-pooled statistics only.
- New regression assets: `smoke/run_cache_regression.py` (CLI-level, synthetic
  `0 -> 5 -> 0` and `5 -> 0 -> 5` shift sequences with known truth) and
  `smoke/cache_lifecycle_probe.cpp` (library-level, exercises both CUDA classes
  through setters plus lazy `compute()`).

### `418cfa1` — Rebuild FFTCC reference caches lazily and enforce cache contracts

- Both FFTCC backends **retain** the POI layout established by
  `prepareRef(pois)`; after `setRefImage()` the next vector `compute()` rebuilds
  the reference spectra once and later calls reuse them.
- `compute(queue, indices)` now requires a previously prepared layout and throws
  on wrong index counts, out-of-range indices, or mismatched reference
  coordinates; it never silently ignores indices. Reordered and repeated slots
  are supported.
- The vendored CPU `FFTCC2D` gained matching reference/target setters, including
  invalidation through the base `DIC::setImages` (see the ABI note in
  `README.md`). The upstream CPU checkout is untouched.
- `processFrame()` gained a Release-active runtime precondition: both engines
  must hold the same prepared reference.
- `--max-threshold` (default 0.5 px) was added to `check_cpu_gpu_parity.py` so a
  single outlier that never reaches p99 is still rejected; a missing CSV column
  is reported as a failed frame instead of an unhandled `TypeError`.
- `OPENCORR_GPU_MVP_BUILD_TESTS=ON` wires `parity_unit` and `cache_lifecycle`
  into CTest; the dead `runFrame` helper was deleted.

## Evidence summary

### The tests are genuinely red on the pre-fix commit

| Check | On `da3bd11` (production) | After the fix |
|---|---|---|
| `smoke/run_cache_regression.py`, 6 CLI sequences (`seq050`/`seq505` x noskip/default/no-preload) | **6/6 FAIL** — frame 1 reports `valid=0 correct=0`; runner exit code **1** | all 6 pass, 420/416/420 correct POIs, exit code **0** |
| `smoke/cache_lifecycle_probe` compiled against the `da3bd11` core | **34 PASS / 14 FAIL**, `cache lifecycle failures=14`; exit code 1 | `dfd246e`: 48 PASS / 0 FAIL. `418cfa1`: **65 PASS** (default build), **83 PASS** (with `OPENCORR_GPU_MVP_WITH_FFTW=ON`), 0 FAIL |

All 14 probe failures on `da3bd11` are `*-frame-1` (the second frame) cases with
`mismatches=128 fresh_valid=128/128`, and they span **both** families, e.g.

```
FAIL fft-target-0-indexed-0-frame-1     mismatches=128 fresh_valid=128/128
FAIL icgn-target-0-setImages-0-frame-1  mismatches=128 fresh_valid=128/128
```

which is the direct evidence that `ICGN2D1CudaMvp` carried the same defect at
library level even though the CLI never exposed it.

The public headers of `fftcc2d_cuda.h` / `icgn2d1_cuda_mvp.h` differ between
`da3bd11` and `dfd246e` only in comments, so the probe compiles unmodified
against the old core — the comparison is like-for-like.

### Real data: acceptance-C `gate2` (a) protocol

CPU side is the read-only upstream `opencorr/build/opencorr_cli`; GPU side is the
candidate build. `event10` frames `000002`–`000011` (10 frames), subset radius 7,
grid step 1, `--threads 10 --no-skip-fftcc --no-poi-fallback`, CSV output, joined
on common coordinates.

| GPU binary | `--max-threshold 0.5` | pooled shared-valid | max \|du\| | max \|dv\| |
|---|---|---|---|---|
| `418cfa1` candidate | **PASS** | 1,265,160 | 1.01e-2 px | 7.46e-3 px |
| production `da3bd11` | **FAIL** | 1,264,441 | 8.60e-2 px | **5.2364 px** |

The four points that break the production binary all sit in the dark band around
`(822-823, 46-47)`; the 5.236 px figure reproduces the 2026-09-03 `gate2.log`
record digit for digit, which confirms the protocol was replayed faithfully
rather than changed.

**Scope statement (do not over-read this):** the real-data parity result covers
event10, 10 frames, r7/step1, `--no-skip-fftcc --no-poi-fallback` only. It does
not license the claim that a 0.5 px maximum is safe for every real dataset,
radius, or protocol. See the wording already in `smoke/README.md`.

### Production anchors on `418cfa1`

Protocol (identical to `results/acceptance-c-phase-2026-09-03/logs/gate3.sh` and
`gpu_batch.sh`):

```
REF     = /data/lab-jobs/bimaterial/config1-bimaterial/30kN/photos/event10/10_C001H001S0001000001.jpg
TL60    = lab-agent/perf-p1-p2-p3-2026-07-03/scripts/tar-list-60.txt
TL800   = lab-agent/release-gate-poi-fallback-20260704/scripts/event10_targets_frame000002-plus.txt
ARGS    = 7 1 --gpu-device 1 --threads 10 --conv-threshold 0.001 --skip-fftcc --output-format raw
<bin> --multi-frame --ref $REF --tar-list <list> --output-dir <out> $ARGS [--no-poi-fallback]
tree hash = (cd <out> && sha256sum frame_*.bin | sha256sum)
```

> **Superseded on 2026-09-17.** The table below is the 2026-09-16 archive and
> records the anchors *of that day's numerics*. Stage 2E gave the ICGN kernels
> the partially-out-of-bounds rejection rule, which moved all four; the current
> values live in `smoke/run_production_anchors.sh` (the rule itself is described
> in [`output-format.md`](output-format.md)). Everything this page says about the
> cache fix still holds — it is the anchor digests that have moved on.

| Run | Tree SHA-256 (history: 2026-09-16 numerics) | Current (2026-09-17 onwards) |
|---|---|---|
| 800 frames, default (default build) | `2412d5df4c827110a2f3b207b5c962c24d89c676d4eead489acea01be1b18910` | `d09619465548c6250775a79c9fe67e5dbb78663e74740dd698a5f761cbc836eb` |
| 60 frames, default (default build) | `9c79044b0e7eae74093bd409dc99e70d6d5961f6fc02887b53a3c03916a702a3` | `26ddb93ee552919a8f4e01c547d7463b7517e84757fb8dabad3b3b07b1adf4d0` |
| 60 frames, `--no-poi-fallback` | `f12e0cb991692e4c31962a501c5774323d1b31d11ff4101244f48c0b85febf87` | `9b22c4bc39f07e882e0008c727b68447bce6410261c2ec1984e7359d62e0df12` |
| 800 frames, default, `--fftcc cpu` (FFTW build) | `ff4b2e57d9dbdf93c7d44b7c020a4498555dfac30d0f8463e5fc5846fb38d45d` | `f90740896e0da3aa08fb1fd312d83b005dc6b964c6379eb87524ccba651dab46` |

The 60-frame default anchor **moved** relative to the pre-fix archive
(`a21bb7eb719de2faf2cae6912c930e8e8ede6ce6a048b32fb4716fa6d993e51d`), because the
fix changes which target image FFTCC correlates against.

`f12e0cb9...` is **unchanged** by *the cache fix this page is about*, and that
is expected rather than suspicious (it did move later, on 2026-09-17, for an
unrelated reason — see the note above the table). Under `--skip-fftcc`, FFTCC runs in full only on the first frame;
adding `--no-poi-fallback` removes the POI-level fallback that is the only other
caller, so from frame 2 on FFTCC is never invoked and the stale-target path is
unreachable. This variant is therefore a useful **negative control**: it shows
the fix changed nothing outside the FFTCC-per-frame path.

Dual-GPU (`--dual-gpu`, protocol from
`results/acceptance-c-phase-2026-09-03/logs/gpu_batch.sh`) reproduces the 60-frame
result byte for byte against the single-GPU run.

For the 60-frame default anchor the per-POI delta against the pre-fix output is:
10,261,440 rows compared, 69,387 rows differ (0.676%), of which 99.88% have at
least one side invalid; only **82** POIs are valid on both sides with different
numbers, and the net valid count changes by **+12**.

## Where the evidence lives

The maintainer's private project archive, which is not part of this
repository (reports, scripts, JSON summaries):

- `results/cache-validity-review-2026-09-16/` — `review-A-code.md` (read-only
  code review of `dfd246e`), `review-B-verify.md` (independent rebuild and
  measurement; sections 2 and 8 hold the red-on-`da3bd11` numbers above),
  `review-C-parity-ab.md` (v9 parity smoke and `--fftcc cpu` vs `gpu` A/B),
  `review-A2-followup.md` (read-only review of `418cfa1`),
  `review-B2-followup.md` (independent rebuild of `418cfa1`: the four anchors and
  the real-data parity table above).
- `results/cache-followup-2026-09-16/` — the `418cfa1` patch, build/production
  provenance, CTest and pipeline logs, `check_fixed_reference.py` and
  `fixed-reference-summary.json`.

Remote run trees on `simulacrum` (none of the three existed any more when
checked on 2026-10-03):

- `/tmp/opencorr-cache-validity-verify-20260916/` — `dfd246e` verification
  (numbered `01`-`08` for the review-B sections, `11`-`12` for review-C).
- `/tmp/opencorr-followup-verify-20260916/` — `418cfa1` verification, including
  the `gate2a/` real-data parity JSON reports.
- `/tmp/opencorr-step3-verify-20260916/` — the pre-commit re-verification for
  this document (README.md there indexes the commands).

Note on the remote trees: the bulk `frame_*.bin` raw outputs have been deleted
after their tree hashes were taken. Logs, `*.time` files, JSON reports and the
parity CSVs remain, so the hashes above can be traced to a log but the raw
outputs must be regenerated to be re-hashed.

## What is not claimed here

> As of 2026-09-16. Both commits were merged later and are in the production
> build of 2026-09-17, and CPU-only multi-frame runs work: the `cpu-only`
> preset's `cpu_sequence_t1`/`cpu_sequence_t4` tests run them (passing on
> 2026-10-03).

- No merge to `master` and no replacement of the production binary
  (`opencorr-gpu-mvp-build/opencorr_cli_cuda_mvp`, `c755d663...72fec`) has been
  performed; `master` is still `da3bd11`.
- No formal performance re-timing was done for these commits.
- A CPU-only (`OPENCORR_GPU_MVP_ENABLE_CUDA=OFF`) build still does not handle
  multi-frame runs correctly; that is a known open item, not something these
  commits claim to fix.
