# FastICGN v1.1.0 Release Record (2026-07-10)

Unlike `release-v1.0.0.md` (a how-to checklist), this is the **record of what
was actually executed**, so the release evidence survives (`v1.0.0`'s scratch
build in `/tmp` evaporated and its smoke runs could no longer be verified).

## Tag

- `v1.1.0` (annotated) at `fc34647` = HEAD at release time; tree clean.
- Since `v1.0.0` (`9ebdc6f`), 12 commits: raw+manifest output, ICGN
  conv-threshold default 0.05→0.001 (CPU-aligned), `raw_to_csv.py`, high-bit
  input (`--input-depth/--input-align/--input-normalize`), spaced-path
  tar-list, pipeline scripts into git (`pipeline/`), pipeline `--conv-threshold`
  always-explicit passthrough + `--input-*` passthrough, high-bit smoke × 2.

## Release build (durable, NOT in /tmp)

- Build tree: `/home/chonglang/opencorr-gpu-mvp-build-20260710/` — keep; this
  is the promote source. Logs: `~/opencorr-gpu-mvp-build-20260710-{cmake,make}.log`.
- Fresh configure from scratch (version header regenerated), Release,
  CUDA 13.2. Provenance verified:
  `--version` → `git_hash=fc3464779849`, `build_date=2026-07-10T03:00:47Z`.

## Smoke evidence (run against this build, 2026-07-10)

- `smoke/run_high_bit_input.sh` — passed (8-bit legacy byte-identity, 12/16-bit
  values, preload/sync manifest `input_image` blocks).
- `smoke/run_high_bit_input_extra.sh` — 15/15 assertions passed: `auto`
  inference anchors incl. documented dark-left-shifted misjudgment,
  `--input-normalize to-8/to-16/unit` output ranges, spaced-path tar-list
  end-to-end (space- and tab-separated fields, 26-row CSVs).

## Promote to production (`~/opencorr-gpu-mvp-build/`)

- Pre-check: default `opencorr_cli_cuda_mvp` md5 == `_v10_uyvy` backup
  (`975be6cc…`) → safe to overwrite.
- Copied: new binary → `opencorr_cli_cuda_mvp_v11_hibit` **and** over default
  `opencorr_cli_cuda_mvp`; `image_input_depth_probe` alongside.
- `dic_pipeline.py` / `run_dic.sh` / `DIC_PIPELINE_GUIDE.md` in the build dir
  replaced by **symlinks into `…-mvp-src/pipeline/`** (single source of truth;
  the stale May-3 GUIDE moved to `~/.trash-serf-20260710/`).

## Post-promote end-to-end check (production entry point)

- `run_dic.sh --image-dir <synthetic 128×128 PNG event> --output-dir … --line-y 50,70 --cal 1.0`
  - 8-bit event: banner shows `conv-threshold: 0.001（显式透传）`; 440 COD rows
    (110 POI × 4 frames), identical frames → `max|cod_um| = 0` exactly.
  - 12-bit left-shifted event (`--input-depth 12 --input-align left`): banner
    shows `输入位深: depth=12 align=left`; same 440 rows, `max|cod| = 0`.

## Known state after release

- Production default binary = v11_hibit (`fc34647`); previous default preserved
  as `opencorr_cli_cuda_mvp_v10_uyvy` (conv default 0.05 — pass
  `--conv-threshold 0.001` explicitly if ever used directly).
- Pipeline pins `--conv-threshold` explicitly on every run, so convergence
  semantics no longer depend on binary vintage.
- Superseded build/src trees moved to `~/.trash-serf-20260710/` (nothing
  deleted).
