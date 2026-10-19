# Example data

`smooth-30kN-0725B-B2L2/` is one real laboratory rupture: the reference frame
and target frames 443 to 564 (122 frames, 20 µs apart) of a stick-slip event on
a frictional interface, 1024 x 112 px 8-bit JPEG from a Photron SA1.1 at
50 000 fps, 7.7 MB in all. Its `MANIFEST.md` records the acquisition
parameters, the calibration (0.635527061 mm/px), the interface row (56) and
measurement lines (48 / 64), the frame-selection rule, the expected results and
the SHA-256 of every file; `LICENSE` is CC BY 4.0 (copyright Chonglang Wang,
attribution line in the manifest). From the repository root, after a CUDA
build:

```bash
pipeline/run_dic.sh --image-dir examples/smooth-30kN-0725B-B2L2 --output-dir out/B2L2 \
  --subset 15 --step 1 --interface-y 56 --gauge 16 --cal 0.635527061 --roi-cols 106,892 \
  --gpu-bin "$PWD/build/cuda-release/opencorr_cli_cuda_mvp" \
  --strict --json-summary out/B2L2/summary.json
```

Expected: exit 0, `verdict` `ok`, 122 frames x 787 points in
`out/B2L2/cod_result.csv`, `valid_ratio` 0.959, `zncc_median` 0.995; the
spatial median of the slip per frame stays near 1 µm until frame 462, rises
from frames 463 to 464 at up to 24 µm per frame (110 µm at frame 471) and
levels off near 260 µm (261 µm at frame 564). README step 5 under Quickstart shows how to read
the per-frame medians and how to run one full-field frame with the binary
alone.
