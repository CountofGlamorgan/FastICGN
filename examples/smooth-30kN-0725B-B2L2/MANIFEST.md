# FastICGN example sequence — smooth-30kN-0725B-B2L2 (reference + 122 target frames)

Released 2026-10-19 under CC BY 4.0 (section 5). Pixels are untouched: every
JPEG is a byte-identical copy of the camera file (section 6).

## 1. What this is

A real high-speed speckle-image sequence of one laboratory stick-slip (rupture)
event on a polished ("smooth") frictional interface, cut down to the reference
frame plus the 122 consecutive target frames that contain the rupture: the slip
across the interface rises from about 1 µm to about 260 µm inside this window
(the event's final plateau, 260 frames later, is 278 µm).

| item | value |
|---|---|
| event id (laboratory naming) | `smooth-30kN-0725B-B2L2` = surface `smooth`, normal load 30 kN (3 MPa), session `0725B`, event `B2L2` (round B2, loading, 2nd event) |
| full event (not distributed) | 801 JPEG frames `B2L2_C001H001S0001000001.jpg` … `000801.jpg`, 50 083 964 bytes |
| this sample | frame `000001` (reference) + frames `000443` … `000564` (122 targets) = 123 files, 7 698 361 bytes (7.34 MiB) |
| files beside the frames | `MANIFEST.md` (this file), `LICENSE` (CC BY 4.0 legal code) |

## 2. Acquisition parameters (from the camera header `B2L2_C001H001S0001.cihx` and the laboratory data guide)

| parameter | value | source |
|---|---|---|
| camera | Photron FASTCAM SA1.1 model 675K-M1 | `.cihx` deviceName |
| frame rate | 50 000 fps (20 µs per frame) | `.cihx` recordRate |
| shutter | 1/593 000 s (1 685 ns) | `.cihx` shutterSpeed / shutterSpeedNsec |
| image size | 1024 x 112 px | `.cihx` resolution, verified on the files |
| pixel format | 8-bit grey stored as JPEG; the files decode as 3-channel RGB with all three channels identical (maximum channel difference 0) | `.cihx` colorInfo bit=8; verified with Pillow |
| recording date/time | 2023-07-25 15:17:52 | `.cihx` |
| recorded frames | 801 (startFrame −400, i.e. frame 401 is the trigger frame); frame 1 = 8.0 ms before the trigger | `.cihx` frameInfo |
| spatial calibration | 0.635527061 mm/px (July-2023 calibration group) | laboratory calibration table, 2026-07-27 |
| interface row | `interface_y = 56` px (image row, increasing downward) | same |
| measurement lines | L0 = row 48, L1 = row 64 (`--interface-y 56 --gauge 16`: interface ± 8 px = ± 5.08 mm) | same |
| physical mid-point | `midpoint_x = 499` px; `x_centered_mm = (x_px − 499) × 0.635527061` | same |
| analysis ROI | columns 106 … 892 inclusive (`--roi-cols 106,892`, ± 250 mm about the mid-point); columns below about 30 and above about 980 are outside the specimen (blank) | same |
| image content | speckle pattern over the full field; no camera overlay or text bar (no flat rows); the interface is the dark horizontal line at row 56 | verified on frame 1 |
| saturation | 0.20 % of reference pixels at 255 (0.03 % inside the L0 measurement band, 0 % in L1): the pipeline grades `input_8bit_max_255` as *info* | pipeline summary, 2026-10-19 |

## 3. Frame-selection rule

Computed from the laboratory's 2026-07-27 production batch result of this
event (fill-boundary COD, subset 15, step 1, lines 48 / 64, ROI 106–892), using
the spatial median of `cod_um` over the ROI per frame:

* baseline (frames 2–51): 1.1 ± 1.0 µm
* onset = first frame whose median exceeds the baseline by 5 µm and stays above for 5 frames: **frame 463**
* fastest rise: 24.5 µm per frame at frame 466; 50 % of the plateau at frame 476; 90 % at 516; 95 % at 544
* final plateau (median of frames 752–801): 278.0 µm

Window = **onset − 20 … (95 %-plateau frame) + 20 = frames 443 … 564** (122
target frames): twenty quiet frames before the rupture, the whole rise, and
twenty frames after the median has reached 95 % of the final plateau.

The reference frame is the camera's first frame (frame 1), as in the
laboratory convention and in the 2026-07-27 batch, so frame numbers and results
are directly comparable to the batch. File names are kept as recorded by the
camera (the pipeline derives the frame number from the trailing six digits; the
gap 2–442 is simply absent and the pipeline treats 443–564 as 122 targets, see
`--dry-run`).

## 4. How to run, and what to expect

Laboratory COD post-processing through the production entry point (from the
repository root, after a CUDA build; `run_dic.sh` runs the pipeline with
`python3`, or with `FASTICGN_PYTHON` when numpy and Pillow live in a virtual
environment):

```
pipeline/run_dic.sh --image-dir examples/smooth-30kN-0725B-B2L2 --output-dir out/B2L2 \
  --subset 15 --step 1 --interface-y 56 --gauge 16 --cal 0.635527061 --roi-cols 106,892 \
  --gpu-bin "$PWD/build/cuda-release/opencorr_cli_cuda_mvp" \
  --strict --json-summary out/B2L2/summary.json
```

Measured 2026-10-19 with a CUDA build of the released tree (its `cod_result.csv`
is byte-identical to the laboratory's production binary of the same day):

| quantity | naive (default) | `--cod-method fillboundary` |
|---|---|---|
| exit code / `verdict` / `warnings` | 0 / `ok` / `[]` (`input_8bit_max_255` is an *info* entry of `warning_details`) | same |
| frames x points in `cod_result.csv` | 122 x 787 = 96 014 rows | same |
| `valid_ratio` / `zncc_median` | 0.9591 / 0.9952 | same (same displacement field) |
| spatial median of `cod_um`, frame 443 / 463 / 466 / 471 / 476 / 501 / 542 / 564 | 1.4 / 7.3 / 39.1 / 110.2 / 142.3 / 213.8 / 256.5 / **261.3** µm | 2.4 / 11.3 / 33.8 / 110.3 / 146.4 / 221.0 / 263.3 / **267.3** µm |
| median over frames 545–564 | 259.1 µm | 266.1 µm |
| `cod_stats.max_um` (frame, `abs_p95_um`) | 297.5 (564, 271.4) | 327.7 (529, 286.3; naive at the same point 274.2) |
| GPU time | about 1 s | about 1 s |

Full-field, one target frame, binary alone (positional `7` is the subset
**radius**, `2` the grid step):

```
opencorr_cli_cuda_mvp examples/smooth-30kN-0725B-B2L2/B2L2_C001H001S0001000001.jpg \
  examples/smooth-30kN-0725B-B2L2/B2L2_C001H001S0001000471.jpg 7 2 frame471.csv
```

| target frame | POI (x 9–1013, y 9–101) | valid (`zncc >= 0`) | zncc median | `-3` / `-4` / `-5` | where the invalid points are |
|---|---|---|---|---|---|
| 471 (fastest slip) | 23 641 | 22 659 (0.9585) | 0.9972 | 13 / 969 / 0 | all in the blank columns outside the specimen (x < 40 or x > 970) |
| 564 (last frame) | 23 641 | 22 568 (0.9546) | 0.9939 | 21 / 1052 / 0 | same |

## 5. Licence and attribution

This data set (the 123 image files, this manifest and `README.md` in the same
directory) is released under the **Creative Commons Attribution 4.0
International licence (CC BY 4.0)**; the legal code is in the `LICENSE` file
beside this manifest and at <https://creativecommons.org/licenses/by/4.0/>.
It is distributed beside, not as part of, the FastICGN software, which is
MPL-2.0 (`NOTICE.md`, section 10).

```
Copyright (c) 2026 Chonglang Wang
```

Attribution: "FastICGN example sequence smooth-30kN-0725B-B2L2, laboratory
stick-slip experiment of 2023-07-25, Chonglang Wang, CC BY 4.0", with a link
to the FastICGN repository. A publication that uses the sequence should also
cite the software (`CITATION.cff` in the repository root). The full campaign
(hundreds of events, several terabytes) is not distributed; contact the author.

## 6. File list and SHA-256 (123 files, 7 698 361 bytes; every hash equals that of the camera file in the laboratory archive)

| file | sha256 |
|---|---|
| `B2L2_C001H001S0001000001.jpg` | `62e2fb268d0940b291c08bb229599e23fc668d5edd12daf18e3a49a7bba7fabc` |
| `B2L2_C001H001S0001000443.jpg` | `636c06568b26bff0a963466ff5164e5d8ae68734857e8766e622d2be643c0093` |
| `B2L2_C001H001S0001000444.jpg` | `91982432515db636a2408be1b35c4b4da51667ca2710c9e9c097e4182515b027` |
| `B2L2_C001H001S0001000445.jpg` | `af3d8b32272f9934961b480f15e1c6c7359095a63cfd34aecaf786663a686b3a` |
| `B2L2_C001H001S0001000446.jpg` | `04b9b27c7373f7e97e4e9c2be31b1139e2e112db3e89f8c3cefa0bb571130108` |
| `B2L2_C001H001S0001000447.jpg` | `48f39e228bf52db6e39570e18890bad7204c7434a5536b1b115c671b5938deb1` |
| `B2L2_C001H001S0001000448.jpg` | `0169e4dd0f6bb58b4e40088a8457d31c5c513f20cfa8152305f67c4c608a7fd7` |
| `B2L2_C001H001S0001000449.jpg` | `246e47f91d3b92f6ac16d622731f912c241088457c5e65c2baeb9d48966dfcf6` |
| `B2L2_C001H001S0001000450.jpg` | `52751c47d6151baea28aa834f1a48339a47bec460d8481c76b7ad50843e21793` |
| `B2L2_C001H001S0001000451.jpg` | `380bbb0a374602cd82e56a451087334076c6d68966f83935da4b34c85e9447e1` |
| `B2L2_C001H001S0001000452.jpg` | `0d8dedf5d495b51cdf828fc825774d36e507403b2e82718e4ede1ee1da7843e8` |
| `B2L2_C001H001S0001000453.jpg` | `85d59468ea3a2de51951a71de1667317af791da452e9767fb2626ed25f610583` |
| `B2L2_C001H001S0001000454.jpg` | `1dc8cba770a06969182b07825ffc360b8cf5760e8fb1c3f875dda3752ca186ba` |
| `B2L2_C001H001S0001000455.jpg` | `522e58be0b613b5ec74257efccbefcc2407cf50a29cb3dc8c26b4de393c6bc17` |
| `B2L2_C001H001S0001000456.jpg` | `9d67393666659b32f223f3631b1d9e132fd9ec215c9e2728bca15c964960ad63` |
| `B2L2_C001H001S0001000457.jpg` | `b7cf86dc9be84128fceb9e20b4344703f4a80a0b903ac570554e131c15dfb3b8` |
| `B2L2_C001H001S0001000458.jpg` | `a44863ddddb9cd432cef91d29e8ce90d0d798785672b4c4a4dcc28f12ebba774` |
| `B2L2_C001H001S0001000459.jpg` | `113812d8adf35124ea39ca0a10a85fe1536d1f8fe731e741507e55fb1227f3da` |
| `B2L2_C001H001S0001000460.jpg` | `9043cc9eaf9ce6b5d7368235e644efd54b15a00946cf5af91c560ac9f1481ce9` |
| `B2L2_C001H001S0001000461.jpg` | `0524346873286da3ab522f71ae873dd9459e644b4163debd36004cda672c8b6f` |
| `B2L2_C001H001S0001000462.jpg` | `64b83ac6469c6a9fb91b703b66a39cd02434482d8a86baba98699e45d13bacf6` |
| `B2L2_C001H001S0001000463.jpg` | `473e6fa2b723bc3f133a047c9bb5ebd28eaf838d22852c6ad328aec3b4a4d12e` |
| `B2L2_C001H001S0001000464.jpg` | `8cecdb9fac47bcb23a41fc848a6876f63cfbbc72499f629811bc6459533b3a52` |
| `B2L2_C001H001S0001000465.jpg` | `d0ff20b3a102d0f3b3d4dc71f5257f0bae570474eed40a83106e6d3b6ab0279d` |
| `B2L2_C001H001S0001000466.jpg` | `1b25adc3de044207d4a30fe79d8a875385467c389072f0087bd94d7c2b95ac1a` |
| `B2L2_C001H001S0001000467.jpg` | `cfb5a5a91b5827e1a64d78601dda4efa0c419c9f4e2d5fb1c174356e81a256e8` |
| `B2L2_C001H001S0001000468.jpg` | `bfcce6405b82f51064389be1a294a27bfa0ab84d0027c64f95405a14b834ebc0` |
| `B2L2_C001H001S0001000469.jpg` | `91996d7fc94bb50f3d71d743f741857daf6b107b641346dc78de3f63b1797adf` |
| `B2L2_C001H001S0001000470.jpg` | `1176a149fafa3e9276865dd36977ddc1a93fb838eb0c05c10fd500092da00e6e` |
| `B2L2_C001H001S0001000471.jpg` | `4119d848f22f4d6f35bd14cc8a1ffd28769b7ee1aee25e432372f333b8cdfaa5` |
| `B2L2_C001H001S0001000472.jpg` | `580726ebd579fb00fbb504de6478289cb02b3af34f090152ed90ccb3b912d877` |
| `B2L2_C001H001S0001000473.jpg` | `4f0f0cb618b880bb10117ad07d3e744b06cc59c0c981cfded82490c7f353f024` |
| `B2L2_C001H001S0001000474.jpg` | `8bcf4988969a396c7c4023a5f343a315f8c916e6379ad91c89ec609d895977c9` |
| `B2L2_C001H001S0001000475.jpg` | `3bf2d301fc9b4279861af5f96b0a6836deffdc9fa236d8c7e4fac6295349cab8` |
| `B2L2_C001H001S0001000476.jpg` | `4cae84cb3547715918f48b4c5c5545b55e58e0f21d15eb3928fd2a062e515d6f` |
| `B2L2_C001H001S0001000477.jpg` | `6811d929a0ab1e1d3adf067cb1b1288686bf845454b05811082fa5e32c69464b` |
| `B2L2_C001H001S0001000478.jpg` | `ca109412d7c5a7c53f5e7ee447ee800ad69f130181594254e32ebada78985f52` |
| `B2L2_C001H001S0001000479.jpg` | `cca5bbbd460781ae737701311ec18a832458a1d58b99f6c39bb8fbea8e9507ea` |
| `B2L2_C001H001S0001000480.jpg` | `a031bed3b0881083f2223f1f53aedb0b96ad755827fdf94b498f8b6829fa3a4d` |
| `B2L2_C001H001S0001000481.jpg` | `f665af6aa540b191e8d146b6b9012af315ab08ed0549aec7a552551935d6fae0` |
| `B2L2_C001H001S0001000482.jpg` | `fc8efe1231cb73b66656f9c3b4e3aefe151639c0ccece4e3847c589db87cddfb` |
| `B2L2_C001H001S0001000483.jpg` | `72ce113330f6244c134302e8d7c81ad4ae44b291e49554816ffee6a500903c19` |
| `B2L2_C001H001S0001000484.jpg` | `d6749e075a58565ffa92cd8acd6aa97dd157bd14b6eb399329a66458e6b209c4` |
| `B2L2_C001H001S0001000485.jpg` | `adcf4e4e20aeadf20f90dade7cd6fcef3aa64af36c2b8c1feb843664a6842547` |
| `B2L2_C001H001S0001000486.jpg` | `b27035e1a4f44ed8167e73845b5b7a1a3bc6ccf452d6e54c615dc8f65590f7ef` |
| `B2L2_C001H001S0001000487.jpg` | `a920f985bf3b8b78079e009cd65b83f968e95f5fd1dd4126778571e00d69b675` |
| `B2L2_C001H001S0001000488.jpg` | `5339c654c75a32bfada919ff8a595e3927b6858a96c05201ffcf926f016d3b6a` |
| `B2L2_C001H001S0001000489.jpg` | `eee5a62d4542d50c475a3da7edddf8f60abcd5734edd51dc27ff1b66a77d7c22` |
| `B2L2_C001H001S0001000490.jpg` | `50e2e2d3ea6dcf0a5c8e5ada9721eed1040129d54f4bbed5d3a80fc58fb8bce3` |
| `B2L2_C001H001S0001000491.jpg` | `a259fce719c33e662dd8cab7b0cf12dec0868d8149ffdaa009a371560672d64b` |
| `B2L2_C001H001S0001000492.jpg` | `0443512b43a3e5a6d1feea1dfe4fae1bb938d24ded63f61817b23c7023062973` |
| `B2L2_C001H001S0001000493.jpg` | `c8249d24aca600dd601fd4d03f35db22e6f60720441158e197a3f867b4777e34` |
| `B2L2_C001H001S0001000494.jpg` | `a489ff116f66d94b4fa82e6bb2fbe0d6997b7d70932774377a3374606e96968f` |
| `B2L2_C001H001S0001000495.jpg` | `5264bae50b09ba1f269616a139f19643808f503c5f042b8bd1bb55e9568aec0c` |
| `B2L2_C001H001S0001000496.jpg` | `f6b6ddf8b4d857fb958b38f7c1f9bcc75cfc9f86b3ab2040fd6be71bc05f7009` |
| `B2L2_C001H001S0001000497.jpg` | `aa5ed5f60106e33a130f16b49fcfc63ea85b8cc5abe69d353a56da988f4eebac` |
| `B2L2_C001H001S0001000498.jpg` | `b9b0420983f7ae749b66acb38ca1cf2dad2b2f58cf2777f0e47563289cf31242` |
| `B2L2_C001H001S0001000499.jpg` | `eec63c62acbb6bbd1077e398c932f9d950642769be45fd665dda8824daa1af64` |
| `B2L2_C001H001S0001000500.jpg` | `b1a00292f2d82223912b4fb43eb9ab20ce5f669e9f4e50f9fc7dcebd96d4c53c` |
| `B2L2_C001H001S0001000501.jpg` | `1f741f32c88b3d473979b0cf47410b99c125b77292e18d1e19418ddb161d779c` |
| `B2L2_C001H001S0001000502.jpg` | `13b4480aae6080ec30868901dc0145057422cb91c18e6bb29e57d2b74a4a721e` |
| `B2L2_C001H001S0001000503.jpg` | `a8d50027089f11fbeb916cee0218fa264f0b08d93057cc54d7fdbab177f71e5a` |
| `B2L2_C001H001S0001000504.jpg` | `24a8c5867c896c38b61a2f875bfcc27aa04a3f5d4799a81bc270446fb861dfe4` |
| `B2L2_C001H001S0001000505.jpg` | `3f93aa2d050ce6d18af647a2f371f97755ea17591da39f638535e723e1812b33` |
| `B2L2_C001H001S0001000506.jpg` | `89849cd1b003b7778322f31cb11b3c50ce9ee17d0cc71c455156a37bd37c7fd0` |
| `B2L2_C001H001S0001000507.jpg` | `651c6df26de79ae27248839524391c4bfcb4ac16300099ed4c4ee200ad8219db` |
| `B2L2_C001H001S0001000508.jpg` | `3827c367c6735b9dcec8683fa7b89c8670ceaa30fedfdceaccc9c8777d329bfc` |
| `B2L2_C001H001S0001000509.jpg` | `2d9782543f862b569baa28b891b9968e8bd8f3a9c403f01239ea01564a846c4e` |
| `B2L2_C001H001S0001000510.jpg` | `7179828223a9355a98ff9e7ae0233b242af027b546273df462336df4d1e0452a` |
| `B2L2_C001H001S0001000511.jpg` | `ddccfe547022ea09df863fd885c2210af7adb7b152f9cf918fdae54480aea00d` |
| `B2L2_C001H001S0001000512.jpg` | `3036fe17d41ddd126a3260f7955cbec6d322cb5dc172839efd039114746ca104` |
| `B2L2_C001H001S0001000513.jpg` | `6f1705f734dbbaefe0a19a03a88503b69d2e14753d773322c3c223ff15fb6479` |
| `B2L2_C001H001S0001000514.jpg` | `8995ada6483106adfc13dd4be8b479687c99fda1a55b843f6ef87ded40fea985` |
| `B2L2_C001H001S0001000515.jpg` | `e51fb70315c4f9e537b6e2f2a3a7dce07f12e88a605ad0213c2e1b6738095edd` |
| `B2L2_C001H001S0001000516.jpg` | `075531d84b5d1dd289293c02074836c6da0046701f00ebffc316342ada12c74a` |
| `B2L2_C001H001S0001000517.jpg` | `486cb29107eef4b6bb3ed774729bbb89c27fe8e086171b7ce3fe586110f68afa` |
| `B2L2_C001H001S0001000518.jpg` | `3cd9d0dc37c18c195cf5c7c2516e6a56b018409e896b7cf364dd66c340bd5e8f` |
| `B2L2_C001H001S0001000519.jpg` | `fa57aed3a62a15b68b9cd574c18aeb598d15b8b2021bc3866fb64b35124db723` |
| `B2L2_C001H001S0001000520.jpg` | `56acd1d35f9ec12c03c682419e8fd48caabbf0fe4af4b954784bb1555b5712ab` |
| `B2L2_C001H001S0001000521.jpg` | `bdd3806ae3f056d5aa9de49bd9aeb9c7cc83485a7f046fa78ad96879ace8c4e7` |
| `B2L2_C001H001S0001000522.jpg` | `ed5e4e2c384931f9f7fad7911e861422ff53a0a38493d1fdd3d294068ec0d0d0` |
| `B2L2_C001H001S0001000523.jpg` | `5b391097baf75ecd7fb84b898188844677d4fac1ea066048078f87fa30079c85` |
| `B2L2_C001H001S0001000524.jpg` | `bb529acaa2d5deb68a4c341841e49d24371705bf350958f63c0ea8de63063dda` |
| `B2L2_C001H001S0001000525.jpg` | `54c4be770f51c2900427dc4ab5e53e3e828bb28e56eb8c48490213ebdfe4e2bb` |
| `B2L2_C001H001S0001000526.jpg` | `161d67a06f0fe0b10ae2f609d780c42a31d8232527e863797439a2c0df70f00b` |
| `B2L2_C001H001S0001000527.jpg` | `ba0ce9f69c1eb4bc8976b9de9c70545d901a9b0ea7d6c4fb6d95a6c8dbf9f8a7` |
| `B2L2_C001H001S0001000528.jpg` | `e2af637dd36def0df11c264874738d8941e5f211fcf4c62169f357667e72b0ba` |
| `B2L2_C001H001S0001000529.jpg` | `846f830e11d40ce3e0ba0042af5277f2d4a45733f62206ac7e894ee7670a9f4f` |
| `B2L2_C001H001S0001000530.jpg` | `66220251c527fa645081b074f736747720fb36ab49db5c132fb9050c0e894d84` |
| `B2L2_C001H001S0001000531.jpg` | `7c148275945c41eb8f12ff17931dfd99d14b048977ee6fe5e89d16f489578e21` |
| `B2L2_C001H001S0001000532.jpg` | `8a0c0cacadb4a02ba66fceaae7346e1e53a0242fa2c6f27f1ec78bb50952e599` |
| `B2L2_C001H001S0001000533.jpg` | `a76db7810a1916657d401d8e54183db868f2620a92ed57fc28bd64f3f289dc8b` |
| `B2L2_C001H001S0001000534.jpg` | `2620b412ecedebaa6f660d6002f7b1fe9290d582192e9e5888d7ea1b15508026` |
| `B2L2_C001H001S0001000535.jpg` | `e2723e7c4a54cce28f6eb1776b314faadfc0ff975c0237fec5ce81dfc50a8646` |
| `B2L2_C001H001S0001000536.jpg` | `2023a8fbb0f1133d81e203de17b497e009d632969d6535c1cd997219c12361e9` |
| `B2L2_C001H001S0001000537.jpg` | `ea1e4a956c1140cdf04b338b1ab6447609f834f1ff6eecfe47246cd7d2bb38e9` |
| `B2L2_C001H001S0001000538.jpg` | `a0bd74321d9fa27d505482b2f01237e92b6de871c7ad5553d2eacfa47f8e1f47` |
| `B2L2_C001H001S0001000539.jpg` | `88092b0200f830747adfbb77375751238f8a539e7826273330373503e68ecfb3` |
| `B2L2_C001H001S0001000540.jpg` | `27f26bb87dfd5e41a0cccc244461c496a7c88049d9ab297c8ce7be193013ddec` |
| `B2L2_C001H001S0001000541.jpg` | `03897f09a150fd525bb665f8aab6c55ea48362414e084a57785dd90a1dd38da5` |
| `B2L2_C001H001S0001000542.jpg` | `6d96437fc5f5a0db57dfad5aa1353690f23f8a861776477606045b505eaef52d` |
| `B2L2_C001H001S0001000543.jpg` | `1309af5deb7a25879193914b5a9cb5d16998646ed99b63289eba5976fb953fa1` |
| `B2L2_C001H001S0001000544.jpg` | `cb71c7890757643e3558c3c742c546e2f68a21bb951687f2365c559096066dff` |
| `B2L2_C001H001S0001000545.jpg` | `b28ba038393c4491a4fa2b54ffb41feff8ff306e3e7f9423789a9813b735f40a` |
| `B2L2_C001H001S0001000546.jpg` | `19e0ce519eb65c4c6525a104c26ae01794e4d3fb6b2995b879d18382e3ba2488` |
| `B2L2_C001H001S0001000547.jpg` | `b38e0915b80dbb395a078bdd9cff33a5bb600783fae112b0243485fdfb95acf5` |
| `B2L2_C001H001S0001000548.jpg` | `d3c179356ad7941645d7618b532e2e27c613c8e0e25afa35c9bda297db750bc6` |
| `B2L2_C001H001S0001000549.jpg` | `6113e02a4893cefba3ba785239defcb63b242b4666339a0fc1d12c8fcdf5f7f4` |
| `B2L2_C001H001S0001000550.jpg` | `ea48186e03ec7b625e9713dcc9cbe6bbaba8aaee4e94bc5acfa589d413e4e7a9` |
| `B2L2_C001H001S0001000551.jpg` | `43146d83f6606c0cd14f17656c9abac2b77434a2db795a318819f81cadd6cf58` |
| `B2L2_C001H001S0001000552.jpg` | `b5ede09d9165cec4dd3128499cd15c8ff601b35b4ee050d266c7d8130f1ba2f9` |
| `B2L2_C001H001S0001000553.jpg` | `6fe2f1f34ae805ae0aebf80f4978235d314e74fbf1375c1847ad9136b00a92e6` |
| `B2L2_C001H001S0001000554.jpg` | `487499bd94a45a11f43d3e27db1dfb28ad51261f9914914f90b6cdb9edddb6e6` |
| `B2L2_C001H001S0001000555.jpg` | `6d5e0e8a3c3975acf62e2563da015ebf543edee5750ba2988bdc84fa749c4f44` |
| `B2L2_C001H001S0001000556.jpg` | `6c81130dcf88f71e84e890a2c935951c97868f8302c724e2dc85ee50ffc4053e` |
| `B2L2_C001H001S0001000557.jpg` | `50d9b383d3583efea9c51f7e40cdbccf322d4d119b2ceaffdcb35e81b3bf5773` |
| `B2L2_C001H001S0001000558.jpg` | `c29111d22d7b3b867e1cc933bd58116233194170f61dfc40519a21f1d01ea8af` |
| `B2L2_C001H001S0001000559.jpg` | `6158f2d9c60a0ea02aa0ed2bcc1e30fa031c19f7c38a84474735d88b0141e0c8` |
| `B2L2_C001H001S0001000560.jpg` | `d17f8b1d84315fb75bea6255b31c43fc21c7890ccf200e926c7a21ebe6dbf8c8` |
| `B2L2_C001H001S0001000561.jpg` | `ca3266e0fc9a34e4dc9199cb328b049ae288e8f546f45258be3feb891e1bf121` |
| `B2L2_C001H001S0001000562.jpg` | `9fe4e6722df29aba875f0f2e361e1eaa733ecc4a53eb57720f23537d8dfffc22` |
| `B2L2_C001H001S0001000563.jpg` | `a47c41faee4b75d2f8daaffc186850fd360e0d93cc931a46c5acc484424f0f41` |
| `B2L2_C001H001S0001000564.jpg` | `00ee51dd764c5d269fbf072b60b685310b8eb429d26850ffb1d1e4f4a96f0470` |
